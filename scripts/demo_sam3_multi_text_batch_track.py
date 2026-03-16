#!/usr/bin/env python3
"""Run SAM3 multi-query text tracking on an image directory."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

try:
    import supervision as sv
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "The demo requires the `supervision` package. Install it before running "
        "this script, for example with `pip install supervision`."
    ) from exc

from sam3.model_builder import build_sam3_video_predictor


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect and track multiple text queries across frames in an image "
            "directory using the SAM3 multi-query session flow."
        )
    )
    parser.add_argument(
        "input_image_dir",
        type=Path,
        help="Directory containing ordered video frames as images.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where annotated frames will be written.",
    )
    parser.add_argument(
        "--text-query",
        action="append",
        required=True,
        help="Text query to detect and track. Repeat for multiple queries.",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Seed frame index used for the initial text prompts. Default: 0.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=20,
        help=(
            "Maximum number of sampled frames to save, including the seed frame. "
            "With --frame-stride > 1, this means N strided frames."
        ),
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Sample every Nth frame for tracking and caching. Default: 1.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional local checkpoint path.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="*",
        default=None,
        help="Optional GPU ids passed to the SAM3 multi-GPU predictor.",
    )
    parser.add_argument(
        "--mask-alpha",
        type=float,
        default=0.45,
        help="Mask overlay alpha in [0, 1]. Default: 0.45.",
    )
    parser.add_argument(
        "--offload-video-to-cpu",
        action="store_true",
        help="Keep input frames in CPU memory while running the model on CUDA.",
    )
    parser.add_argument(
        "--async-loading-frames",
        action="store_true",
        help="Load image-directory frames lazily instead of materializing them all at once.",
    )
    parser.add_argument(
        "--limit-initial-load-to-window",
        action="store_true",
        help=(
            "Only load the dense frame window needed for this run. With "
            "--frame-stride > 1, the dense window spans enough frames to produce "
            "--max-frames sampled frames."
        ),
    )
    return parser.parse_args()


def list_frames(image_dir: Path) -> list[Path]:
    frames = [
        p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    if not frames:
        raise FileNotFoundError(f"No image frames found in {image_dir}")

    def sort_key(path: Path):
        stem = path.stem
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    return sorted(frames, key=sort_key)


def outputs_to_detections(
    outputs: dict, image_size: tuple[int, int]
) -> tuple[sv.Detections, list[str]]:
    width, height = image_size
    obj_ids = np.asarray(outputs["out_obj_ids"], dtype=np.int32)
    probs = np.asarray(outputs["out_probs"], dtype=np.float32)
    tracker_probs = np.asarray(outputs["out_tracker_probs"], dtype=np.float32)
    boxes_xywh = np.asarray(outputs["out_boxes_xywh"], dtype=np.float32)
    masks = np.asarray(outputs["out_binary_masks"], dtype=bool)
    text_queries = np.asarray(outputs.get("out_text_queries", []), dtype=object)

    if len(obj_ids) == 0:
        detections = sv.Detections(
            xyxy=np.zeros((0, 4), dtype=np.float32),
            mask=np.zeros((0, height, width), dtype=bool),
            confidence=np.zeros((0,), dtype=np.float32),
            class_id=np.zeros((0,), dtype=np.int32),
            data={"class_name": np.empty((0,), dtype=object)},
        )
        return detections, []

    xyxy = np.empty((len(boxes_xywh), 4), dtype=np.float32)
    xyxy[:, 0] = boxes_xywh[:, 0] * width
    xyxy[:, 1] = boxes_xywh[:, 1] * height
    xyxy[:, 2] = (boxes_xywh[:, 0] + boxes_xywh[:, 2]) * width
    xyxy[:, 3] = (boxes_xywh[:, 1] + boxes_xywh[:, 3]) * height

    class_names = np.array(
        [query if query is not None else "<unknown>" for query in text_queries],
        dtype=object,
    )
    labels = [
        f"{name}#{int(obj_id)} ({float(tracker_score):.2f})"
        for name, obj_id, tracker_score in zip(class_names, obj_ids, tracker_probs)
    ]
    detections = sv.Detections(
        xyxy=xyxy,
        mask=masks,
        confidence=probs,
        class_id=obj_ids,
        data={"class_name": class_names},
    )
    return detections, labels


def log_outputs(title: str, outputs: dict, image_size: tuple[int, int]) -> None:
    detections, _ = outputs_to_detections(outputs, image_size)
    tracker_probs = np.asarray(outputs["out_tracker_probs"], dtype=np.float32)
    class_names = np.asarray(detections.data.get("class_name", []), dtype=object)
    print(title)
    if len(detections) == 0:
        print("  no detections")
        return
    for obj_id, confidence, tracker_confidence, xyxy, class_name in zip(
        detections.class_id,
        detections.confidence,
        tracker_probs,
        detections.xyxy,
        class_names,
    ):
        x0, y0, x1, y1 = [round(float(value), 2) for value in xyxy]
        print(
            f"  label={class_name} id={int(obj_id)} score={float(confidence):.4f} "
            f"tracker_score={float(tracker_confidence):.4f} box=[{x0}, {y0}, {x1}, {y1}]"
        )


def annotate_frame(image_path: Path, outputs: dict, mask_alpha: float) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    detections, labels = outputs_to_detections(outputs, image.size)

    mask_annotator = sv.MaskAnnotator(opacity=mask_alpha)
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated = mask_annotator.annotate(scene=image_np.copy(), detections=detections)
    annotated = box_annotator.annotate(scene=annotated, detections=detections)
    annotated = label_annotator.annotate(
        scene=annotated, detections=detections, labels=labels
    )
    return Image.fromarray(annotated)


def save_outputs(
    frame_paths: list[Path],
    propagated_results: Iterable[dict],
    output_dir: Path,
    mask_alpha: float,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for result in propagated_results:
        frame_idx = int(result["frame_index"])
        outputs = result["outputs"]
        if frame_idx < 0 or frame_idx >= len(frame_paths):
            continue
        image_size = Image.open(frame_paths[frame_idx]).size
        log_outputs(f"Frame {frame_idx} detections:", outputs, image_size)
        annotated = annotate_frame(frame_paths[frame_idx], outputs, mask_alpha)
        annotated.save(output_dir / frame_paths[frame_idx].name)
        saved += 1
    return saved


def main() -> None:
    args = parse_args()
    frame_paths = list_frames(args.input_image_dir)

    if args.frame_index < 0 or args.frame_index >= len(frame_paths):
        raise IndexError(
            f"--frame-index must be in [0, {len(frame_paths) - 1}], got {args.frame_index}"
        )
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")
    if not 0.0 <= args.mask_alpha <= 1.0:
        raise ValueError("--mask-alpha must be between 0 and 1")

    predictor = build_sam3_video_predictor(
        checkpoint_path=str(args.checkpoint) if args.checkpoint else None,
        gpus_to_use=args.gpus,
        async_loading_frames=args.async_loading_frames,
    )
    session_id = None
    selected_frame_paths = frame_paths
    prompt_frame_index = args.frame_index
    max_frame_num_to_track = (
        min(args.max_frames, len(frame_paths) - args.frame_index) - 1
    )
    start_request = {
        "type": "start_session",
        "resource_path": str(args.input_image_dir),
        "offload_video_to_cpu": args.offload_video_to_cpu,
    }
    max_dense_end = min(
        len(frame_paths),
        args.frame_index + (args.max_frames - 1) * args.frame_stride + 1,
    )

    if args.frame_stride > 1:
        dense_window = frame_paths[args.frame_index:max_dense_end]
        selected_frame_paths = dense_window[:: args.frame_stride]
        if not selected_frame_paths:
            raise RuntimeError("No frames selected after applying --frame-stride")
        prompt_frame_index = 0
        max_frame_num_to_track = max(0, len(selected_frame_paths) - 1)
        start_request["resource_path"] = [str(path) for path in selected_frame_paths]
    elif args.limit_initial_load_to_window:
        frames_to_load = max_dense_end - args.frame_index
        selected_frame_paths = frame_paths[
            args.frame_index : args.frame_index + frames_to_load
        ]
        prompt_frame_index = 0
        max_frame_num_to_track = max(0, len(selected_frame_paths) - 1)
        start_request["frame_start_index"] = args.frame_index
        start_request["max_frames_to_load"] = frames_to_load

    try:
        start_response = predictor.handle_request(request=start_request)
        session_id = start_response["session_id"]

        prompt_response = predictor.handle_request(
            request={
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": prompt_frame_index,
                "text": args.text_query,
            }
        )
        prompt_image_size = Image.open(selected_frame_paths[prompt_frame_index]).size
        log_outputs(
            f"Prompt frame {prompt_frame_index} detections:",
            prompt_response["outputs"],
            prompt_image_size,
        )

        stream = predictor.handle_stream_request(
            request={
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": "forward",
                "start_frame_index": prompt_frame_index,
                "max_frame_num_to_track": max_frame_num_to_track,
            }
        )
        saved = save_outputs(
            frame_paths=selected_frame_paths,
            propagated_results=stream,
            output_dir=args.output_dir,
            mask_alpha=args.mask_alpha,
        )
        print(
            f"Saved {saved} annotated frame(s) to {args.output_dir} "
            f"for queries {args.text_query!r}."
        )
    finally:
        if session_id is not None:
            predictor.handle_request(
                request={"type": "close_session", "session_id": session_id}
            )
        predictor.shutdown()


if __name__ == "__main__":
    main()
