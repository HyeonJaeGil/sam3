#!/usr/bin/env python3
"""Run SAM3 multi-query text detection on individual frames with no tracking."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    import supervision as sv
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "The demo requires the `supervision` package. Install it before running "
        "this script, for example with `pip install -e \".[notebooks]\"`."
    ) from exc

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def parse_text_queries(values: list[str]) -> list[str]:
    queries = []
    for value in values:
        for item in value.split(","):
            query = item.strip()
            if query:
                queries.append(query)
    if not queries:
        raise ValueError("at least one non-empty text query must be provided")
    return queries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect multiple text queries independently on sampled frames from an "
            "image directory. No tracking is performed across frames."
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
        nargs="+",
        required=True,
        help="One or more text queries to detect.",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Start processing from this frame index. Default: 0.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=20,
        help=(
            "Maximum number of sampled frames to save. With --frame-stride > 1, "
            "this means N strided frames."
        ),
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Sample every Nth frame. Default: 1.",
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
        help="Optional GPU ids. Detector-only mode uses the first provided GPU id.",
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
        help="Accepted for CLI compatibility; not used in detector-only mode.",
    )
    parser.add_argument(
        "--async-loading-frames",
        action="store_true",
        help="Accepted for CLI compatibility; not used in detector-only mode.",
    )
    parser.add_argument(
        "--limit-initial-load-to-window",
        action="store_true",
        help="Accepted for CLI compatibility; frame selection is always windowed locally.",
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


def select_frames(
    frame_paths: list[Path],
    frame_index: int,
    max_frames: int,
    frame_stride: int,
) -> list[Path]:
    max_dense_end = min(
        len(frame_paths),
        frame_index + (max_frames - 1) * frame_stride + 1,
    )
    return frame_paths[frame_index:max_dense_end:frame_stride]


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
    xyxy[:, 0] = boxes_xywh[:, 0]
    xyxy[:, 1] = boxes_xywh[:, 1]
    xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2]
    xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3]

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


def log_outputs(title: str, outputs: dict) -> None:
    obj_ids = np.asarray(outputs["out_obj_ids"], dtype=np.int32)
    probs = np.asarray(outputs["out_probs"], dtype=np.float32)
    tracker_probs = np.asarray(outputs["out_tracker_probs"], dtype=np.float32)
    boxes_xywh = np.asarray(outputs["out_boxes_xywh"], dtype=np.float32)
    text_queries = np.asarray(outputs.get("out_text_queries", []), dtype=object)
    print(title)
    if len(obj_ids) == 0:
        print("  no detections")
        return
    for obj_id, confidence, tracker_confidence, box_xywh, class_name in zip(
        obj_ids,
        probs,
        tracker_probs,
        boxes_xywh,
        text_queries,
    ):
        x0, y0, w, h = [round(float(value), 2) for value in box_xywh]
        x1 = round(x0 + w, 2)
        y1 = round(y0 + h, 2)
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


def state_to_outputs(state: dict, text_queries: list[str]) -> dict:
    boxes_xyxy = state["boxes"].detach().cpu()
    scores = state["scores"].detach().cpu()
    masks = state["masks"].detach().cpu().squeeze(1).numpy().astype(bool)

    boxes_xywh = boxes_xyxy.clone()
    boxes_xywh[:, 2] -= boxes_xywh[:, 0]
    boxes_xywh[:, 3] -= boxes_xywh[:, 1]

    obj_ids = np.arange(len(scores), dtype=np.int32)
    if len(scores) == 0:
        query_names = np.empty((0,), dtype=object)
    else:
        query_names = np.full((len(scores),), text_queries[0], dtype=object)
        if len(text_queries) > 1 and "query_indices" in state:
            query_indices = state["query_indices"]
            query_names = np.array([text_queries[int(i)] for i in query_indices], dtype=object)

    return {
        "out_obj_ids": obj_ids,
        "out_probs": scores.numpy().astype(np.float32),
        "out_tracker_probs": scores.numpy().astype(np.float32),
        "out_boxes_xywh": boxes_xywh.numpy().astype(np.float32),
        "out_binary_masks": masks,
        "out_text_queries": query_names,
    }


@torch.inference_mode()
def detect_frame(
    processor: Sam3Processor,
    image_path: Path,
    text_queries: list[str],
) -> dict:
    image = Image.open(image_path).convert("RGB")
    scores_list = []
    boxes_list = []
    masks_list = []
    query_indices = []

    for query_idx, query in enumerate(text_queries):
        state = processor.set_image(image)
        state = processor.set_text_prompt(query, state)
        num_det = len(state["scores"])
        if num_det == 0:
            continue
        scores_list.append(state["scores"].detach().cpu())
        boxes_list.append(state["boxes"].detach().cpu())
        masks_list.append(state["masks"].detach().cpu())
        query_indices.extend([query_idx] * num_det)

    if scores_list:
        state = {
            "scores": torch.cat(scores_list, dim=0),
            "boxes": torch.cat(boxes_list, dim=0),
            "masks": torch.cat(masks_list, dim=0),
            "query_indices": np.asarray(query_indices, dtype=np.int64),
        }
    else:
        width, height = image.size
        state = {
            "scores": torch.zeros(0, dtype=torch.float32),
            "boxes": torch.zeros(0, 4, dtype=torch.float32),
            "masks": torch.zeros(0, 1, height, width, dtype=torch.bool),
            "query_indices": np.zeros(0, dtype=np.int64),
        }
    return state_to_outputs(state, text_queries)


def main() -> None:
    args = parse_args()
    args.text_query = parse_text_queries(args.text_query)
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

    selected_frame_paths = select_frames(
        frame_paths=frame_paths,
        frame_index=args.frame_index,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
    )
    if not selected_frame_paths:
        raise RuntimeError("No frames selected after applying frame selection")

    if args.gpus:
        device = f"cuda:{args.gpus[0]}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_sam3_image_model(
        checkpoint_path=str(args.checkpoint) if args.checkpoint else None,
        device=device,
    )
    processor = Sam3Processor(model=model, device=device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for frame_path in selected_frame_paths:
        outputs = detect_frame(processor, frame_path, args.text_query)
        log_outputs(f"Frame {frame_path.stem} detections:", outputs)
        annotated = annotate_frame(frame_path, outputs, args.mask_alpha)
        annotated.save(args.output_dir / frame_path.name)
        saved += 1

    print(
        f"Saved {saved} annotated frame(s) to {args.output_dir} "
        f"for detector-only queries {args.text_query!r}."
    )


if __name__ == "__main__":
    main()
