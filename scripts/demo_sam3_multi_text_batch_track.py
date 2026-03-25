#!/usr/bin/env python3
"""
Run schedule-driven SAM3 multi-text tracking on an RGB frame directory.

Expected JSON format
--------------------
The JSON file must contain a `prompt_schedule` field. Each schedule entry applies
from its `frame_idx` until the next scheduled frame.

Minimal example:
{
  "prompt_schedule": [
    {"frame_idx": 0, "text_prompts": ["chair", "table"]},
    {"frame_idx": 100, "text_prompts": ["drawer", "chair"]}
  ]
}

This means:
  - frames 0..99: allow new detections for "chair" and "table"
  - frames 100..end: allow new detections for "drawer" and "chair"
  - existing tracked objects from earlier prompts continue to be tracked even if
    their prompt is no longer active for new detections

Optional top-level fields:
{
  "prompt_schedule": [...],
  "start_frame_idx": 0,
  "max_frame_num_to_track": null,
  "propagation_direction": "forward",
  "allow_new_detections": true,
  "offload_video_to_cpu": false,
  "offload_state_to_cpu": false,
  "frame_start_index": 0,
  "max_frames_to_load": null
}

Notes:
  - `text_prompts` can be a string or a list of strings.
  - `start_frame_idx` defaults to the earliest scheduled frame.
  - `max_frame_num_to_track` defaults to tracking until the end of the loaded clip.
  - `propagation_direction` can be "forward", "backward", or "both".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

try:
    import supervision as sv
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "The demo requires the `supervision` package. Install it before running "
        "this script, for example with `pip install -e \".[notebooks]\"`."
    ) from exc

from sam3.model_builder import build_sam3_scheduled_video_model
from sam3.train.masks_ops import rle_encode


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Track multiple scheduled text prompts across an RGB frame directory "
            "using Sam3VideoInferenceWithScheduledTextPrompts."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""JSON example:
{
  "prompt_schedule": [
    {"frame_idx": 0, "text_prompts": ["chair", "table"]},
    {"frame_idx": 100, "text_prompts": ["drawer", "chair"]}
  ],
  "start_frame_idx": 0,
  "max_frame_num_to_track": null,
  "propagation_direction": "forward",
  "allow_new_detections": true,
  "offload_video_to_cpu": false,
  "offload_state_to_cpu": false
}

Each schedule entry stays active until the next scheduled frame.
""",
    )
    parser.add_argument(
        "input_image_dir",
        type=Path,
        help="Directory containing ordered RGB video frames as images.",
    )
    parser.add_argument(
        "prompt_schedule_json",
        type=Path,
        help="JSON file describing the per-frame text prompt schedule and run options.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where annotated frames and JSON annotations will be written.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional local checkpoint path.",
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
        help="Override the JSON and keep input frames in CPU memory.",
    )
    parser.add_argument(
        "--offload-state-to-cpu",
        action="store_true",
        help="Override the JSON and offload tracker state tensors to CPU memory.",
    )
    parser.add_argument(
        "--async-loading-frames",
        action="store_true",
        help="Load image-directory frames lazily instead of materializing them all at once.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help=(
            "Sample every Nth frame from the input directory before running the "
            "scheduled tracking flow. Default: 1."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help=(
            "Limit the number of frames considered from the effective start frame. "
            "When used with --limit-initial-load-to-window, only this window is loaded."
        ),
    )
    parser.add_argument(
        "--limit-initial-load-to-window",
        action="store_true",
        help=(
            "Only load the dense frame window needed for this run. When combined with "
            "--frame-stride > 1, the dense window spans enough frames to produce the "
            "requested number of sampled frames."
        ),
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile in the scheduled video model.",
    )
    parser.add_argument(
        "--log-memory",
        action="store_true",
        help="Log CUDA/CPU memory usage before, during, and after propagation.",
    )
    parser.add_argument(
        "--empty-cache-every",
        type=int,
        default=0,
        help=(
            "If > 0, call torch.cuda.empty_cache() every N propagated frames and log "
            "memory before/after. This is for diagnosing allocator reserve growth."
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


def load_prompt_schedule_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    if not isinstance(config, dict):
        raise TypeError("prompt schedule JSON must be a top-level object")
    if "prompt_schedule" not in config:
        raise KeyError("prompt schedule JSON must contain `prompt_schedule`")

    prompt_schedule = config["prompt_schedule"]
    if not isinstance(prompt_schedule, list) or len(prompt_schedule) == 0:
        raise ValueError("`prompt_schedule` must be a non-empty list")

    normalized_schedule = []
    for entry in prompt_schedule:
        if not isinstance(entry, dict):
            raise TypeError("each prompt schedule entry must be an object")
        if "frame_idx" not in entry:
            raise KeyError("each prompt schedule entry must contain `frame_idx`")
        if "text_prompts" not in entry and "text_prompt" not in entry:
            raise KeyError(
                "each prompt schedule entry must contain `text_prompts` or `text_prompt`"
            )
        text_prompts = entry.get("text_prompts", entry.get("text_prompt"))
        normalized_schedule.append(
            {
                "frame_idx": entry["frame_idx"],
                "text_prompts": text_prompts,
            }
        )

    config["prompt_schedule"] = normalized_schedule
    return config


def remap_schedule_for_sampled_frames(
    prompt_schedule: list[dict],
    selected_dense_indices: list[int],
) -> list[dict]:
    dense_to_sampled = {
        dense_frame_idx: sampled_idx
        for sampled_idx, dense_frame_idx in enumerate(selected_dense_indices)
    }
    remapped_schedule = []
    for entry in prompt_schedule:
        dense_frame_idx = entry["frame_idx"]
        if dense_frame_idx in dense_to_sampled:
            remapped_schedule.append(
                {
                    "frame_idx": dense_to_sampled[dense_frame_idx],
                    "text_prompts": entry["text_prompts"],
                }
            )
    if not remapped_schedule:
        raise ValueError(
            "No prompt schedule entries fall on sampled frames after applying --frame-stride"
        )
    return remapped_schedule


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


def convert_frame_stats(frame_stats):
    if frame_stats is None:
        return None
    if isinstance(frame_stats, dict):
        return {key: convert_frame_stats(value) for key, value in frame_stats.items()}
    if isinstance(frame_stats, (list, tuple)):
        return [convert_frame_stats(value) for value in frame_stats]
    if isinstance(frame_stats, np.ndarray):
        return frame_stats.tolist()
    if isinstance(frame_stats, np.generic):
        return frame_stats.item()
    return frame_stats


def collect_memory_snapshot(tag: str, frame_idx: int | None = None) -> dict:
    snapshot = {
        "time": time.time(),
        "tag": tag,
        "frame_index": frame_idx,
    }
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        snapshot.update(
            {
                "cuda_device": int(device),
                "cuda_memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "cuda_memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
                "cuda_max_memory_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device)
                ),
                "cuda_max_memory_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(device)
                ),
                "cuda_free_bytes": int(free_bytes),
                "cuda_total_bytes": int(total_bytes),
            }
        )
    else:
        snapshot["cuda_device"] = None
    return snapshot


def print_memory_snapshot(snapshot: dict) -> None:
    if snapshot["cuda_device"] is None:
        print(f"[memory] {snapshot['tag']}: CUDA unavailable")
        return
    allocated_mib = snapshot["cuda_memory_allocated_bytes"] / 1024**2
    reserved_mib = snapshot["cuda_memory_reserved_bytes"] / 1024**2
    max_allocated_mib = snapshot["cuda_max_memory_allocated_bytes"] / 1024**2
    max_reserved_mib = snapshot["cuda_max_memory_reserved_bytes"] / 1024**2
    free_mib = snapshot["cuda_free_bytes"] / 1024**2
    total_mib = snapshot["cuda_total_bytes"] / 1024**2
    frame_part = (
        f" frame={snapshot['frame_index']}" if snapshot["frame_index"] is not None else ""
    )
    print(
        f"[memory] {snapshot['tag']}{frame_part}: "
        f"allocated={allocated_mib:.1f} MiB "
        f"reserved={reserved_mib:.1f} MiB "
        f"max_allocated={max_allocated_mib:.1f} MiB "
        f"max_reserved={max_reserved_mib:.1f} MiB "
        f"free={free_mib:.1f}/{total_mib:.1f} MiB"
    )


def append_memory_snapshot(memory_log_path: Path | None, snapshot: dict) -> None:
    if memory_log_path is None:
        return
    with memory_log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(snapshot) + "\n")


def collect_frame_debug_stats(
    frame_idx: int,
    outputs: dict | None,
    active_text_prompts: list[str] | None = None,
) -> dict:
    if outputs is None:
        output_object_count = 0
        text_query_count = 0
    else:
        output_object_count = int(len(outputs.get("out_obj_ids", [])))
        text_query_count = int(len(outputs.get("out_text_queries", [])))
    return {
        "time": time.time(),
        "tag": "frame_debug",
        "frame_index": frame_idx,
        "output_object_count": output_object_count,
        "output_text_query_count": text_query_count,
        "active_text_prompt_count": (
            0 if active_text_prompts is None else len(active_text_prompts)
        ),
        "active_text_prompts": [] if active_text_prompts is None else active_text_prompts,
    }


def serialize_outputs(frame_idx: int, image_path: Path, outputs: dict) -> dict:
    masks = np.asarray(outputs["out_binary_masks"], dtype=bool)
    if masks.size == 0:
        masks_rle = []
    else:
        masks_rle = [
            {"counts": item["counts"], "size": item["size"]}
            for item in rle_encode(torch.from_numpy(masks).to(dtype=torch.bool))
        ]

    return {
        "frame_index": frame_idx,
        "image_path": str(image_path),
        "out_obj_ids": np.asarray(outputs["out_obj_ids"], dtype=np.int64).tolist(),
        "out_probs": np.asarray(outputs["out_probs"], dtype=np.float32).tolist(),
        "out_tracker_probs": np.asarray(
            outputs["out_tracker_probs"], dtype=np.float32
        ).tolist(),
        "out_boxes_xywh": np.asarray(
            outputs["out_boxes_xywh"], dtype=np.float32
        ).tolist(),
        "out_text_query_indices": np.asarray(
            outputs.get("out_text_query_indices", []), dtype=np.int64
        ).tolist(),
        "out_text_queries": [
            None if query is None else str(query)
            for query in np.asarray(outputs.get("out_text_queries", []), dtype=object)
        ],
        "out_binary_masks_rle": masks_rle,
        "frame_stats": convert_frame_stats(outputs.get("frame_stats")),
    }


def save_outputs(
    frame_paths: list[Path],
    propagated_results,
    output_dir: Path,
    mask_alpha: float,
    memory_log_path: Path | None = None,
    active_prompt_lookup=None,
    empty_cache_every: int = 0,
) -> int:
    annotated_dir = output_dir / "annotated_frames"
    annotation_dir = output_dir / "annotations"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for frame_idx, outputs in propagated_results:
        active_text_prompts = (
            None if active_prompt_lookup is None else active_prompt_lookup(frame_idx)
        )
        frame_debug = collect_frame_debug_stats(
            frame_idx,
            outputs,
            active_text_prompts=active_text_prompts,
        )
        append_memory_snapshot(memory_log_path, frame_debug)
        memory_snapshot = collect_memory_snapshot("after_frame", frame_idx=frame_idx)
        print_memory_snapshot(memory_snapshot)
        append_memory_snapshot(memory_log_path, memory_snapshot)
        if (
            empty_cache_every > 0
            and torch.cuda.is_available()
            and (frame_idx + 1) % empty_cache_every == 0
        ):
            before_empty = collect_memory_snapshot(
                "before_empty_cache", frame_idx=frame_idx
            )
            print_memory_snapshot(before_empty)
            append_memory_snapshot(memory_log_path, before_empty)
            torch.cuda.empty_cache()
            after_empty = collect_memory_snapshot(
                "after_empty_cache", frame_idx=frame_idx
            )
            print_memory_snapshot(after_empty)
            append_memory_snapshot(memory_log_path, after_empty)
        if outputs is None or frame_idx < 0 or frame_idx >= len(frame_paths):
            continue
        image_path = frame_paths[frame_idx]
        image_size = Image.open(image_path).size
        log_outputs(f"Frame {frame_idx} detections:", outputs, image_size)

        annotated = annotate_frame(image_path, outputs, mask_alpha)
        annotated.save(annotated_dir / image_path.name)

        annotation_payload = serialize_outputs(frame_idx, image_path, outputs)
        with (annotation_dir / f"{image_path.stem}.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(annotation_payload, f, indent=2)
        saved += 1
    return saved


def main() -> None:
    args = parse_args()
    frame_paths = list_frames(args.input_image_dir)

    if not 0.0 <= args.mask_alpha <= 1.0:
        raise ValueError("--mask-alpha must be between 0 and 1")
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if args.empty_cache_every < 0:
        raise ValueError("--empty-cache-every must be non-negative")

    config = load_prompt_schedule_config(args.prompt_schedule_json)
    prompt_schedule = config["prompt_schedule"]
    earliest_prompt_frame = min(entry["frame_idx"] for entry in prompt_schedule)
    start_frame_idx = config.get("start_frame_idx", earliest_prompt_frame)
    max_frame_num_to_track = config.get("max_frame_num_to_track", None)
    propagation_direction = config.get("propagation_direction", "forward")
    allow_new_detections = config.get("allow_new_detections", None)
    offload_video_to_cpu = args.offload_video_to_cpu or config.get(
        "offload_video_to_cpu", False
    )
    offload_state_to_cpu = args.offload_state_to_cpu or config.get(
        "offload_state_to_cpu", False
    )
    frame_start_index = config.get("frame_start_index", 0)
    max_frames_to_load = config.get("max_frames_to_load", None)

    if propagation_direction not in {"forward", "backward", "both"}:
        raise ValueError(
            "propagation_direction must be one of: forward, backward, both"
        )
    if start_frame_idx < 0 or start_frame_idx >= len(frame_paths):
        raise IndexError(
            f"start_frame_idx must be in [0, {len(frame_paths) - 1}], got {start_frame_idx}"
        )
    for entry in prompt_schedule:
        frame_idx = entry["frame_idx"]
        if frame_idx < 0 or frame_idx >= len(frame_paths):
            raise IndexError(
                f"prompt schedule frame_idx must be in [0, {len(frame_paths) - 1}], got {frame_idx}"
            )

    effective_start_dense_idx = start_frame_idx
    if args.max_frames is None:
        requested_sample_count = None
    else:
        requested_sample_count = args.max_frames

    if args.frame_stride == 1:
        if requested_sample_count is None:
            selected_dense_indices = list(range(effective_start_dense_idx, len(frame_paths)))
        else:
            selected_dense_indices = list(
                range(
                    effective_start_dense_idx,
                    min(len(frame_paths), effective_start_dense_idx + requested_sample_count),
                )
            )
    else:
        max_dense_end = len(frame_paths)
        if requested_sample_count is not None:
            max_dense_end = min(
                len(frame_paths),
                effective_start_dense_idx
                + (requested_sample_count - 1) * args.frame_stride
                + 1,
            )
        selected_dense_indices = list(
            range(effective_start_dense_idx, max_dense_end, args.frame_stride)
        )

    if not selected_dense_indices:
        raise RuntimeError("No frames selected after applying frame window/stride")

    selected_frame_paths = [frame_paths[idx] for idx in selected_dense_indices]
    remapped_prompt_schedule = remap_schedule_for_sampled_frames(
        prompt_schedule, selected_dense_indices
    )
    remapped_start_frame_idx = min(
        entry["frame_idx"] for entry in remapped_prompt_schedule
    )
    if requested_sample_count is None:
        remapped_max_frame_num_to_track = max(0, len(selected_frame_paths) - 1)
    else:
        remapped_max_frame_num_to_track = min(
            max(0, requested_sample_count - 1),
            max(0, len(selected_frame_paths) - 1),
        )
    if max_frame_num_to_track is not None:
        remapped_max_frame_num_to_track = min(
            remapped_max_frame_num_to_track,
            max_frame_num_to_track,
        )

    model = build_sam3_scheduled_video_model(
        checkpoint_path=str(args.checkpoint) if args.checkpoint else None,
        compile=args.compile,
    )

    memory_log_path = None
    if args.log_memory:
        memory_log_path = args.output_dir / "memory_usage.jsonl"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if memory_log_path.exists():
            memory_log_path.unlink()
        initial_memory = collect_memory_snapshot("before_init_state")
        print_memory_snapshot(initial_memory)
        append_memory_snapshot(memory_log_path, initial_memory)

    init_resource_path: str | list[str] = str(args.input_image_dir)
    init_frame_start_index = frame_start_index
    init_max_frames_to_load = max_frames_to_load
    if args.frame_stride > 1:
        init_resource_path = [str(path) for path in selected_frame_paths]
        init_frame_start_index = 0
        init_max_frames_to_load = None
    elif args.limit_initial_load_to_window:
        dense_window_end = selected_dense_indices[-1] + 1
        init_frame_start_index = effective_start_dense_idx
        init_max_frames_to_load = dense_window_end - effective_start_dense_idx

    inference_state = model.init_state(
        resource_path=init_resource_path,
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
        async_loading_frames=args.async_loading_frames,
        frame_start_index=init_frame_start_index,
        max_frames_to_load=init_max_frames_to_load,
        allow_new_detections=allow_new_detections,
    )
    if args.log_memory:
        post_init_memory = collect_memory_snapshot("after_init_state")
        print_memory_snapshot(post_init_memory)
        append_memory_snapshot(memory_log_path, post_init_memory)
    model.set_text_prompt_schedule(inference_state, remapped_prompt_schedule)
    if args.log_memory:
        post_schedule_memory = collect_memory_snapshot("after_set_text_prompt_schedule")
        print_memory_snapshot(post_schedule_memory)
        append_memory_snapshot(memory_log_path, post_schedule_memory)

    def active_prompt_lookup(frame_idx: int) -> list[str]:
        active_prompts = []
        for entry in remapped_prompt_schedule:
            if entry["frame_idx"] <= frame_idx:
                active_prompts = entry["text_prompts"]
            else:
                break
        return list(active_prompts)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "input_image_dir": str(args.input_image_dir),
                "prompt_schedule_json": str(args.prompt_schedule_json),
                "prompt_schedule": prompt_schedule,
                "effective_prompt_schedule": remapped_prompt_schedule,
                "start_frame_idx": start_frame_idx,
                "effective_start_frame_idx": remapped_start_frame_idx,
                "max_frame_num_to_track": max_frame_num_to_track,
                "effective_max_frame_num_to_track": remapped_max_frame_num_to_track,
                "propagation_direction": propagation_direction,
                "allow_new_detections": allow_new_detections,
                "offload_video_to_cpu": offload_video_to_cpu,
                "offload_state_to_cpu": offload_state_to_cpu,
                "frame_start_index": frame_start_index,
                "max_frames_to_load": max_frames_to_load,
                "async_loading_frames": args.async_loading_frames,
                "frame_stride": args.frame_stride,
                "requested_max_frames": args.max_frames,
                "limit_initial_load_to_window": args.limit_initial_load_to_window,
                "selected_dense_indices": selected_dense_indices,
                "mask_alpha": args.mask_alpha,
                "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                "compile": args.compile,
                "log_memory": args.log_memory,
                "empty_cache_every": args.empty_cache_every,
            },
            f,
            indent=2,
        )

    if propagation_direction == "both":
        streams = [
            model.propagate_in_video(
                inference_state=inference_state,
                start_frame_idx=remapped_start_frame_idx,
                max_frame_num_to_track=remapped_max_frame_num_to_track,
                reverse=False,
            ),
            model.propagate_in_video(
                inference_state=inference_state,
                start_frame_idx=remapped_start_frame_idx,
                max_frame_num_to_track=remapped_max_frame_num_to_track,
                reverse=True,
            ),
        ]
    else:
        streams = [
            model.propagate_in_video(
                inference_state=inference_state,
                start_frame_idx=remapped_start_frame_idx,
                max_frame_num_to_track=remapped_max_frame_num_to_track,
                reverse=(propagation_direction == "backward"),
            )
        ]

    saved = 0
    for stream in streams:
        if args.log_memory:
            pre_propagation_memory = collect_memory_snapshot("before_propagation")
            print_memory_snapshot(pre_propagation_memory)
            append_memory_snapshot(memory_log_path, pre_propagation_memory)
        saved += save_outputs(
            frame_paths=selected_frame_paths,
            propagated_results=stream,
            output_dir=args.output_dir,
            mask_alpha=args.mask_alpha,
            memory_log_path=memory_log_path,
            active_prompt_lookup=active_prompt_lookup,
            empty_cache_every=args.empty_cache_every,
        )

    if args.log_memory:
        final_memory = collect_memory_snapshot("after_propagation")
        print_memory_snapshot(final_memory)
        append_memory_snapshot(memory_log_path, final_memory)

    print(
        f"Saved {saved} annotated frame(s) and JSON annotation file(s) to {args.output_dir}."
    )


if __name__ == "__main__":
    main()
