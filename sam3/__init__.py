# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

from .model_builder import build_sam3_image_model

__version__ = "0.1.0"

__all__ = ["build_sam3_image_model", "Sam3ImageDetector", "Sam3Prompt"]


def __getattr__(name):
    if name in {"Sam3ImageDetector", "Sam3Prompt"}:
        from .model.sam3_image_detector import Sam3ImageDetector, Sam3Prompt

        return {
            "Sam3ImageDetector": Sam3ImageDetector,
            "Sam3Prompt": Sam3Prompt,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
