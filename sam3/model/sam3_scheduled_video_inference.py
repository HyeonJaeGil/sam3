# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

from collections.abc import Mapping, Sequence

from sam3.model.sam3_video_inference import Sam3VideoInference


class Sam3VideoInferenceWithScheduledTextPrompts(Sam3VideoInference):
    """
    A Sam3VideoInference variant that applies text prompts according to a
    frame-indexed schedule. The active prompt set at frame t is the most recent
    schedule entry whose frame index is <= t.

    Scheduled prompts only control which text queries are eligible to spawn new
    detections on a frame. Existing tracked objects continue to propagate until
    normal tracker heuristics remove them.
    """

    def init_state(self, *args, **kwargs):
        inference_state = super().init_state(*args, **kwargs)
        inference_state["text_prompt_schedule"] = {}
        inference_state["text_query_id_registry"] = {}
        return inference_state

    def reset_state(self, inference_state):
        super().reset_state(inference_state)
        inference_state["text_prompt_schedule"] = {}
        inference_state["text_query_id_registry"] = {}

    def set_text_prompt_schedule(
        self,
        inference_state,
        prompt_schedule,
        replace=True,
    ):
        """
        Replace or extend the frame-indexed text prompt schedule.

        Accepted formats:
          - {frame_idx: "chair"}
          - {frame_idx: ["chair", "table"]}
          - [{"frame_idx": 0, "text_prompts": ["chair", "table"]}, ...]
          - [(0, ["chair", "table"]), (100, ["drawer", "chair"])]
        """
        normalized_entries = self._normalize_prompt_schedule(prompt_schedule)
        if replace:
            inference_state["text_prompt_schedule"] = {}
            inference_state["text_query_id_registry"] = {}

        for frame_idx, prompts in normalized_entries:
            inference_state["text_prompt_schedule"][frame_idx] = prompts
            self._register_text_prompts(inference_state, prompts)

        return inference_state

    def set_text_prompts_at_frame(self, inference_state, frame_idx, text_prompts):
        prompts = self._normalize_text_prompts(text_prompts)
        inference_state["text_prompt_schedule"][frame_idx] = prompts
        self._register_text_prompts(inference_state, prompts)
        return inference_state

    def get_text_prompt_schedule(self, inference_state):
        return dict(sorted(inference_state["text_prompt_schedule"].items()))

    def _register_text_prompts(self, inference_state, prompts):
        registry = inference_state["text_query_id_registry"]
        for prompt in prompts:
            if prompt not in registry:
                registry[prompt] = len(registry)

    def _get_active_text_prompts_for_frame(self, inference_state, frame_idx):
        schedule = inference_state.get("text_prompt_schedule", {})
        active_frame_idx = None
        for scheduled_frame_idx in schedule:
            if scheduled_frame_idx <= frame_idx and (
                active_frame_idx is None or scheduled_frame_idx > active_frame_idx
            ):
                active_frame_idx = scheduled_frame_idx
        if active_frame_idx is None:
            return []
        return list(schedule[active_frame_idx])

    def _get_active_text_query_ids_for_frame(self, inference_state, frame_idx):
        registry = inference_state.get("text_query_id_registry", {})
        return [
            registry[prompt]
            for prompt in self._get_active_text_prompts_for_frame(
                inference_state, frame_idx
            )
        ]

    def _run_single_frame_inference(self, inference_state, frame_idx, reverse):
        active_text_prompts = self._get_active_text_prompts_for_frame(
            inference_state, frame_idx
        )
        active_text_query_ids = self._get_active_text_query_ids_for_frame(
            inference_state, frame_idx
        )
        previous_text_prompts = inference_state.get("text_prompts", [])
        previous_active_text_query_ids = inference_state["feature_cache"].get(
            "active_text_query_ids", None
        )

        inference_state["text_prompts"] = active_text_prompts
        inference_state["feature_cache"]["active_text_query_ids"] = (
            active_text_query_ids
        )
        try:
            return super()._run_single_frame_inference(
                inference_state, frame_idx, reverse
            )
        finally:
            inference_state["text_prompts"] = previous_text_prompts
            if previous_active_text_query_ids is None:
                inference_state["feature_cache"].pop("active_text_query_ids", None)
            else:
                inference_state["feature_cache"]["active_text_query_ids"] = (
                    previous_active_text_query_ids
                )

    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
    ):
        if (
            start_frame_idx is None
            and all(out is None for out in inference_state["previous_stages_out"])
            and inference_state.get("text_prompt_schedule")
        ):
            start_frame_idx = min(inference_state["text_prompt_schedule"])
        yield from super().propagate_in_video(
            inference_state=inference_state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=reverse,
        )

    @classmethod
    def _normalize_prompt_schedule(cls, prompt_schedule):
        if isinstance(prompt_schedule, Mapping):
            items = prompt_schedule.items()
        elif isinstance(prompt_schedule, Sequence) and not isinstance(
            prompt_schedule, (str, bytes)
        ):
            items = []
            for item in prompt_schedule:
                if isinstance(item, Mapping):
                    if "frame_idx" not in item:
                        raise KeyError("prompt schedule entry is missing frame_idx")
                    if "text_prompts" in item:
                        prompts = item["text_prompts"]
                    elif "text_prompt" in item:
                        prompts = item["text_prompt"]
                    else:
                        raise KeyError(
                            "prompt schedule entry is missing text_prompts/text_prompt"
                        )
                    items.append((item["frame_idx"], prompts))
                elif (
                    isinstance(item, Sequence)
                    and not isinstance(item, (str, bytes))
                    and len(item) == 2
                ):
                    items.append((item[0], item[1]))
                else:
                    raise TypeError("invalid prompt schedule entry")
        else:
            raise TypeError("prompt_schedule must be a mapping or a sequence")

        normalized_entries = []
        for frame_idx, prompts in items:
            if not isinstance(frame_idx, int) or frame_idx < 0:
                raise ValueError("frame_idx must be a non-negative integer")
            normalized_entries.append((frame_idx, cls._normalize_text_prompts(prompts)))

        normalized_entries.sort(key=lambda x: x[0])
        return normalized_entries
