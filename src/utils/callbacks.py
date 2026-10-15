"""Callback configuration for the standalone FiSCoRe release."""

from typing import List

from transformers import TrainerCallback


def get_callbacks(train_config, model_config) -> List[TrainerCallback]:
    """Return callbacks configured by the public recipes."""
    del model_config
    if train_config.callbacks:
        raise ValueError(
            "No optional callbacks are bundled with FiSCoRe. "
            "Set `callbacks: []` in the recipe."
        )
    return []
