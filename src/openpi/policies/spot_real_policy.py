import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_spot_real_example() -> dict:
    """Creates a random input example for the Spot real policy."""
    return {
        "observation/state": np.random.rand(8).astype(np.float32),
        "observation/base_image": np.random.randint(256, size=(240, 320, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(240, 320, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(255 * image, 0, 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class SpotRealInputs(transforms.DataTransformFn):
    """Maps a LeRobot-converted Spot real sample to the pi model input format."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"], dtype=np.float32)
        base_image = _parse_image(data["observation/base_image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class SpotRealOutputs(transforms.DataTransformFn):
    """Restores the action dimensionality used by the Spot real dataset."""

    action_dim: int = 10

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., : self.action_dim])}
