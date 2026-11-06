"""Pixel-only images and a deterministic crop operation."""

from __future__ import annotations

import base64
import io

from PIL import Image


def encode_png(image: Image.Image) -> bytes:
    # Reconstruct from pixels to strip EXIF, comments, and answer-bearing metadata.
    clean = Image.frombytes("RGB", image.size, image.convert("RGB").tobytes())
    output = io.BytesIO()
    clean.save(output, format="PNG")
    return output.getvalue()


def sanitize_image(data: bytes) -> tuple[bytes, int, int]:
    with Image.open(io.BytesIO(data)) as image:
        return encode_png(image), image.width, image.height


def crop_image(data: bytes, box: list[int], long_side: int = 1024) -> bytes:
    with Image.open(io.BytesIO(data)) as original:
        if not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box):
            raise ValueError("box must contain four integer coordinates")
        left, top, right, bottom = box
        if not (0 <= left < right <= original.width and 0 <= top < bottom <= original.height):
            raise ValueError("box must be nonempty and inside original image bounds")
        crop = original.crop((left, top, right, bottom))
        scale = long_side / max(crop.size)
        size = (max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))
        return encode_png(crop.resize(size, Image.Resampling.LANCZOS))


def image_content(data: bytes) -> dict:
    return {
        "type": "input_image",
        "detail": "high",
        "image_url": "data:image/png;base64," + base64.b64encode(data).decode(),
    }
