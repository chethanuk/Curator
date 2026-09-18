# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""General-purpose PDF utilities shared by the interleaved PDF stages.

Provides page rendering, image serialization, processor-canvas construction,
bbox cropping, and PDF extraction from CC-MAIN zip archives and base64 JSONL.
Nothing here is specific to a particular parsing model.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import zipfile
from typing import Any

from loguru import logger
from PIL import Image

DEFAULT_MIN_CROP_PX = 10
DEFAULT_MAX_PAGES = 50
_CV2_INSTALL_HINT = (
    "opencv-python-headless is required for PDF page rendering. Install with: pip install nemo_curator[cv2]"
)


def _render_scale_to_fit(page: Any, base_scale: float, max_wh: tuple[int, int] | None) -> float:  # noqa: ANN401
    """Return the render scale capped so the output fits within max_wh pixels.

    Mirrors NeMo-Retriever's ``_compute_render_scale_to_fit``: uses the
    standard fit-to-box formula min(target_w/page_w, target_h/page_h) and
    clamps to a minimum of 1e-3 to avoid degenerate renders.  When max_wh is
    None the base_scale is returned unchanged.
    """
    if max_wh is None:
        return base_scale
    target_w, target_h = max_wh
    if target_w <= 0 or target_h <= 0:
        return base_scale
    page_w, page_h = float(page.get_width()), float(page.get_height())
    if page_w <= 0.0 or page_h <= 0.0:
        return base_scale
    fit_scale = max(min(target_w / page_w, target_h / page_h), 1e-3)
    return min(base_scale, fit_scale)


def _bitmap_to_rgb(bitmap: Any) -> Image.Image:  # noqa: ANN401
    """Convert a pypdfium2 bitmap to an RGB PIL image using OpenCV."""
    try:
        import cv2
    except ImportError as e:
        raise ImportError(_CV2_INSTALL_HINT) from e

    arr = bitmap.to_numpy().copy()
    mode = bitmap.mode
    if mode in {"BGRA", "BGRX"}:
        cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA, dst=arr)
        img = Image.fromarray(arr, "RGBA").convert("RGB")
    elif mode == "BGR":
        cv2.cvtColor(arr, cv2.COLOR_BGR2RGB, dst=arr)
        img = Image.fromarray(arr, "RGB")
    else:
        img = Image.fromarray(arr)
        if img.mode != "RGB":
            img = img.convert("RGB")
    return img


def _render_page(doc: Any, page_num: int, base_scale: float, max_size: tuple[int, int] | None) -> Image.Image | None:  # noqa: ANN401
    """Render a single PDF page; returns None on any error."""
    page = None
    bitmap = None
    try:
        page = doc[page_num]
        scale = _render_scale_to_fit(page, base_scale, max_size)
        bitmap = page.render(scale=scale)
        return _bitmap_to_rgb(bitmap)
    except Exception as e:  # noqa: BLE001
        # Log rather than swallow: a missing optional dependency (e.g. cv2) fails
        # here for every page, and the callers treat an empty render as "no pages",
        # so without this the whole pipeline silently produces zero output.
        logger.warning(f"Failed to render page {page_num}: {type(e).__name__}: {e}")
        return None
    finally:
        with contextlib.suppress(Exception):
            if bitmap is not None:
                bitmap.close()
        with contextlib.suppress(Exception):
            if page is not None:
                page.close()


def render_pdf_pages(
    pdf_bytes: bytes,
    dpi: int = 300,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_size: tuple[int, int] | None = (1664, 2048),
) -> list[Image.Image]:
    """Render PDF pages to PIL images using pypdfium2.

    Follows the same pattern as NeMo-Retriever to avoid two pdfium pitfalls:
    1. Explicitly close each page/bitmap after use so the weakref finalizer
       never fires (avoids SIGABRT in _close_impl during GC).
    2. Use ``bitmap.to_numpy().copy()`` + OpenCV for BGR->RGB conversion
       instead of pdfium's ``rev_byteorder`` flag, which triggers a
       non-thread-safe code path in CFX_AggDeviceDriver::GetDIBits().

    The render scale is capped per page via ``_render_scale_to_fit`` so that
    no rendered image exceeds ``max_size`` pixels (default: 1664x2048 =
    Nemotron-Parse processor size).  This bounds the bitmap size regardless of
    how large the PDF page dimensions are, eliminating decompression-bomb
    errors downstream and keeping render time predictable.
    """
    import pypdfium2 as pdfium

    images: list[Image.Image] = []
    doc = None
    try:
        doc = pdfium.PdfDocument(pdf_bytes)
        base_scale = dpi / 72.0
        for page_num in range(min(len(doc), max_pages)):
            img = _render_page(doc, page_num, base_scale, max_size)
            if img is not None:
                images.append(img)
    except Exception as e:  # noqa: BLE001
        # Encrypted and truncated PDFs fail here, in PdfDocument(). Callers treat
        # an empty render as "no pages", so without this the file is dropped with
        # no record of why.
        logger.warning(f"Failed to read PDF document: {type(e).__name__}: {e}")
    with contextlib.suppress(Exception):
        if doc is not None:
            doc.close()
    return images


def image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    """Serialize a PIL Image to bytes."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def build_canvas(page_img: Image.Image, proc_size: tuple[int, int]) -> Image.Image:
    """Replicate the model processor's resize-then-center-pad to build the canvas.

    This lets us crop bboxes directly in the model's coordinate space.
    """
    try:
        import cv2
    except ImportError as e:
        raise ImportError(_CV2_INSTALL_HINT) from e
    import numpy as np

    proc_h, proc_w = proc_size
    orig_w, orig_h = page_img.size
    arr = np.asarray(page_img)

    ar = orig_w / orig_h
    new_h, new_w = orig_h, orig_w
    if new_h > proc_h:
        new_h = proc_h
        new_w = int(new_h * ar)
    if new_w > proc_w:
        new_w = proc_w
        new_h = int(new_w / ar)

    if (new_w, new_h) != (orig_w, orig_h):
        arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_h = max(0, proc_h - arr.shape[0])
    pad_w = max(0, proc_w - arr.shape[1])
    if pad_h > 0 or pad_w > 0:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        arr = np.pad(
            arr,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=255,
        )

    return Image.fromarray(arr)


def crop_to_bbox(
    canvas: Image.Image,
    bbox: list[float] | None,
    proc_size: tuple[int, int],
    min_crop_px: int = DEFAULT_MIN_CROP_PX,
) -> Image.Image | None:
    """Crop a region from the padded canvas using normalized bbox coordinates.

    Returns None if the crop is too small (likely a degenerate bbox).
    """
    if bbox is None:
        return canvas
    proc_h, proc_w = proc_size
    x0 = int(bbox[0] * proc_w)
    y0 = int(bbox[1] * proc_h)
    x1 = int(bbox[2] * proc_w)
    y1 = int(bbox[3] * proc_h)
    x0, x1 = max(0, min(x0, x1)), min(proc_w, max(x0, x1))
    y0, y1 = max(0, min(y0, y1)), min(proc_h, max(y0, y1))
    if x1 - x0 < min_crop_px or y1 - y0 < min_crop_px:
        return None
    return canvas.crop((x0, y0, x1, y1))


# ---------------------------------------------------------------------------
# CC-MAIN PDF zip archive helpers
# ---------------------------------------------------------------------------


def resolve_cc_pdf_zip_path(file_name: str, zip_base_dir: str) -> tuple[str, str]:
    """Map a CC-MAIN PDF filename to its zip archive path and member name.

    The CC-MAIN-2021-31-PDF-UNTRUNCATED dataset organises PDFs into zip
    archives using a two-level numeric grouping::

        <zip_base_dir>/0000-0999/0001.zip  → contains 0001000.pdf .. 0001999.pdf
        <zip_base_dir>/1000-1999/1234.zip  → contains 1234000.pdf .. 1234999.pdf

    Args:
        file_name: PDF filename (e.g. ``"0001234.pdf"``).
        zip_base_dir: Root directory containing the zip archive hierarchy.

    Returns:
        Tuple of (zip_path, member_name).
    """
    num = int(file_name.replace(".pdf", ""))
    zip_num = num // 1000
    group_start = (zip_num // 1000) * 1000
    group_end = group_start + 999
    return (
        os.path.join(zip_base_dir, f"{group_start:04d}-{group_end:04d}", f"{zip_num:04d}.zip"),
        file_name,
    )


def extract_pdf_from_zip(file_name: str, zip_base_dir: str) -> bytes | None:
    """Extract a PDF file from a CC-MAIN zip archive.

    Returns None if extraction fails.
    """
    try:
        zip_path, member = resolve_cc_pdf_zip_path(file_name, zip_base_dir)
    except ValueError:
        return None
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            return zf.read(member)
    except (OSError, KeyError, zipfile.BadZipFile):
        return None


def extract_pdf_from_jsonl(
    jsonl_file: str,
    line_idx: int | None = None,
    byte_offset: int | None = None,
) -> bytes | None:
    """Extract a base64-encoded PDF from a JSONL file.

    Used for GitHub-style PDF datasets where each line contains a JSON object
    with a ``content`` field holding a base64-encoded PDF.

    Prefer ``byte_offset`` (O(1) seek) over ``line_idx`` (O(N) linear scan).
    When both are absent, returns None.
    """
    import base64

    try:
        if byte_offset is not None:
            with open(jsonl_file, "rb") as f:
                f.seek(byte_offset)
                line = f.readline()
                record = json.loads(line)
                return base64.b64decode(record["content"])
        if line_idx is not None:
            with open(jsonl_file) as f:
                for i, line in enumerate(f):
                    if i == line_idx:
                        record = json.loads(line)
                        return base64.b64decode(record["content"])
    except Exception:  # noqa: BLE001
        return None
    return None


def extract_pdfs_from_jsonl_batch(
    jsonl_file: str,
    offsets: list[int],
) -> dict[int, bytes | None]:
    """Extract multiple PDFs from a JSONL file in a single file open.

    Opens the file once and seeks to each byte offset in sorted order.
    Returns a dict mapping byte_offset -> pdf_bytes (None on error).
    """
    import base64

    results: dict[int, bytes | None] = {}
    try:
        with open(jsonl_file, "rb") as f:
            for offset in sorted(offsets):
                result: bytes | None = None
                with contextlib.suppress(Exception):
                    f.seek(offset)
                    line = f.readline()
                    record = json.loads(line)
                    result = base64.b64decode(record["content"])
                results[offset] = result
    except OSError:
        for offset in offsets:
            results[offset] = None
    return results
