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

"""Utility functions for Nemotron-Parse model output.

Provides raw-output parsing, Picture/Caption reordering, and interleaved-row
construction used by the preprocess / postprocess stages.  General PDF
helpers live in :mod:`nemo_curator.stages.interleaved.pdf.utils`.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from nemo_curator.stages.interleaved.pdf.utils import (
    DEFAULT_MIN_CROP_PX,
    build_canvas,
    crop_to_bbox,
    image_to_bytes,
)

if TYPE_CHECKING:
    from PIL import Image


def parse_nemotron_output(raw_text: str) -> list[dict[str, Any]]:
    """Parse Nemotron-Parse raw output into structured elements.

    Each element is a dict with keys ``class``, ``text``, and ``bbox``
    (normalized [x1, y1, x2, y2]).
    """
    elements: list[dict[str, Any]] = []
    pattern = re.compile(
        r"<x_([\d.]+)><y_([\d.]+)>"
        r"(.*?)"
        r"<x_([\d.]+)><y_([\d.]+)>"
        r"<class_([^>]+)>",
        re.DOTALL,
    )
    for match in pattern.finditer(raw_text):
        x1, y1 = float(match.group(1)), float(match.group(2))
        x2, y2 = float(match.group(4)), float(match.group(5))
        cls = match.group(6)
        text = re.sub(r"<[^>]+>", "", match.group(3)).strip()
        bbox = [x1, y1, x2, y2]
        if text or cls == "Picture":
            elements.append({"class": cls, "text": text, "bbox": bbox})

    if not elements and raw_text.strip():
        cleaned = re.sub(r"<[^>]+>", "", raw_text).strip()
        if cleaned:
            elements.append({"class": "Text", "text": cleaned, "bbox": None})
    return elements


def _bbox_center_y(bbox: list[float] | None) -> float:
    if bbox is None:
        return 0.0
    return (bbox[1] + bbox[3]) / 2.0


def _pair_pictures_and_captions(
    floaters: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Group each Caption with its nearest Picture by bbox proximity."""
    pictures = [(i, f) for i, f in enumerate(floaters) if f["class"] == "Picture"]
    captions = [(i, f) for i, f in enumerate(floaters) if f["class"] == "Caption"]

    pic_taken: set[int] = set()
    cap_to_pic: dict[int, int] = {}

    for ci, cap in captions:
        cap_y = _bbox_center_y(cap.get("bbox"))
        best_pi = None
        best_dist = float("inf")
        for pi, pic in pictures:
            if pi in pic_taken:
                continue
            dist = abs(_bbox_center_y(pic.get("bbox")) - cap_y)
            if dist < best_dist:
                best_dist = dist
                best_pi = pi
        if best_pi is not None:
            cap_to_pic[ci] = best_pi
            pic_taken.add(best_pi)

    groups: list[list[dict[str, Any]]] = []
    used_caps: set[int] = set(cap_to_pic.keys())

    for pi, pic in pictures:
        group = [pic]
        matched_cap = [(ci, cap) for ci, cap in captions if cap_to_pic.get(ci) == pi]
        for _ci, cap in matched_cap:
            group.append(cap)
        groups.append(group)

    for ci, cap in captions:
        if ci not in used_caps:
            groups.append([cap])

    groups.sort(key=lambda g: _bbox_center_y(g[0].get("bbox")))
    return groups


def interleave_floaters(
    anchored: list[dict[str, Any]],
    floaters: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Insert floater elements (Pictures/Captions) next to the closest anchor.

    Anchored elements keep their original model output order.  Pictures and
    Captions are first paired, then each pair is inserted after the anchored
    element whose bbox center-y is closest.

    This is needed for Nemotron-Parse v1.1 which emits Picture/Caption at the
    end of the page output rather than in reading order.  v1.2+ outputs them
    in correct reading order so this reordering can be skipped.
    """
    if not floaters:
        return list(anchored)
    if not anchored:
        result: list[dict[str, Any]] = []
        for group in _pair_pictures_and_captions(floaters):
            result.extend(group)
        return result

    groups = _pair_pictures_and_captions(floaters)
    anchor_ys = [_bbox_center_y(e.get("bbox")) for e in anchored]

    insert_map: dict[int, list[list[dict[str, Any]]]] = {}
    for group in groups:
        gy = _bbox_center_y(group[0].get("bbox"))
        best_idx = min(range(len(anchor_ys)), key=lambda i: abs(anchor_ys[i] - gy))
        insert_map.setdefault(best_idx, []).append(group)

    for groups_at_idx in insert_map.values():
        groups_at_idx.sort(key=lambda g: _bbox_center_y(g[0].get("bbox")))

    result = []
    for i, elem in enumerate(anchored):
        result.append(elem)
        if i in insert_map:
            for group in insert_map[i]:
                result.extend(group)
    return result


def build_interleaved_rows(  # noqa: PLR0913
    sample_id: str,
    url: str,
    pdf_name: str,
    page_images: list[Image.Image],
    page_outputs: list[str],
    proc_size: tuple[int, int] = (2048, 1664),
    reorder_floaters: bool = True,
    min_crop_px: int = DEFAULT_MIN_CROP_PX,
) -> list[dict[str, Any]]:
    """Convert Nemotron-Parse page outputs into interleaved-schema rows.

    Args:
        sample_id: Unique identifier for this PDF.
        url: Source URL of the PDF.
        pdf_name: Original PDF filename.
        page_images: Rendered page images.
        page_outputs: Raw Nemotron-Parse output per page.
        proc_size: Model processor's expected (height, width).
        reorder_floaters: If True, re-insert Pictures/Captions in reading order
            (needed for v1.1).  If False, preserve raw model output order (v1.2+).
        min_crop_px: Minimum pixel dimension for image crops.
    """
    rows: list[dict[str, Any]] = [
        {
            "sample_id": sample_id,
            "position": -1,
            "modality": "metadata",
            "content_type": "application/json",
            "text_content": json.dumps({"url": url, "pdf_name": pdf_name, "num_pages": len(page_images)}),
            "binary_content": None,
            "source_ref": None,
            "url": url,
            "page_number": None,
            "pdf_name": pdf_name,
            "element_class": None,
        }
    ]

    position = 0
    for page_num, (page_img, raw_output) in enumerate(zip(page_images, page_outputs, strict=True)):
        canvas = build_canvas(page_img, proc_size)
        elements = parse_nemotron_output(raw_output)

        if reorder_floaters:
            anchored = [e for e in elements if e["class"] not in ("Picture", "Caption")]
            floaters = [e for e in elements if e["class"] in ("Picture", "Caption")]
            ordered = interleave_floaters(anchored, floaters)
        else:
            ordered = elements

        for elem in ordered:
            cls = elem["class"]
            bbox = elem.get("bbox")
            source_ref = json.dumps({"page": page_num, "bbox": bbox})

            if cls == "Picture":
                modality, content_type = "image", "image/png"
                cropped = crop_to_bbox(canvas, bbox, proc_size, min_crop_px)
                if cropped is None:
                    continue
                binary, text = image_to_bytes(cropped), elem.get("text")
            elif cls == "Table":
                modality, content_type = "table", "text/markdown"
                binary, text = None, elem["text"]
            else:
                modality, content_type = "text", "text/markdown"
                binary, text = None, elem["text"]

            rows.append(
                {
                    "sample_id": sample_id,
                    "position": position,
                    "modality": modality,
                    "content_type": content_type,
                    "text_content": text,
                    "binary_content": binary,
                    "source_ref": source_ref,
                    "url": url,
                    "page_number": page_num,
                    "pdf_name": pdf_name,
                    "element_class": cls,
                }
            )
            position += 1

    return rows
