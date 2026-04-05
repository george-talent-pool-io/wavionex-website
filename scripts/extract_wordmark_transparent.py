#!/usr/bin/env python3
"""
Conservative transparency for WAVIONEX wordmark crop:
1) Flood from image edges through pixels similar to corner background (dark, near ref color).
2) Remove enclosed blobs of the same background signature (O, counters) that never touch the border.

Does NOT erode letter interiors (no 'muddy' crush / unprotected-core passes).
"""
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def corner_samples(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    return np.vstack(
        [
            rgb[0, 0].astype(np.float64),
            rgb[0, w - 1].astype(np.float64),
            rgb[h - 1, 0].astype(np.float64),
            rgb[h - 1, w - 1].astype(np.float64),
        ]
    )


def min_dist_to_corners(rgb: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Min Euclidean RGB distance to any corner (panel bg varies across the crop)."""
    stack = []
    for c in corners:
        d = rgb.astype(np.float64) - c.reshape(1, 1, 3)
        stack.append(np.sqrt(np.sum(d * d, axis=-1)))
    return np.minimum.reduce(stack)


def luminance(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def edge_flood_background(
    rgba: np.ndarray,
    corners: np.ndarray,
    rgb_thresh: float,
    lum_max_bg: float,
) -> np.ndarray:
    """Mark pixels reachable from edges through 'background-like' cells."""
    h, w = rgba.shape[:2]
    rgb = rgba[..., :3]
    dist = min_dist_to_corners(rgb, corners)
    lum = luminance(rgb)
    # Background: close to at least one corner AND dark enough (avoids bright wave interior).
    similar = (dist <= rgb_thresh) & (lum <= lum_max_bg)

    reachable = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for x in range(w):
        for y in (0, h - 1):
            if similar[y, x] and not reachable[y, x]:
                reachable[y, x] = True
                q.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if similar[y, x] and not reachable[y, x]:
                reachable[y, x] = True
                q.append((y, x))

    while q:
        y, x = q.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and not reachable[ny, nx] and similar[ny, nx]:
                reachable[ny, nx] = True
                q.append((ny, nx))

    return reachable


def clear_enclosed_background(
    rgba: np.ndarray,
    corners: np.ndarray,
    rgb_thresh: float,
    lum_max_bg: float,
) -> None:
    """Clear background-like connected components that do not touch the image border."""
    rgb = rgba[..., :3]
    a = rgba[..., 3]
    dist = min_dist_to_corners(rgb, corners)
    lum = luminance(rgb)
    bg_like = (dist <= rgb_thresh) & (lum <= lum_max_bg) & (a > 200)

    labeled, n = ndimage.label(bg_like, structure=np.ones((3, 3), dtype=bool))
    for k in range(1, n + 1):
        comp = labeled == k
        touches = (
            comp[0, :].any()
            or comp[-1, :].any()
            or comp[:, 0].any()
            or comp[:, -1].any()
        )
        if not touches:
            rgba[comp] = [0, 0, 0, 0]


def light_edge_defringe(
    rgba: np.ndarray,
    corners: np.ndarray,
    lum_max: float = 40.0,
    rgb_max: float = 36.0,
) -> np.ndarray:
    """Clear only near-background pixels touching transparency (avoids eating dark letter fill by gaps)."""
    out = rgba.copy()
    a = out[..., 3]
    transparent = a < 42
    if not transparent.any():
        return out
    near = ndimage.binary_dilation(transparent, structure=np.ones((3, 3), dtype=bool))
    rgb = out[..., :3]
    dist = min_dist_to_corners(rgb.astype(np.float64), corners)
    lum = luminance(rgb.astype(np.float64))
    kill = near & (lum <= lum_max) & (dist <= rgb_max) & (a >= 22)
    out[kill] = [0, 0, 0, 0]
    return out


def process(
    rgba: np.ndarray,
    # min-dist to corners: true gaps ~35–40; dark letter fill often ~43–48 — keep edge/hole rgb in between.
    edge_rgb: float = 40.0,
    edge_lum: float = 96.0,
    hole_rgb: float = 40.0,
    hole_lum: float = 108.0,
) -> np.ndarray:
    corners = corner_samples(rgba[..., :3])
    out = rgba.copy()
    reachable = edge_flood_background(out, corners, edge_rgb, edge_lum)
    out[reachable] = [0, 0, 0, 0]
    clear_enclosed_background(out, corners, hole_rgb, hole_lum)
    out = light_edge_defringe(out, corners)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--edge-rgb", type=float, default=40.0)
    ap.add_argument("--edge-lum", type=float, default=96.0)
    ap.add_argument("--hole-rgb", type=float, default=40.0)
    ap.add_argument("--hole-lum", type=float, default=108.0)
    args = ap.parse_args()

    img = Image.open(args.input).convert("RGBA")
    rgba = np.array(img)
    fixed = process(
        rgba,
        edge_rgb=args.edge_rgb,
        edge_lum=args.edge_lum,
        hole_rgb=args.hole_rgb,
        hole_lum=args.hole_lum,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(fixed.astype(np.uint8), "RGBA").save(args.output, optimize=True)


if __name__ == "__main__":
    main()
