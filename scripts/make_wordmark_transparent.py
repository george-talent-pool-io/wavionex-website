#!/usr/bin/env python3
"""
Make WAVIONEX wordmark PNG transparent:
1) Remove background connected to image edges (near-black, low chroma).
2) Remove enclosed voids (e.g. inside O) with the same dark/neutral signature.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def luminance(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def chroma(rgb: np.ndarray) -> np.ndarray:
    mx = np.max(rgb, axis=-1)
    mn = np.min(rgb, axis=-1)
    return mx - mn


def void_mask(
    rgba: np.ndarray,
    lum_max: float,
    chroma_max: float,
    alpha_in: int,
) -> np.ndarray:
    rgb = rgba[..., :3].astype(np.float64)
    a = rgba[..., 3]
    lum = luminance(rgb)
    ch = chroma(rgb)
    return (a >= alpha_in) & (lum <= lum_max) & (ch <= chroma_max)


def _touches_border(comp: np.ndarray) -> bool:
    return (
        comp[0, :].any()
        or comp[-1, :].any()
        or comp[:, 0].any()
        or comp[:, -1].any()
    )


def expand_transparent_into_dark(
    rgba: np.ndarray,
    iterations: int = 4,
    lum_max: float = 98.0,
    alpha_trans: int = 42,
    alpha_kill_floor: int = 22,
) -> np.ndarray:
    """Iteratively clear darker pixels adjacent to transparency (cleans I/O/N gaps, halos)."""
    out = rgba.copy()
    struct8 = np.ones((3, 3), dtype=bool)
    for _ in range(iterations):
        a = out[..., 3]
        transparent = a < alpha_trans
        if not transparent.any():
            break
        near = ndimage.binary_dilation(transparent, structure=struct8)
        rgb = out[..., :3].astype(np.float64)
        lum = luminance(rgb)
        kill = near & (lum <= lum_max) & (a >= alpha_kill_floor)
        out[kill] = [0, 0, 0, 0]
    return out


def remove_dark_halos(
    rgba: np.ndarray,
    dilate_px: int = 3,
    lum_max: float = 72.0,
    alpha_floor: int = 20,
) -> np.ndarray:
    """Remove a thin ring of dark fringe next to transparency (anti-alias cleanup)."""
    out = rgba.copy()
    a = out[..., 3]
    transparent = a < 35
    if not transparent.any():
        return out
    k = 2 * dilate_px + 1
    struct = np.ones((k, k), dtype=bool)
    near = ndimage.binary_dilation(transparent, structure=struct)
    rgb = out[..., :3].astype(np.float64)
    lum = luminance(rgb)
    kill = near & (lum <= lum_max) & (a >= alpha_floor)
    out[kill] = [0, 0, 0, 0]
    return out


def remove_enclosed_light_counters(
    rgba: np.ndarray,
    lum_min: float = 215.0,
    chroma_max: float = 28.0,
    alpha_in: int = 60,
) -> np.ndarray:
    """Clear only clearly enclosed near-white holes (O, A, E counters)."""
    out = rgba.copy()
    rgb = out[..., :3].astype(np.float64)
    lum = luminance(rgb)
    ch = chroma(rgb)
    a = out[..., 3]
    void = (a >= alpha_in) & (lum >= lum_min) & (ch <= chroma_max)
    labeled, n = ndimage.label(void)
    for k in range(1, n + 1):
        comp = labeled == k
        if not _touches_border(comp):
            out[comp] = [0, 0, 0, 0]
    return out


def strip_dark_fringe(
    rgba: np.ndarray,
    erode_iter: int = 2,
    lum_max: float = 96.0,
    alpha_floor: int = 35,
) -> np.ndarray:
    """Remove dark semi-opaque ring just outside eroded letter interior (anti-halo)."""
    out = rgba.copy()
    rgb = out[..., :3].astype(np.float64)
    lum = luminance(rgb)
    a = out[..., 3] >= alpha_floor
    struct = np.ones((3, 3), dtype=bool)
    inner = ndimage.binary_erosion(a, structure=struct, iterations=erode_iter)
    fringe = a & ~inner
    out[fringe & (lum <= lum_max)] = [0, 0, 0, 0]
    return out


def remove_unprotected_opaque(
    rgba: np.ndarray,
    dilate: int = 18,
) -> np.ndarray:
    """
    Keep only pixels near 'letter substance' (lit waves, bright rims, saturated blues).
    Removes leftover opaque junk in holes and gaps.
    """
    out = rgba.copy()
    rgb = out[..., :3].astype(np.float64)
    lum = luminance(rgb)
    ch = chroma(rgb)
    # Interior waves + edges: either fairly bright, or clearly chromatic (not muddy black).
    core = ((lum >= 52) & (ch >= 22)) | (lum >= 88) | (ch >= 55)
    struct = np.ones((3, 3), dtype=bool)
    thick = ndimage.binary_dilation(core, structure=struct, iterations=dilate)
    a = out[..., 3]
    kill = (a > 25) & ~thick
    out[kill] = [0, 0, 0, 0]
    return out


def transparent_outside_and_holes(
    rgba: np.ndarray,
    lum_edge: float = 74.0,
    chroma_edge: float = 54.0,
    lum_hole: float = 118.0,
    chroma_hole: float = 88.0,
    alpha_in: int = 20,
) -> np.ndarray:
    h, w = rgba.shape[:2]
    # Tight mask: only clearly “background” connects to the image edge.
    void_edge = void_mask(rgba, lum_edge, chroma_edge, alpha_in)

    edge = np.zeros((h, w), dtype=bool)
    edge[0, :] = void_edge[0, :]
    edge[-1, :] = void_edge[-1, :]
    edge[:, 0] = void_edge[:, 0]
    edge[:, -1] = void_edge[:, -1]

    struct8 = np.ones((3, 3), dtype=bool)
    outside = ndimage.binary_propagation(edge, mask=void_edge, structure=struct8)
    out = rgba.copy()
    out[outside] = [0, 0, 0, 0]

    # Looser mask: dark teal / black inside O, between letters, E bars, etc.
    a = out[..., 3]
    loose = void_mask(out, lum_hole, chroma_hole, alpha_in=15) & (a > 200)
    labeled, n = ndimage.label(loose, structure=struct8)
    for k in range(1, n + 1):
        comp = labeled == k
        if not _touches_border(comp):
            out[comp] = [0, 0, 0, 0]

    out = remove_dark_halos(out, dilate_px=4, lum_max=92.0)
    out = remove_enclosed_light_counters(out)
    out = expand_transparent_into_dark(out, iterations=6, lum_max=108.0)
    out = remove_unprotected_opaque(out, dilate=20)
    out = strip_dark_fringe(out, erode_iter=2, lum_max=104.0)
    out = strip_dark_fringe(out, erode_iter=1, lum_max=118.0)
    out = remove_dark_halos(out, dilate_px=2, lum_max=85.0)
    out = expand_transparent_into_dark(out, iterations=3, lum_max=95.0)
    out = crush_muddy_pixels(out)

    return out


def crush_muddy_pixels(rgba: np.ndarray) -> np.ndarray:
    """Zero alpha on low-luminance, low-chroma mud (fringes, hole noise)."""
    out = rgba.copy()
    rgb = out[..., :3].astype(np.float64)
    lum = luminance(rgb)
    ch = chroma(rgb)
    a = out[..., 3]
    mud = (lum < 74.0) & (ch < 50.0) & (a > 28)
    out[mud] = [0, 0, 0, 0]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--lum-max", type=float, default=68.0)
    p.add_argument("--chroma-max", type=float, default=46.0)
    args = p.parse_args()

    img = Image.open(args.input).convert("RGBA")
    rgba = np.array(img)
    fixed = transparent_outside_and_holes(
        rgba, lum_edge=args.lum_max, chroma_edge=args.chroma_max
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(fixed.astype(np.uint8), "RGBA").save(args.output, optimize=True)


if __name__ == "__main__":
    main()
