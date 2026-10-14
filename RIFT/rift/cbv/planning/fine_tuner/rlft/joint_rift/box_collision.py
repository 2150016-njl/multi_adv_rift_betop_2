"""Exact rectangle-vs-obstacle collision traces using separating axes."""
import numpy as np


def aabb_collision_trace(center, obstacles):
    center, obstacles = np.asarray(center), np.asarray(obstacles)
    low1, high1 = center.min(-2)[:, None], center.max(-2)[:, None]
    low2, high2 = obstacles.min(-2)[None], obstacles.max(-2)[None]
    return ((high1 >= low2) & (high2 >= low1)).all(-1).any(1)


def box_collision_trace(center, obstacles, chunk_size=8):
    center, obstacles = np.asarray(center, dtype=float), np.asarray(obstacles, dtype=float)
    if center.ndim != 4 or obstacles.ndim != 4 or center.shape[1:] != obstacles.shape[1:] or center.shape[-2:] != (4, 2):
        raise ValueError("boxes must be [G,T,4,2] and [N,T,4,2]")
    if chunk_size < 1 or not np.isfinite(center).all() or not np.isfinite(obstacles).all():
        raise ValueError("finite boxes and positive chunk size required")
    result = np.zeros(center.shape[:2], dtype=bool)
    if not len(obstacles):
        return result
    def axes(vertices):
        edges = np.stack((vertices[..., 1, :] - vertices[..., 0, :], vertices[..., 2, :] - vertices[..., 1, :]), -2)
        return np.stack((-edges[..., 1], edges[..., 0]), -1)
    for start in range(0, len(center), chunk_size):
        boxes = center[start:start + chunk_size]
        shape = (len(boxes), len(obstacles), *center.shape[1:])
        first = np.broadcast_to(boxes[:, None], shape)
        second = np.broadcast_to(obstacles[None], shape)
        directions = np.concatenate((axes(first), axes(second)), -2)
        a = np.einsum("...vc,...ac->...va", first, directions)
        b = np.einsum("...vc,...ac->...va", second, directions)
        separated = (a.max(-2) < b.min(-2)) | (b.max(-2) < a.min(-2))
        result[start:start + len(boxes)] = (~separated.any(-1)).any(1)
    return result
