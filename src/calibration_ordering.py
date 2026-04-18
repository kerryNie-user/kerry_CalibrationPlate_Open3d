import numpy as np


def reorder_points(points: np.ndarray, cols: int, rows: int, transpose: bool, flip_x: bool, flip_y: bool) -> np.ndarray:
    grid = points.reshape(rows, cols, -1)
    if transpose:
        grid = np.transpose(grid, (1, 0, 2))
    if flip_x:
        grid = grid[:, ::-1, :]
    if flip_y:
        grid = grid[::-1, :, :]
    return grid.reshape(-1, points.shape[1])


def build_order_candidates(points: np.ndarray, pattern_cols: int, pattern_rows: int):
    n = len(points)
    candidates = []
    source_shapes = [(pattern_cols, pattern_rows)]
    if pattern_cols != pattern_rows:
        source_shapes.append((pattern_rows, pattern_cols))
    seen = set()
    for cols, rows in source_shapes:
        if cols * rows != n:
            continue
        for transpose in (False, True):
            out_cols, out_rows = (rows, cols) if transpose else (cols, rows)
            if out_cols * out_rows != n:
                continue
            if out_cols != pattern_cols or out_rows != pattern_rows:
                continue
            for flip_x in (False, True):
                for flip_y in (False, True):
                    key = (cols, rows, transpose, flip_x, flip_y)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(
                        {
                            "source_cols": cols,
                            "source_rows": rows,
                            "transpose": transpose,
                            "flip_x": flip_x,
                            "flip_y": flip_y,
                            "points": reorder_points(points, cols, rows, transpose, flip_x, flip_y),
                        }
                    )
    if len(candidates) == 0:
        candidates.append(
            {
                "source_cols": pattern_cols,
                "source_rows": pattern_rows,
                "transpose": False,
                "flip_x": False,
                "flip_y": False,
                "points": points.copy(),
            }
        )
    return candidates
