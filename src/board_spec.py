from dataclasses import dataclass


@dataclass(frozen=True)
class BoardSpec:
    inner_cols: int
    inner_rows: int

    def __post_init__(self):
        if self.inner_cols <= 0 or self.inner_rows <= 0:
            raise RuntimeError(
                f"棋盘规格必须为正整数，当前为 cols={self.inner_cols}, rows={self.inner_rows}"
            )

    @property
    def grid_cols(self) -> int:
        return self.inner_cols + 1

    @property
    def grid_rows(self) -> int:
        return self.inner_rows + 1

    @property
    def point_count(self) -> int:
        return self.inner_cols * self.inner_rows

    def validate_point_count(self, count: int) -> int:
        if count != self.point_count:
            raise RuntimeError(
                f"角点数量与棋盘规格不一致: expected={self.point_count}, actual={count}, "
                f"inner_cols={self.inner_cols}, inner_rows={self.inner_rows}"
            )
        return count
