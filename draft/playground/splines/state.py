"""Mutable image-series state for the interactive playground."""

from dataclasses import dataclass

import torch

from demeter.utils.spline_data import TimedImageBatch


@dataclass
class ImageSeries:
    source: torch.Tensor
    targets: torch.Tensor
    times: list[float | None]
    source_path: str
    paths: list[str]
    selected: int = 1

    @classmethod
    def from_batch(cls, batch: TimedImageBatch) -> "ImageSeries":
        return cls(
            batch.source.clone(),
            batch.target.clone(),
            list(batch.target_times),
            batch.source_path,
            list(batch.target_paths or ("",) * len(batch.target)),
        )

    def order(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                range(len(self.targets)),
                key=lambda index: (
                    self.times[index] is None,
                    self.times[index] or 0.0,
                    index,
                ),
            )
        )

    def number(self, index: int) -> int:
        return self.order().index(index) + 1

    def target_at(self, step: int, steps: int) -> int:
        placed = [index for index in self.order() if self.times[index] is not None]
        if not placed:
            return min(max(self.selected - 1, 0), len(self.targets) - 1)
        return next(
            (
                index
                for index in placed
                if round(self.times[index] * steps) >= step
            ),
            placed[-1],
        )

    def names(self) -> tuple[str, ...]:
        targets = tuple(
            path or f"target_{self.number(index):03d}"
            for index, path in enumerate(self.paths)
        )
        return (self.source_path or "source", *targets)

    def place(self, index: int, time: float) -> None:
        if any(
            other != index and value is not None and abs(value - time) < 1e-8
            for other, value in enumerate(self.times)
        ):
            raise ValueError("Another target already occupies that node.")
        self.times[index] = float(time)
        self.selected = index + 1

    def promote(self, index: int) -> None:
        source = self.source.clone()
        source_path = self.source_path
        self.source = self.targets[index : index + 1].clone()
        self.source_path = self.paths[index]
        self.targets[index : index + 1] = source
        self.paths[index] = source_path
        self.selected = 0

    def add(self, images: list[torch.Tensor], paths: list[str]) -> None:
        self.targets = torch.cat((self.targets, *images))
        self.times.extend([None] * len(images))
        self.paths.extend(paths)
        self.selected = len(self.targets) - len(images) + 1

    def remove_selected(self) -> None:
        index = self.selected - 1
        keep = [candidate for candidate in range(len(self.targets)) if candidate != index]
        self.targets = self.targets[keep]
        self.times = [self.times[candidate] for candidate in keep]
        self.paths = [self.paths[candidate] for candidate in keep]
        self.selected = min(index, len(self.targets) - 1) + 1

    def replace_targets(self, image: torch.Tensor, path: str) -> None:
        self.targets = image
        self.times = [1.0]
        self.paths = [path]
        self.selected = 1

    def to_batch(self, target_index: int | None = None) -> TimedImageBatch:
        if target_index is None:
            if any(time is None for time in self.times):
                raise ValueError("place every target image before running a spline")
            order = self.order()
            times = tuple(self.times[index] for index in order)
        else:
            order = (target_index,)
            times = (1.0,)
        return TimedImageBatch(
            self.source,
            self.targets[list(order)],
            times,
            self.source_path,
            tuple(self.paths[index] for index in order),
        )
