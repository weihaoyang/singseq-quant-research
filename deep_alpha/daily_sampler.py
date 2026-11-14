"""Shuffle dates, never mix dates inside a correlation-loss batch."""
import numpy as np


class DailyBatchSampler:
    def __init__(self, dates, batch_size=256, seed=42):
        if batch_size < 2:
            raise ValueError("correlation needs at least two samples")
        self.dates = np.asarray(dates)
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        days = np.unique(self.dates)
        self.rng.shuffle(days)
        for day in days:
            indices = np.flatnonzero(self.dates == day)
            self.rng.shuffle(indices)
            if len(indices) < 2:
                continue
            chunks = list(np.array_split(indices, max(1, len(indices) // self.batch_size)))
            for chunk in chunks:
                yield chunk.tolist()
