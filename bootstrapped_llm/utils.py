from collections.abc import Sequence, Iterable

import torch


def split_by_boundaries(sequence: Sequence, boundaries: torch.Tensor) -> Iterable:
    """Splits a sequence into sub-sequences based on the provided boundaries.

    Args:
        sequence (Sequence): The sequence to be split.
        boundaries (torch.Tensor): A tensor indicating the boundaries.
            1 indicates a boundary, 0 indicates no boundary.
    Returns:
        Iterable: A list of sub-sequences.
    """
    splits = []
    start_idx = 0
    for i, boundary in enumerate(boundaries):
        if boundary.item() == 1:
            splits.append(sequence[start_idx:i])
            start_idx = i

    return splits
