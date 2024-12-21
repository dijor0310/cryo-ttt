# from membrain

from typing import Callable, List, Union

import torch
from monai.transforms import (
    Compose,
    ToTensord,
)


def get_training_transforms(
    prob_to_one: bool = False,
    return_as_list: bool = False,
) -> Union[List[Callable], Compose]:
    # raise NotImplementedError
    return get_validation_transforms(return_as_list=return_as_list)


def get_validation_transforms(
    return_as_list: bool = False,
) -> Union[List[Callable], Compose]:
    """
    Returns the data augmentation transforms for the validation phase.

    The function sets up a sequence of transformations including downsampling
    and tensor conversion. If desired, the sequence can be returned as a list.

    Parameters
    ----------
    return_as_list : bool, optional
        If True, the sequence of transformations is returned as a list.
        If False, the sequence is returned as a Compose object. Default is False.

    Returns
    -------
    List[Callable] or Compose
        If return_as_list is True, the function returns a list of
            transformation functions.
        If return_as_list is False, the function returns a Compose object
            containing the sequence of transformations.

    """
    aug_sequence = [
        ToTensord(keys=["image"], dtype=torch.float),
    ]
    if return_as_list:
        return aug_sequence
    return Compose(aug_sequence)
