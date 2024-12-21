from .dataset import CryoETMemSegDataset
from .deepict_dataset import DeepictPatchDataset

__all__ = {
    "deepict_h5": DeepictPatchDataset,
    "memseg": CryoETMemSegDataset,
}


def build_dataset(config, val=False):
    dataset = __all__[config.method.dataset](config=config, is_validation=val)
    return dataset
