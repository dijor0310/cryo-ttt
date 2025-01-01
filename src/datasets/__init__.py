from .dataset import CryoETMemSegDataset
from .deepict_dataset import DeepictPatchDataset

__all__ = {
    "deepict_h5": DeepictPatchDataset,
    "memseg": CryoETMemSegDataset,
}


def build_dataset(config, val=False, test=False):
    dataset = __all__[config.method.dataset](
        config=config, is_validation=val, is_test=test
    )
    return dataset
