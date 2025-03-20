from .dataset import CryoETMemSegDataset
from .deepict_dataset import DeepictPatchDataset
from .denoiseg_dataset import DenoisegPatchDataset
from .denoiseg_dataset import DenoisegPatchInMemoryDataset
from .denoiseg_dataset import DenoisegPatchDatasetV2
# from .f2fd_dataset import singleCET_FourierDataset
from .f2fd_dataset import F2FD_Dataset, F2FD_DatasetV2

__all__ = {
    "deepict_h5": DeepictPatchDataset,
    "memseg": CryoETMemSegDataset,
    "denoiseg": DenoisegPatchDataset,
    "denoiseg_v2": DenoisegPatchDatasetV2,
    "denoiseg_in_memory": DenoisegPatchInMemoryDataset,
    "denoiseg_f2fd": F2FD_DatasetV2,
}


def build_dataset(config, val=False, test=False):
    dataset = __all__[config.method.dataset](
        config=config, is_validation=val, is_test=test
    )
    return dataset
