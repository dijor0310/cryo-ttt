from .memseg import MemSeg, MemDenoiseg, MemDenoisegTTT
from .memseg import MemDenoisegTTTSubset, MemDenoisegTentSubset

__all__ = {
    "memseg": MemSeg,
    "memdenoiseg": MemDenoiseg,
    "memdenoiseg_ttt": MemDenoisegTTT,
    "memdenoiseg_ttt_subset": MemDenoisegTTTSubset,
    "memdenoiseg_tent_subset": MemDenoisegTentSubset,
}


def build_model(config):
    model = __all__[config.method.model_name](config=config)
    return model
