from .deepict_unet3d import UNet3D_Lightning, UNet3D_Lightning_Rotation
from .deepict_unet3d_ttt import UNet3D_Lightning_ITTT
from .denoiseg import Denoiseg
# from .mae import LightningMaskedAutoencoder
# from .f2fd.model import Denoising_3DUNet as Denoiseg_F2FD
from .memseg import MemSeg, MemDenoiseg, MemDenoisegTTT
from .unet3d import UNet3D
from .memseg import MemDenoisegTTTSubset, MemDenoisegTentSubset
from .memseg_n2v import MemDenoiseg as MemDenoisegN2V
from .memseg_n2v import MemDenoisegTTTSubset as MemDenoisegN2VTTTSubset

__all__ = {
    "deepict_unet3d": UNet3D_Lightning,
    "deepict_unet3d_rotation": UNet3D_Lightning_Rotation,
    "deepict_unet3d_ittt": UNet3D_Lightning_ITTT,
    # "mae": LightningMaskedAutoencoder,
    "denoiseg": Denoiseg,
    # "denoiseg_f2fd": Denoiseg_F2FD, 
    "memseg": MemSeg,
    "memdenoiseg": MemDenoiseg,
    "memdenoiseg_ttt": MemDenoisegTTT,
    "memdenoiseg_ttt_subset": MemDenoisegTTTSubset,
    "memdenoiseg_tent_subset": MemDenoisegTentSubset,
    "memdenoiseg_n2v": MemDenoisegN2V,
    "memdenoiseg_n2v_ttt_subset": MemDenoisegN2VTTTSubset,
}


def build_model(config):
    model = __all__[config.method.model_name](config=config)

    return model
