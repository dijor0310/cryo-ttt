from .deepict_unet3d import UNet3D_Lightning, UNet3D_Lightning_Rotation
from .deepict_unet3d_ttt import UNet3D_Lightning_ITTT
from .denoiseg import Denoiseg
from .mae import LightningMaskedAutoencoder

__all__ = {
    "deepict_unet3d": UNet3D_Lightning,
    "deepict_unet3d_rotation": UNet3D_Lightning_Rotation,
    "deepict_unet3d_ittt": UNet3D_Lightning_ITTT,
    "mae": LightningMaskedAutoencoder,
    "denoiseg": Denoiseg,
}


def build_model(config):
    model = __all__[config.method.model_name](config=config)

    return model
