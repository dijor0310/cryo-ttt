from .deepict_unet3d import UNet3D_Lightning, UNet3D_Lightning_Rotation

__all__ = {
    "deepict_unet3d": UNet3D_Lightning,
    "deepict_unet3d_rotation": UNet3D_Lightning_Rotation,
}


def build_model(config):
    model = __all__[config.method.model_name](config=config)

    return model
