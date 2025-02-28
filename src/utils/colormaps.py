from PIL import Image
import numpy as np
from matplotlib import cm


def grayscale_to_viridis(gray_array):
    """
    Convert a grayscale image to an RGB image using the viridis colormap.

    Parameters:
    - input_path (str): Path to the input grayscale image.
    - output_path (str, optional): Path to save the output RGB image. If None, the image will not be saved.

    Returns:
    - Image object: The converted viridis RGB image as a Pillow Image object.
    """
    # Load the grayscale image
    # gray_image = Image.open(input_path).convert("L")  # Ensure it's grayscale

    # Convert the image to a NumPy array
    # gray_array = np.array(gray_image)

    # Normalize the grayscale values to the range [0, 1]
    normalized_array = gray_array / 255.0

    # Use the viridis colormap to map the normalized values to RGB
    colormap = cm.get_cmap("viridis")
    viridis_array = colormap(normalized_array)[:, :, :3]  # Drop the alpha channel

    # Convert the RGB array (float 0-1) to an 8-bit integer array (0-255)
    rgb_array = (viridis_array * 255).astype(np.uint8)

    # Create an RGB image using Pillow
    viridis_image = Image.fromarray(rgb_array)

    # Save the output image if a path is provided
    # if output_path:
    #     viridis_image.save(output_path)

    return viridis_image


# Example usage:
# viridis_image = grayscale_to_viridis("grayscale_image.png", "viridis_image.png")
# viridis_image.show()
