import numpy as np


def reassemble_subtomograms(sub_tomograms, centers, original_shape, d, overlap):
    """
    Reassemble the predicted segmentation masks from sub-tomograms into the original tomogram shape.

    Args:
        sub_tomograms (list of numpy.ndarray): List of sub-tomogram predictions (binary, single-channel).
        centers (list of list of int): List of center coordinates for each sub-tomogram.
        original_shape (tuple of int): Shape of the original tomogram (X, Y, Z).
        d (int): Size of each sub-tomogram (d, d, d).

    Returns:
        numpy.ndarray: Reassembled tomogram of shape `original_shape`.
    """
    # Initialize arrays for reassembled mask and normalization counts
    reassembled_mask = np.zeros(
        [elem + (overlap * 2) for elem in original_shape], dtype=np.float32
    )
    normalization_count = np.zeros(
        [elem + (overlap * 2) for elem in original_shape], dtype=np.float32
    )

    half_d = d // 2  # Half size of the sub-tomogram

    for sub_tomogram, center in zip(sub_tomograms, centers):
        # Calculate the bounds of the sub-tomogram in the original tomogram space
        start_x = max(center[0] - half_d, 0)
        start_y = max(center[1] - half_d, 0)
        start_z = max(center[2] - half_d, 0)
        end_x = min(center[0] + half_d, original_shape[0])
        end_y = min(center[1] + half_d, original_shape[1])
        end_z = min(center[2] + half_d, original_shape[2])

        # Calculate the corresponding region in the sub-tomogram
        # sub_start_x = half_d - (center[0] - start_x)
        # sub_start_y = half_d - (center[1] - start_y)
        # sub_start_z = half_d - (center[2] - start_z)
        sub_start_x = 0
        sub_start_y = 0
        sub_start_z = 0
        sub_end_x = sub_start_x + (end_x - start_x)
        sub_end_y = sub_start_y + (end_y - start_y)
        sub_end_z = sub_start_z + (end_z - start_z)

        # Add the sub-tomogram values to the reassembled mask
        reassembled_mask[start_x:end_x, start_y:end_y, start_z:end_z] += sub_tomogram[
            sub_start_x:sub_end_x, sub_start_y:sub_end_y, sub_start_z:sub_end_z
        ]

        # Increment the normalization count for the overlapping regions
        normalization_count[start_x:end_x, start_y:end_y, start_z:end_z] += 1

    # Normalize the reassembled mask to account for overlapping regions
    normalization_count[normalization_count == 0] = 1  # Avoid division by zero
    reassembled_mask /= normalization_count

    # Threshold to obtain binary segmentation (optional, if required)
    # reassembled_mask = (reassembled_mask.sigmoid() > 0.5).astype(np.uint8)

    return reassembled_mask[
        overlap : reassembled_mask.shape[0] - overlap,
        overlap : reassembled_mask.shape[1] - overlap,
        overlap : reassembled_mask.shape[2] - overlap,
    ]


def reassemble_subtomograms_v2(sub_tomograms, centers, original_shape, d, overlap):
    """
    Reassemble the predicted segmentation masks from sub-tomograms into the original tomogram shape.

    Args:
        sub_tomograms (list of numpy.ndarray): List of sub-tomogram predictions (binary, single-channel).
        centers (list of list of int): List of center coordinates for each sub-tomogram.
        original_shape (tuple of int): Shape of the original tomogram (X, Y, Z).
        d (int): Size of each sub-tomogram (d, d, d).

    Returns:
        numpy.ndarray: Reassembled tomogram of shape `original_shape`.
    """
    # Initialize arrays for reassembled mask and normalization counts
    reassembled_mask = np.zeros(original_shape, dtype=np.float32)
    normalization_count = np.zeros(original_shape, dtype=np.float32)

    half_d = d // 2  # Half size of the sub-tomogram

    for sub_tomogram, center in zip(sub_tomograms, centers):
        # Calculate the bounds of the sub-tomogram in the original tomogram space
        start_x = max(center[0] - half_d, 0)
        start_y = max(center[1] - half_d, 0)
        start_z = max(center[2] - half_d, 0)
        end_x = min(center[0] + half_d - 2 * overlap, original_shape[0])
        end_y = min(center[1] + half_d - 2 * overlap, original_shape[1])
        end_z = min(center[2] + half_d - 2 * overlap, original_shape[2])

        # Calculate the corresponding region in the sub-tomogram
        # sub_start_x = half_d - (center[0] - start_x)
        # sub_start_y = half_d - (center[1] - start_y)
        # sub_start_z = half_d - (center[2] - start_z)
        sub_start_x = overlap
        sub_start_y = overlap
        sub_start_z = overlap
        sub_end_x = sub_start_x + (end_x - start_x)
        sub_end_y = sub_start_y + (end_y - start_y)
        sub_end_z = sub_start_z + (end_z - start_z)

        # Add the sub-tomogram values to the reassembled mask
        reassembled_mask[start_x:end_x, start_y:end_y, start_z:end_z] += sub_tomogram[
            sub_start_x:sub_end_x, sub_start_y:sub_end_y, sub_start_z:sub_end_z
        ]

        # Increment the normalization count for the overlapping regions
        normalization_count[start_x:end_x, start_y:end_y, start_z:end_z] += 1

    # Normalize the reassembled mask to account for overlapping regions
    normalization_count[normalization_count == 0] = 1  # Avoid division by zero
    reassembled_mask /= normalization_count

    # Threshold to obtain binary segmentation (optional, if required)
    # reassembled_mask = (reassembled_mask.sigmoid() > 0.5).astype(np.uint8)

    return reassembled_mask
