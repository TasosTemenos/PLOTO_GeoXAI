
import numpy as np


def cccut(img, min_percent = 2, max_percent = 98):
    lo, hi = np.percentile(img, (min_percent, max_percent))
    # Apply linear "stretch" - lo goes to 0, and hi goes to 1
    res_img = (img.astype(np.float32) - lo) / (hi - lo)
    # Multiply by 1, clamp range to [0, 1] and convert to np.float32
    res_img = np.maximum(np.minimum(res_img * 1, 1), 0).astype(np.float32)
    return res_img


# Define function to split image into patches, handling edges
def split_image_into_patches(image, patch_size):
    patches = []
    height, width, channels = image.shape
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            patch = image[y:min(y+patch_size, height), x:min(x+patch_size, width)]
            patches.append(patch)
    return np.array(patches)

# Define function to reconstruct image from patches, handling edges
def reconstruct_image_from_patches(patches, original_shape):
    reconstructed_image = np.zeros(original_shape)
    idx = 0
    patch_size = patches.shape[1]
    for y in range(0, original_shape[0], patch_size):
        for x in range(0, original_shape[1], patch_size):
            patch = patches[idx]
            patch_height, patch_width, _ = patch.shape
            reconstructed_image[y:y+patch_height, x:x+patch_width] = patch
            idx += 1
    return reconstructed_image

# Pad the initial image to size 2560x2560
def pad_image(image, target_size):
    height, width, channels = image.shape
    padded_image = np.zeros((target_size, target_size, channels), dtype=image.dtype)
    pad_height = (target_size - height) // 2
    pad_width = (target_size - width) // 2
    padded_image[pad_height:pad_height+height, pad_width:pad_width+width, :] = image
    return padded_image