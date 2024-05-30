# import rasterio
import numpy as np
import tensorflow as tf
import rasterio
from matplotlib import pyplot as plt


from utils import *
from models import unet_model

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

path = "./"

# Open the VV band GeoTIFF file
with rasterio.open(path+'VV/budapest_3.tif') as vv_src:
    vv_band = vv_src.read(1)  # Read VV band data

# Open the VH band GeoTIFF file
with rasterio.open(path+'VH/budapest_3.tif') as vh_src:
    vh_band = vh_src.read(1)  # Read VH band data



stacked_image = cccut(np.stack((vv_band, vh_band), axis=-1))

print(stacked_image.shape)

print(stacked_image.shape, stacked_image.min(), stacked_image.max(), stacked_image.dtype, stacked_image.mean(), stacked_image.std())

vv_band = stacked_image[:,:,0]  # VV band
vh_band = stacked_image[:,:,1]  # VH band

input_shape = (256,256,2)
unet_model =  tf.keras.models.load_model('./MMflood_unet.h5')

# Define patch size
patch_size = 256

# Pad the stacked image
padded_stacked_image = pad_image(stacked_image, patch_size*10)
# Split the image into patches
patches = split_image_into_patches(padded_stacked_image, patch_size)



# Run prediction on patches
predicted_patches = []

print("Number of patches: ", len(patches))
for patch in patches:
    # patch = bandeq(patch)
    # print(patch.shape, patch.min(), patch.max(), patch.dtype, patch.mean(axis = (0,1,2)), patch.std(axis=(0,1,2)))
    patch = np.expand_dims(patch, axis = 0)
    # print("patch : ", patch.shape)
    pred = unet_model.predict(patch)
    prediction = np.squeeze(pred, axis = 0)
    # print("pred : ", prediction.shape, prediction.min(), prediction.max(), prediction.dtype)
    # Apply threshold to the predicted mask
    thresh_pred = (prediction > prediction.mean()).astype(int)
    # print("thresh pred : ", thresh_pred.shape, thresh_pred.min(), thresh_pred.max(), thresh_pred.dtype)
    # heatmap = generate_grad_cam(unet_model, patch)
    
    # Adjust layout
    predicted_patches.append(prediction)

print("End of inference")
# Convert the list of predicted patches to a numpy array
predicted_patches = np.array(predicted_patches)
print(predicted_patches.shape)

# Reconstruct the image from predicted patches
reconstructed_image = reconstruct_image_from_patches(predicted_patches, (2560,2560,1))
print(reconstructed_image.shape)

# Reconstruct the image from predicted patches
reconstructed_original_image = reconstruct_image_from_patches(patches, (2560,2560,2))
print(reconstructed_original_image.shape)


plt.figure()
plt.imshow(reconstructed_image, cmap='gray')
plt.savefig('prediction.png')

plt.figure()
plt.imshow(reconstructed_original_image[:, :, 0], cmap='gray')
plt.savefig('original.png')

plt.figure()
plt.imshow(reconstructed_original_image[:, :, 1], cmap='gray')
plt.savefig('original1.png')
print()
print()
print("End here")