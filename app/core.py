import rasterio

from utils import *
from models import unet_model

path = "C:/Users/USER/PycharmProjects/Test_tf_gpu/MMflood_train/PLOTO/rtc_gamma0_budapest/"

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


model_type = 'MM_flood_unet'
results_path = 'C:/Users/USER/PycharmProjects/Test_tf_gpu/MMflood_train/MM_flood_results/'+ model_type+'/'
model_path = results_path+'MMflood_unet'

input_shape = (256,256,2)
unet_model = unet_model(input_shape)

unet_model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

# unet_model.load_weights(model_path)

# unet_model.save(results_path+'MMflood_unet.h5')

unet_model =  tf.keras.models.load_model(results_path+'MMflood_unet.h5')

# Define patch size
patch_size = 256

# Pad the stacked image
padded_stacked_image = pad_image(stacked_image, patch_size*10)
# Split the image into patches
patches = split_image_into_patches(padded_stacked_image, patch_size)



# Run prediction on patches
predicted_patches = []


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
    heatmap = generate_grad_cam(unet_model, patch)
    
    # Adjust layout
    predicted_patches.append(prediction)
    
# Convert the list of predicted patches to a numpy array
predicted_patches = np.array(predicted_patches)
print(predicted_patches.shape)

# Reconstruct the image from predicted patches
reconstructed_image = reconstruct_image_from_patches(predicted_patches, (2560,2560,1))
print(reconstructed_image.shape)

# Reconstruct the image from predicted patches
reconstructed_original_image = reconstruct_image_from_patches(patches, (2560,2560,2))
print(reconstructed_original_image.shape)