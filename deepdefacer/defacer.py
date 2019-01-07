import os
import sys
import numpy as np
import nibabel as nib
import pdb
import os

try:
    from keras.models import *
except Exception as e:
    print(e)
    print('---------------------------------------------------------------------------------------------------------------------------------------')
    print('ERROR: Failed to initialize tensorflow and Keras. If you are using deepdefacer[tf-gpu], please make sure that your CUDA and NVIDIA drivers are properly installed.')
    print('---------------------------------------------------------------------------------------------------------------------------------------')
    sys.exit(1)

from defacer_utils import resize_img, dice_coefficient, resize_img, pre_process_image, get_available_gpus, check_for_resampling, resample_image


def deface_3D_MRI():

    if len(sys.argv) < 2:
        print('----------------------------------------------------------------------------------------------------')
        print(
            "usage: Please specify the filepath of a MRI image for defacing....(e.g deepdefacer <path of MRI>")
        print('----------------------------------------------------------------------------------------------------')
        sys.exit(1)

    if not get_available_gpus():
        print('---------------------------------------------------------------------------------------------------------------------------------------')
        print("WARNING: Could not find an available GPU on your system. Defaulting to CPU.")
        print('---------------------------------------------------------------------------------------------------------------------------------------')

    MRI_image_path = sys.argv[1]
    if '.nii' not in MRI_image_path and '.nii.gz' not in MRI_image_path:
        print('------------------------------------------------------------------------')
        print("ERROR: Please ensure that the input MRI file is in .nii or .nii.gz format")
        print('------------------------------------------------------------------------')
        sys.exit(1) 

    print('Preproessing input MRI image...')

    MRI_image_shape = np.squeeze(nib.load(MRI_image_path).get_data()).shape
    resampling_required = check_for_resampling(MRI_image_shape)

    if len(MRI_image_shape) != 3:
        print('------------------------------------------------------------------------')
        print("ERROR: Unable to deface MRI: Please ensure that input dimensions are 3D.")
        print('------------------------------------------------------------------------')

    MRI_image_data, MRI_unnormalized_data = pre_process_image(MRI_image_path)


    deepdeface_model = load_model(
        'deepdefacer/model.hdf5', custom_objects={'dice_coefficient': dice_coefficient})

    print('-------------------------------------------------')
    print('Masking %s ....' % (MRI_image_path))

    mask_prediction = deepdeface_model.predict(MRI_image_data)

    mask_prediction[mask_prediction < 0.5] = 0
    mask_prediction[mask_prediction >= 0.5] = 1

    mask_prediction = np.squeeze(mask_prediction)

    if resampling_required:
        mask_prediction = resample_image(mask_prediction, specified_shape=MRI_image_shape, mask=True)
        MRI_unnormalized_data = np.squeeze(nib.load(MRI_image_path).get_data())

    masked_image = np.multiply(MRI_unnormalized_data, mask_prediction)

    masked_image_save = nib.Nifti1Image(
        masked_image, nib.load(MRI_image_path).affine)

    if not resampling_required:

        masked_image_resampled = resize_img(
            masked_image_save, orig=MRI_image_shape, get_nifti=True)

    else:
        masked_image_resampled = masked_image_save

    output_file = os.path.splitext(os.path.splitext(
        os.path.basename(MRI_image_path))[0])[0] + '_defaced.nii.gz'

    print('Completed! Saving to %s...' % (output_file))

    nib.save(masked_image_resampled, output_file)

    print('Saved.')
    print('----------')


if __name__ == "__main__":
    deface_3D_MRI() 