import numpy as np
import nibabel as nib
import SimpleITK as sitk
import os

from termcolor import colored

try:
    from keras import backend as K
    from keras.models import *
    from tensorflow.python.client import device_lib
except Exception as e:
   print(e)
   print('-' * 100)
   print(colored("""ERROR: Failed to initialize tensorflow and Keras.
       If you are using deepdefacer[tf-gpu], please make
       sure that your CUDA and NVIDIA drivers are properly
       installed.""", 'red'))
   print('-' * 100)
   sys.exit(1)

def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def resample_image(nifti_img, specified_shape, mask=False):

    if mask:
        revised_nifti = nib.Nifti1Image(nifti_img, np.eye(4)) 
        nib.save(revised_nifti, 'mask_intermediate.nii.gz')
        nifti_img = 'mask_intermediate.nii.gz'

    img = sitk.ReadImage(nifti_img)
    img_data = sitk.GetArrayFromImage(img)

    if len(img_data.shape) != 3:

        img_data = np.squeeze(img_data) 
        revised_nifti = nib.Nifti1Image(img_data, nib.load(nifti_img).affine) 
        nib.save(revised_nifti, nifti_img) 
        img = sitk.ReadImage(nifti_img)

    dimension = img.GetDimension() 

    reference_physical_size = np.zeros(dimension)
    reference_physical_size[:] = [(sz-1)*spc if sz*spc>mx  else mx for sz,spc,mx in
                                 zip(img.GetSize(), img.GetSpacing(),
                                     reference_physical_size)]

    reference_origin = np.zeros(dimension)
    reference_direction = np.identity(dimension).flatten()
    reference_size = specified_shape 
    reference_spacing = [phys_sz/(sz-1) for sz,phys_sz in 
                           zip(reference_size, reference_physical_size)]

    reference_image = sitk.Image(reference_size, img.GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)

    reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))

    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(img.GetDirection())
    transform.SetTranslation(np.array(img.GetOrigin()) - reference_origin)

    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize())/2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    centered_transform = sitk.Transform(transform)
    centered_transform.AddTransform(centering_transform)

    resampled_img_data = sitk.Resample(img, reference_image, centered_transform, sitk.sitkLinear, 0.0)
    resampled_img_data = np.swapaxes(sitk.GetArrayFromImage(resampled_img_data), 0, -1) 

    if mask:
        os.remove('mask_intermediate.nii.gz')

    return resampled_img_data


def pre_process_image(img_file):

    optimal_dims = [160, 160, 160]

    img_data = resample_image(img_file, optimal_dims) 
    
    resamp_img = img_data.astype(np.float32)

    img_data = np.expand_dims(resamp_img, axis=0)

    img_data = np.expand_dims(img_data, axis=0)

    min_val = np.min(img_data)
    max_val = np.max(img_data)

    norm_img_data = (img_data - min_val) / (max_val - min_val + 1e-7)
    return norm_img_data, resamp_img


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
