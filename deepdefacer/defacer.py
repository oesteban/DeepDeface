import os
import sys
import numpy as np
import nibabel as nib
import argparse

from termcolor import colored


try:
    from keras.models import load_model
except Exception as e:
    print(e)
    print('-' * 100)
    print(colored("""ERROR: Failed to initialize tensorflow and Keras.
       If you are using deepdefacer[tf-gpu], please make
       sure that your CUDA and NVIDIA drivers are properly
       installed.""", 'red'))
    print('-' * 100)
    sys.exit(1)

from .defacer_utils import (
    dice_coefficient,
    pre_process_image,
    get_available_gpus,
    resample_image
)


class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write(colored('error: %s!\n' % message, 'red'))
        self.print_help()
        sys.exit(2)


def deface_3D_MRI():

    parser = MyParser()
    parser.add_argument('--input_file', type=str,
        help='(REQUIRED) filepath for input MRI image.', required=True)
    parser.add_argument('--defaced_output_path', type=str,
        help="""(OPTIONAL) Destination path and name of defaced MRI,
                defaults to current "<image_name>_defaced" in current directory.""")
    parser.add_argument('--mask_output_path', type=str,
        help='(OPTIONAL) Destination path and name of predicted mask for MRI.')

    args = parser.parse_args()

    if not get_available_gpus():
        print('-' * 100)
        print(colored("""WARNING: Could not find an available GPU on your system.
         Defaulting to CPU.""", 'yellow'))
        print('-' * 100)

    MRI_image_path = args.input_file

    if '.nii' not in MRI_image_path and '.nii.gz' not in MRI_image_path:
        print('-' * 100)
        print(colored("""ERROR: Please ensure that the input MRI file is in .nii or
         .nii.gz format""", 'red'))
        print('-' * 100)
        sys.exit(1)

    print(colored('Preproessing input MRI image...', 'green'))

    MRI_image_shape = np.squeeze(nib.load(MRI_image_path).get_data()).shape

    if len(MRI_image_shape) != 3:
        print('-' * 100)
        print(colored("""ERROR: Unable to deface MRI: Please ensure that input dimensions
         are 3D.""", 'red'))
        print('-' * 100)

    MRI_image_data, MRI_unnormalized_data = pre_process_image(MRI_image_path)

    path_to_module = os.path.dirname(__file__)
    model_file_path = os.path.join(path_to_module, "model.hdf5")

    deepdeface_model = load_model(model_file_path,
                                 custom_objects={'dice_coefficient': dice_coefficient})

    print('-' * 100)
    print(colored('Masking %s ....' % (MRI_image_path), 'green'))

    mask_prediction = deepdeface_model.predict(MRI_image_data)

    mask_prediction[mask_prediction < 0.5] = 0
    mask_prediction[mask_prediction >= 0.5] = 1

    mask_prediction = np.squeeze(mask_prediction)

    mask_prediction = resample_image(mask_prediction,
                                    specified_shape=MRI_image_shape, mask=True)

    if args.mask_output_path:
        print(colored('Successfully created mask! Saving to %s...' %
                      (args.mask_output_path), 'cyan'))
        mask_save = nib.Nifti1Image(mask_prediction, nib.load(MRI_image_path).affine)
        nib.save(mask_save, args.mask_output_path)
        print(colored('Saved.', 'green'))

    MRI_unnormalized_data = np.squeeze(nib.load(MRI_image_path).get_data())

    masked_image = np.multiply(MRI_unnormalized_data, mask_prediction)

    masked_image_save = nib.Nifti1Image(
        masked_image, nib.load(MRI_image_path).affine)

    masked_image_resampled = masked_image_save

    if args.defaced_output_path:
        output_file = args.defaced_output_path
    else:
        output_file = os.path.splitext(os.path.splitext(
            os.path.basename(MRI_image_path))[0])[0] + '_defaced.nii.gz'

    print(colored('Successfully defaced image! Saving to %s...' %
                   (output_file), 'cyan'))

    nib.save(masked_image_resampled, output_file)

    print(colored('Saved.', 'green'))
    print('-' * 100)


if __name__ == "__main__":
    deface_3D_MRI()
