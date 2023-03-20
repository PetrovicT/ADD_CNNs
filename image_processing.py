import os
import nibabel as nib
import math
from glob import glob
import shutil
from PIL import Image
import numpy as np

from graph import *
from table_processing import *


def load_all_nii_images_from_ssd() -> None:
    source_folder = "preprocessed_OR"  # PATH
    excel_file_path = './tables/preprocessed_OR.xlsx'
    excel_sheet_name = 'preprocessed_OR'

    original_paths = ['D:/{}/ADNI/*/*/*/*/*.nii'.format(source_folder)]
    file_cnt = 0
    for path in original_paths:
        for filename in glob(path):
            file_cnt += 1
            # find desired image name and destination
            image_id = get_image_id(filename)
            image_name_string = get_image_name_with_extension(filename)
            destination_folder = get_image_group(image_id, excel_file_path, excel_sheet_name)
            renamed_image = image_id + ".nii"
            # if the image with desired name and destination already exists, it has been processed in the past
            if not os.path.isfile('D:/{}/{}/{}'.format(source_folder, destination_folder, renamed_image)):
                # not processed in the past, save the image as id.nii
                shutil.copy2(filename, 'D:/{}/{}'.format(source_folder, destination_folder))
                os.rename('D:/{}/{}/{}'.format(source_folder, destination_folder, image_name_string),
                          'D:/{}/{}/{}'.format(source_folder, destination_folder, renamed_image))

            if file_cnt % 40 == 0:
                print("Processed " + str(file_cnt) + " images!")

    print("Number of .nii images: " + str(file_cnt))


def crop_3d_image(original_image: np.ndarray) -> np.ndarray:
    # print("Image dimensions:", original_image.shape)
    dimension0 = original_image.shape[0]  # axial
    dimension1 = original_image.shape[1]  # coronal
    dimension2 = original_image.shape[2]  # sagittal
    # the goal dimension is (79, 95, 79)
    dimension0_difference = dimension0 - 79
    dimension1_difference = dimension1 - 95
    dimension2_difference = dimension2 - 79

    cropped_image = original_image[
                    math.ceil(dimension0_difference / 2):-math.floor(dimension0_difference / 2),
                    math.ceil(dimension1_difference / 2):-math.floor(dimension1_difference / 2),
                    math.ceil(dimension2_difference / 2):-math.floor(dimension2_difference / 2)]

    # normalization
    cropped_image = (cropped_image - cropped_image.min()) / (cropped_image.max() - cropped_image.min())

    # visualization
    slice_index = 30
    view = "sagittal"
    # graph_compare_original_cropped(original_image, cropped_image, slice_index, view)

    return cropped_image


def save_image(array: np.ndarray, path: str, image_name_without_extension: str) -> None:
    np.save('{}/{}.npy'.format(path, image_name_without_extension), array)


def make_2d_rgb_images_from_3d_image_and_save(image_id: str, image_class: str, array: np.ndarray, first_slice_coronal=4, first_slice_sagittal=20,
                                              first_slice_axial=16) -> None:
    source_folder = "preprocessed_OR"  # PATH
    destination_folder = image_class + "_rgb"
    # 24 RGB coronal images, 19 RGB sagittal images, and 16 RGB axial images
    for i in range(24):
        image_tmp = array[(first_slice_coronal + (i * 3)):(3 + first_slice_coronal + (i * 3)), :, :]
        image_tmp = image_tmp.transpose(1, 2, 0)
        new_image_name = image_id + "_coronal" + "_" + str((first_slice_coronal + (i * 3))) + "_" + image_class
        save_image(
            image_tmp,
            'D:/{}/{}'.format(source_folder, destination_folder),
            new_image_name
        )
        # not working properly
        # img = Image.fromarray(image * 255, 'RGB')
        # img.save('D:/{}/{}/proba{}.png'.format(source_folder, destination_folder, i))
        # visualization_2d_image(image)

    for i in range(19):
        image_tmp = array[:, (first_slice_sagittal + (i * 3)):(3 + first_slice_sagittal + (i * 3)), :]
        image_tmp = image_tmp.transpose(0, 2, 1)
        new_image_name = image_id + "_sagittal" + "_" + str((first_slice_sagittal + (i * 3))) + "_" + image_class
        save_image(
            image_tmp,
            'D:/{}/{}'.format(source_folder, destination_folder),
            new_image_name
        )

    for i in range(16):
        image_tmp = array[:, :, (first_slice_axial + (i * 3)):(3 + first_slice_axial + (i * 3))]
        image_tmp = image_tmp.transpose(0, 1, 2)
        new_image_name = image_id + "_axial" + "_" + str((first_slice_axial + (i * 3))) + "_" + image_class
        save_image(
            image_tmp,
            'D:/{}/{}'.format(source_folder, destination_folder),
            new_image_name
        )


def get_image_name_with_extension(path: str) -> str:
    substrings = path.split("\\")
    return substrings[len(substrings) - 1]


def crop_all_nii_images() -> None:
    source_folder = "preprocessed_OR"  # PATH
    excel_file_path = './tables/preprocessed_OR.xlsx'
    excel_sheet_name = 'preprocessed_OR'

    original_paths = ['D:/{}/AD/*.nii'.format(source_folder), 'D:/{}/CN/*.nii'.format(source_folder)]
    file_cnt = 0
    for path in original_paths:
        for filename in glob(path):
            file_cnt += 1
            if file_cnt < 1950:
                continue
            # find desired image name and destination
            image_id = get_image_id_from_filename(filename)
            original_image_name = image_id + ".nii"
            cropped_image_name = image_id + ".npy"
            destination_folder = get_image_group(image_id, excel_file_path, excel_sheet_name) + "_cropped"
            # if the image with desired name and destination already exists, it has been cropped in the past
            if not os.path.isfile('D:/{}/{}/{}'.format(source_folder, destination_folder, cropped_image_name)):
                # not processed in the past, save the cropped image as id.npy
                image_3d = nib.load(
                    'D:/{}/{}/{}'.format(source_folder, get_image_group(image_id, excel_file_path, excel_sheet_name),
                                         original_image_name)
                )
                if image_3d.shape[0] < 80 or image_3d.shape[1] < 96 or image_3d.shape[2] < 80 or len(image_3d.shape) > 3:
                    print("Image dimensions " + str(image_3d.shape) + " not valid.")
                    os.remove('D:/{}/{}/{}'.format(source_folder,
                                                   get_image_group(image_id, excel_file_path, excel_sheet_name),
                                                   original_image_name))
                    file_cnt -= 1
                else:
                    image_3d_array = np.asarray(image_3d.dataobj)
                    image_3d_array_cropped = crop_3d_image(image_3d_array)
                    save_image(
                        image_3d_array_cropped,
                        'D:/{}/{}'.format(source_folder, destination_folder),
                        image_id
                    )
            if file_cnt % 40 == 0:
                print("Cropped " + str(file_cnt) + " images!")

    print("Number of cropped images: " + str(file_cnt))


def make_2d_rgb_images_from_3d_image_and_save_all():
    source_folder = "preprocessed_OR"  # PATH
    excel_file_path = './tables/preprocessed_OR.xlsx'
    excel_sheet_name = 'preprocessed_OR'

    original_paths = ['D:/{}/AD_cropped/*.npy'.format(source_folder), 'D:/{}/CN_cropped/*.npy'.format(source_folder)]
    file_cnt = 0
    for path in original_paths:
        for filename in glob(path):
            file_cnt += 1
            if (file_cnt*59) < (84783 + 30739):
                continue
            # find desired image name and destination
            image_id = get_image_id_from_filename(filename)
            image_name = image_id + ".npy"
            destination_folder = get_image_group(image_id, excel_file_path, excel_sheet_name) + "_rgb"
            # if the image with desired name and destination already exists, it has been saved in the past
            if not os.path.isfile('D:/{}/{}/{}'.format(source_folder, destination_folder, image_name)):
                # not saved in the past, save the rgb image as id.npy
                image_3d_array = np.load(
                    'D:/{}/{}/{}'.format(source_folder,
                                         get_image_group(image_id, excel_file_path, excel_sheet_name) + "_cropped",
                                         image_name)
                )
                if len(image_3d_array.shape) > 3:
                    print("Image dimensions " + str(image_3d_array.shape) + " not valid!")
                make_2d_rgb_images_from_3d_image_and_save(
                    image_id,
                    get_image_group(image_id, excel_file_path, excel_sheet_name),
                    image_3d_array
                )

            if file_cnt % 40 == 0:
                print("Made 2d images from " + str(file_cnt) + " 3d images!")

    print("Made 2d images from " + str(file_cnt) + " 3d images!")
