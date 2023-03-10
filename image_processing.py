import os
import nibabel as nib
from graph import *
import math


def crop_3d_image(original_image):
    print("Image dimensions:", original_image.shape)
    dimension0 = original_image.shape[0]
    dimension1 = original_image.shape[1]
    dimension2 = original_image.shape[2]

    # the goal is (79, 95, 79)
    dimension0_difference = dimension0 - 79
    dimension1_difference = dimension1 - 95
    dimension2_difference = dimension2 - 79

    cropped_image = original_image.slicer[
                    math.ceil(dimension0_difference/2):-math.floor(dimension0_difference/2),
                    math.ceil(dimension1_difference/2):-math.floor(dimension1_difference/2),
                    math.ceil(dimension2_difference/2):-math.floor(dimension2_difference/2)]
    print("Image dimensions:", cropped_image.shape)  # now should be (79, 95, 79)

    path_new_3d_cropped_image = 'C:/Users/teodo/Desktop/thesis/ADD_CNNs/scans/preprocessed/scan1.nii'  # TODO
    # save_3d_image(cropped_image, path_new_3d_cropped_image)  # TODO
    graph_compare_original_cropped(original_image, cropped_image)  # TODO
    return cropped_image


def save_3d_image(image, path):
    nib.save(image, path)


def rename_3d_image(image_name_string, path, class_name):
    file_name_old = os.path.join(path, image_name_string)
    image = nib.load(file_name_old)

    image_name_string_new = split_string_find_image_name(image_name_string) + "_" + class_name + ".nii"
    print(image_name_string_new)
    file_name_new = os.path.join(path, image_name_string_new)
    print(file_name_new)
    save_3d_image(image, file_name_new)

    if os.path.isfile(file_name_old):
        os.remove(file_name_old)
    else:
        print("Error: %s file not found" % file_name_old)


def split_string_find_image_name(image_name_string):
    substrings = image_name_string.split(".")
    return substrings[0]


if __name__ == '__main__':
    path_3d_images_preprocessed = 'C:/Users/teodo/Desktop/thesis/ADD_CNNs/scans/preprocessed'
    image_name = 'preprocessed1.nii'
    file_name = os.path.join(path_3d_images_preprocessed, image_name)
    image_3d = nib.load(file_name)
    cropped_3d_image = crop_3d_image(image_3d)
    visualization_3d_image(cropped_3d_image)
    rename_3d_image(image_name, path_3d_images_preprocessed, "CN")
