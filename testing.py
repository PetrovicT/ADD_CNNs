from table_processing import *
from image_processing import *

if __name__ == '__main__':
    # ------------------------------- table_processing.py -------------------------------
    """
    path_file = './tables/preprocessed_OR.xlsx'
    num_AD_subjects = get_excel_number_of_subjects(path_file, 'preprocessed_OR', "AD")
    num_CN_subjects = get_excel_number_of_subjects(path_file, 'preprocessed_OR', "CN")

    image_name = "ADNI_002_S_0295_MR_MPR-R__GradWarp__B1_Correction__N3_Br_20070319114336780_S13407_I45112"
    image_id = get_image_id(image_name)
    print(image_id)
    image_group = get_image_group(image_id, path_file, 'preprocessed_OR')
    print(image_group)
    """

    # ------------------------------- image_processing.py -------------------------------
    """
    # load_all_nii_images_from_ssd()
    # crop_all_nii_images()
    # make_2d_rgb_images_from_3d_image_and_save_all()
    
    path_3d_image = 'D:/preprocessed_OR/CN_cropped/'  # PATH
    image_name = 'I474758.npy'  # PATH
    file_name = os.path.join(path_3d_image, image_name)
    image_3d_array = np.load(file_name)

    # load 3d nii image - both methods give same results
    # load image - method 1
    # image_3d = nib.load(file_name)
    # image_3d_array = np.asarray(image_3d.dataobj)
    # load image - method 2
    # image_3d_array = io.imread(file_name).T

    # cropped_3d_image = crop_3d_image(image_3d_array)
    # make_2d_rgb_images_from_3d_image_and_save("I474801.npy", "CN", image_3d_array)
    # visualization_2d_image(image_2d)
    # visualization_3d_image(image_3d_array, "sagittal")
    """

