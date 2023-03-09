import pandas as pd
import numpy as np
import os
import nibabel as nib
import matplotlib.pyplot as plt


if __name__ == '__main__':
    df = pd.read_excel("params.xlsx")
    ## print whole excel file
    # print(df)

    path_scans = 'C:/Users/teodo/Desktop/thesis/ADD_CNNs/scans'
    filename = os.path.join(path_scans, 'scan2.nii')
    img = nib.load(filename)
    print("Image dimensions:", img.shape)  # originally (256, 256, 170)

    ## show img using plotly
    test_plotly = img.get_fdata()
    #test = test_plotly[:, :, 100]
    #plt.imshow(test)
    #plt.show()

    # originally (256, 256, 170), the goal is (79, 95, 79)
    # => 256-79=177 => 177/2 = 88 (start from 89th and finish at (end-88)th)
    changed_img = img.slicer[89:-88, 81:-80, 46:-45]
    print("Image dimensions:", changed_img.shape)  # now should be (79, 95, 79)
    #nib.save(changed_img, 'C:/Users/teodo/Desktop/thesis/scans/my_image.nii')

    test_plotly2 = changed_img.get_fdata()
    #test = test_plotly[:, :, 64]
    #plt.imshow(test)
    #plt.show()

    for i in range(2):
        plt.subplot(2, 2, i + 1)
        if i == 0:
            plt.imshow(test_plotly[100, :, ])
        else:
            # should be 11
            plt.imshow(test_plotly2[75, :, ])
        plt.gcf().set_size_inches(10, 10)
    plt.show()

