import matplotlib.pyplot as plt


def graph_compare_original_cropped(original_3d_image, cropped_3d_image):
    plot_original = original_3d_image.get_fdata()
    plot_cropped = cropped_3d_image.get_fdata()
    for i in range(2):
        plt.subplot(2, 2, i + 1)
        if i == 0:
            plt.imshow(plot_original[100, :, ])
        else:
            plt.imshow(plot_cropped[11, :, ])
        plt.gcf().set_size_inches(10, 10)
    plt.show()


