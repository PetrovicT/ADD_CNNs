import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import math


# original_3d_image, cropped_3d_image: numpy array
def graph_compare_original_cropped(original_3d_image, cropped_3d_image, slice_index=30, view="sagittal"):  # DONE
    dimension0_difference = original_3d_image.shape[0] - cropped_3d_image.shape[0]  # axial
    dimension1_difference = original_3d_image.shape[1] - cropped_3d_image.shape[1]  # coronal
    dimension2_difference = original_3d_image.shape[2] - cropped_3d_image.shape[2]  # sagittal

    slice_index_difference_0 = math.ceil(dimension0_difference / 2)
    slice_index_difference_1 = math.ceil(dimension1_difference / 2)
    slice_index_difference_2 = math.ceil(dimension2_difference / 2)

    for i in range(2):
        plt.subplot(2, 2, i + 1)
        if i == 0:
            if view == "axial":
                plt.imshow(original_3d_image[slice_index + slice_index_difference_0, :, ])
            if view == "coronal":
                plt.imshow(original_3d_image[:, slice_index + slice_index_difference_1, ])
            if view == "sagittal":
                plt.imshow(original_3d_image[:, :, slice_index + slice_index_difference_2])
        else:
            if view == "axial":
                plt.imshow(cropped_3d_image[slice_index, :, ])
            if view == "coronal":
                plt.imshow(cropped_3d_image[:, slice_index, ])
            if view == "sagittal":
                plt.imshow(cropped_3d_image[:, :, slice_index])
        plt.gcf().set_size_inches(8, 8)
    plt.show()


# image: numpy array
def visualization_2d_image(image):  # TODO
    fig = plt.imshow(image)
    plt.show()


# image: numpy array
def visualization_3d_image(image, slice_index=30, view="axial"):  # TODO
    if view == "coronal":
        image = np.transpose(image, [1, 2, 0])
    if view == "sagittal":
        image = np.transpose(image, [0, 2, 1])
    # vol2 = np.transpose(image, [1, 2, 0])
    r, c = image[0].shape

    # Define frames
    nb_frames = 68

    fig = go.Figure(frames=[go.Frame(data=go.Surface(
        z=(6.7 - k * 0.1) * np.ones((r, c)),
        surfacecolor=np.flipud(image[67 - k]),
        cmin=0, cmax=1000
    ),
        name=str(k)  # you need to name the frame for the animation to behave properly
    )
        for k in range(nb_frames)])

    # Add data to be displayed before animation starts
    fig.add_trace(go.Surface(
        z=6.7 * np.ones((r, c)),
        surfacecolor=np.flipud(image[67]),
        colorscale=[[0, 'rgb(0,0,0)'], [1, 'rgb(255,255,255)']],
        # colorscale='Gray',
        cmin=0, cmax=200,
        colorbar=dict(thickness=20, ticklen=4)
    ))

    def frame_args(duration):
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f.name], frame_args(0)],
                    "label": str(k),
                    "method": "animate",
                }
                for k, f in enumerate(fig.frames)
            ],
        }
    ]

    # Layout
    fig.update_layout(
        title='Slices in volumetric data',
        width=600,
        height=600,
        scene=dict(
            zaxis=dict(range=[-0.1, 6.8], autorange=False),
            aspectratio=dict(x=1, y=1, z=1),
        ),
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, frame_args(50)],
                        "label": "&#9654;",  # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;",  # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
        ],
        sliders=sliders
    )

    fig.show()


if __name__ == '__main__':
    image_path = './scans/2drgb.nii'
    # visualization_2d_image(image_path)
    # testing

