import os
import time
import yaml
import h5py
from pathlib import Path
from zipfile import ZipFile
from PIL import Image

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import plotly
import plotly.graph_objs as go


def compute_score(z_true, z_pred, w, tau=1.0, border=20):
    """
    Compute the evaluation score, ie. average of unweighted and weighted R2 scores.

    Args:
      z_true  ... Ground truth distance map
      z_pred  ... Predicted distance map
      w       ... Certainty weights (should be in [0, 1]).
                  Value of 1 meand absolutely certain
      tau     ... Clip relative difference image with this value.
                  This avoids that outliers have a large impact.
      border  ... Number of border pixels to remove for evaluation

    Returns:
      score ... Average of unweighted and weighted R2 score
      r2    ... Unweighted R2 score
      r2_w  ... Weighted R2 score
    """
    H, W = z_true.shape
    assert z_true.shape == z_pred.shape == w.shape, "Received mismatching shapes"

    # Remove boundary points, since
    #  1. z_true has some artifact at boundaries
    #  2. Cannot predict boundaries with NCC
    z_true = z_true[border:H-border, border:W-border]
    z_pred = z_pred[border:H-border, border:W-border]
    w = w[border:H-border, border:W-border]

    # Enforce range of certainty score to [0, 1]
    if np.any(w < 0) or np.any(w > 1):
        print("Warning: Certainty scores out of range. Clipping...")
        w = np.clip(w, 0, 1)

    # Consider inverse distance image
    z_true = 1. / z_true
    z_pred = np.divide(1., z_pred, where=z_pred!=0, out=np.zeros_like(z_pred))

    z_avg = np.mean(z_true)

    # Clip difference, whenever relative error > tau
    rel_diff = (z_pred - z_true) / z_true
    mask = np.abs(rel_diff) > tau

    rel_diff[mask] = tau
    diff = z_true * rel_diff

    # Un-weighted R2 score
    err_avg = np.mean((z_true - z_avg)**2)
    err_rec = np.mean(diff**2)

    # Weighted R2 score
    err_avg_w = np.mean(w *(z_true - z_avg)**2)
    err_rec_w = np.mean(w* diff**2)

    r2 = 1. - err_rec / err_avg
    r2_w = 1. - err_rec_w / err_avg_w if err_avg_w != 0 else float('-inf')
    return 0.5 * (r2 + r2_w), r2, r2_w


def eval_scene(data_path, reconstructor_3d):
    """
    This will be run on the server to evaluate your algorithm on the test scene.

    Args:
        data_path         ... Path to data
        reconstructor_3d  ... Instance of Stereo3dReconstructor
    """

    # Load data
    img_l, img_r, z_true, calib_dict, calib_points = load_data(data_path)
    H, W, _ = img_l.shape

    # Start timing
    t0 = time.time()

    # Fill calibration dictionary
    calib_dict = reconstructor_3d.fill_calib_dict(calib_dict, calib_points)

    # Reconstruct distance image
    points3d = reconstructor_3d.recon_scene_3d(img_l, img_r, calib_dict)

    # End timing
    t1 = time.time()

    # Check wheter points3d has expected shape
    assert points3d.shape == (H, W, 4), "Stereo3dReconstructor returned wrong shape!"
    z_pred = points3d[:, :, 2]
    w = points3d[:, :, 3]

    score, r2, r2_w = compute_score(z_true, z_pred, w)
    return score, r2, r2_w, t1 - t0


def load_data(data_path):
    """
    Returns:
        img_l         ... Left camera image
        img_r         ... Right camera image
        img_z         ... Ground truth distance image (Units of [mm])
        calib_dict    ... Dictionary with camera intrinsics
        calib_points  ... DataFrame with some pairs for calibration
    """
    data_path = Path(data_path)

    fname_l = data_path / 'cam_l.png'
    fname_r = data_path / 'cam_r.png'

    img_l = load_image(str(fname_l))
    img_r = load_image(str(fname_r))

    img_l = img_l.astype(float) / 255.
    img_r = img_r.astype(float) / 255.
    img_z = np.load(data_path / 'true_distance.npy')

    # Read points for calibration
    calib_points = pd.read_csv(data_path / 'calib_points.csv')

    # Read calibration dictionary
    with open(data_path / 'calib_dict.yml', 'r') as stream:
        try:
            calib_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return img_l, img_r, img_z, calib_dict, calib_points
    

def load_image(im_path):
    """
    This function loads an image from file and returns it as NumPy array.
    The NumPy array will have shape Height x Witdh x Channels and data of type unit8 in the range [0, 255]
    """
    im = Image.open(im_path)
    return np.asarray(im)


def test_triangulation(est_calib_dict, calib_points, triangulation_fn):
    """
    Validate estimation of focal_length (f), base_line (b) and triangulation function
    
    Args:
        est_calib_dict    ... Calibration dictionary with keys f and b estimated by students
        val_calib_points  ... A set of validation calibration points
        triangulation_fn  ... The triangulation function (takes arguments: ul, ur, v, calib_dict)

    Returns:
        NRMSE between predicted (X, Y, Z) and ground truth (X, Y, Z)
    """

    # Predict (X, Y, Z)
    ul = np.array(calib_points['ul [px]'])
    vl = np.array(calib_points['vl [px]'])
    ur = np.array(calib_points['ur [px]'])
    vr = np.array(calib_points['vr [px]'])

    # Predictions should be in [mm]
    xyz_pred = triangulation_fn(ul, ur, vl, est_calib_dict)

    # Get ground truth from calib_points - converted to [mm]
    X = np.array(calib_points['X [mm]'])
    Y = np.array(calib_points['Y [mm]'])
    Z = np.array(calib_points['Z [mm]'])

    xyz_ref = np.stack([X, Y, Z], axis=-1)

    # Compute difference
    if xyz_pred.shape == xyz_ref.shape:
        mse = np.sum((xyz_pred - xyz_ref)**2, -1)
        error = np.sum(np.sqrt(mse / np.sum(xyz_ref, -1)**2))

        if error >= 1.e-4:
            print(f"Test failed with NRMSE error of {error:1.4e}\n")
        else:
            print("Test succeeded :-)\n")
    else:
        print(f"Test failed because estimation has wrong shape:")
        print(f"Shape of estimation: {xyz_pred.shape}")
        print(f"Expected shape: {xyz_ref.shape}")
    
        print(f"Estimated XYZ: \n {xyz_pred} \nGround truth XYZ: \n {xyz_ref}")

def test_ncc(ncc_fn):
    patch = np.array([
        [0, 0, 0, 0],
        [0, 1, -1, 0],
        [0, 0, 0, 0]
    ])

    corr_sol = np.array([[[1.0, -0.5], [-0.5, 1.0]]])
    corr = ncc_fn(patch, patch, 1)

    if np.allclose(corr, corr_sol, rtol=1e-2):
        print("Test of compute_ncc() successful :)\n")
    else:
        print("ERROR!!! Test of compute_ncc() failed :(\n")

    print('Here is the computed NCC')
    print(corr)


def plot_correlation(img_left, img_right, corr, col_to_plot, mode='colour'):
    """
    Plot the normalized cross-correlation for a given column.
    The column for which NCC is being plotted is marked
    with a red line in the left image.

    Args:
        img_l   (np.array of shape (num_rows, num_cols)): left grayscale image
        img_r   (np.array of shape (num_rows, num_cols)): right grayscale image
        corr    (np.array of shape
                (
                    num_rows - 2*mask_halfwidth,
                    num_cols - 2*mask_halfwidth,
                    num_cols - 2*mask_halfwidth
                ):
                Computed normalized cross-correlation (NCC) between patches
                in the two images.
        col_to_plot: the column in the left image for which to plot the NCC
        mode: Either 'gray' or 'colour'
    """

    # Create copies not to write into originals
    img_l = img_left.copy()
    img_r = img_right.copy()

    # Pad the slice so that it's size is same as the images
    pad_rows = int((img_l.shape[0] - corr.shape[0]) / 2)
    pad_cols = int((img_l.shape[1] - corr.shape[1]) / 2)
    corr = np.pad(corr, (
        (pad_rows, pad_rows),
        (pad_cols, pad_cols),
        (pad_cols, pad_cols)
    ), 'constant', constant_values=0)

    corr_slice = corr[:, col_to_plot, :]

    # Draw line in the left image to denote the column being visualized
    if mode == 'gray':
        img_l = np.dstack([img_l, img_l, img_l])

    img_l[:, col_to_plot, 0] = 1.
    img_l[:, col_to_plot, 1] = 0.
    img_l[:, col_to_plot, 2] = 0.

    plt.ion()
    f, axes_array = plt.subplots(1, 3, figsize=(15, 5))
    axes_array[0].set_title('Left camera image', fontsize=12)
    axes_array[0].imshow(img_l, cmap=plt.cm.gray)

    axes_array[0].tick_params(
        bottom='off', labelbottom='off', left='off', labelleft='off'
    )
    axes_array[1].set_title('Right camera image', fontsize=12)
    axes_array[1].imshow(img_r, cmap=plt.cm.gray)
    axes_array[1].tick_params(
        bottom='off', labelbottom='off', left='off', labelleft='off'
    )

    axes_array[2].set_title('NCC for column marked by red line', fontsize=12)
    axes_array[2].imshow(corr_slice)
    axes_array[2].tick_params(
        bottom='off', labelbottom='off', left='off', labelleft='off'
    )

    plt.show(block=True)


def plot_point_cloud(gray_left, gray_right, points3d, boarder=20):
    """ Visualize the re-constructed point-cloud

        Args:
            gray_left (np.array of shape (num_rows, num_cols)): left grayscale image
            gray_right (np.array of shape (num_rows, num_cols)): right grayscale image
            points3d ((np.array of shape (num_rows - 2*mask_halfwidth, num_cols - 2*mask_halfwidth, 3)):
                3D World co-ordinates for each pixel in the left image (excluding the boundary pixels
                which are ignored during NCC calculation).
        """

    plt.close('all')
    plt.ion()
    f, axes_array = plt.subplots(1, 2, figsize=(15, 6))
    axes_array[0].set_title('Left camera image', fontsize=12)
    axes_array[0].imshow(gray_left, cmap=plt.cm.gray)
    axes_array[0].tick_params(bottom='off', labelbottom='off', left='off', labelleft='off')
    axes_array[1].set_title('Right camera image', fontsize=12)
    axes_array[1].imshow(gray_right, cmap=plt.cm.gray)
    axes_array[1].tick_params(bottom='off', labelbottom='off', left='off', labelleft='off')
    plt.show()

    margin_y = gray_left.shape[0] - points3d.shape[0]
    margin_x = gray_left.shape[1] - points3d.shape[1]

    points3d = points3d[boarder:-boarder,boarder:-boarder,:]
    colors = []

    if gray_left.ndim == 2:
        # Pick colours for grayscale image
        for r in range(points3d.shape[0]):
            for c in range(points3d.shape[1]):
                col = gray_left[r+margin_y,c+margin_x]
                colors.append('rgb('+str(col)+','+str(col)+','+str(col)+')')
    elif gray_left.ndim == 3:
        # Pick colours for RGB image
        for r in range(points3d.shape[0]):
            for c in range(points3d.shape[1]):
                col = gray_left[r+margin_y,c+margin_x]
                colors.append('rgb('+str(col[0])+','+str(col[1])+','+str(col[2])+')')

    data = [go.Scatter3d(
        x=-0.001*points3d[:,:,0].flatten(),
        y=-0.001*points3d[:,:,1].flatten(),
        z=0.001*points3d[:,:,2].flatten(),
        mode='markers',
        marker=dict(
            size=1,
            color=colors,
            line=dict(width=0)
        )
    )]
    layout = go.Layout(
        scene=dict(
            camera=dict(
                    up=dict(x=0, y=1, z=0),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=0.1, y=0.1, z=-1.)
            ),
            xaxis = dict(nticks=4, range=[-3,3],),
            yaxis = dict(nticks=4, range=[-2.4,2.4],),
            zaxis = dict(nticks=4, range=[-0.1,7],)
        ),
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    plotly.offline.init_notebook_mode(connected=True)
    fig = go.Figure(data=data, layout=layout)
    return fig