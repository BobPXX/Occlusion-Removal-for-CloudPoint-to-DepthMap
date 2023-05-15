import math
import os
import json
import glob
import cv2 as cv
import numpy as np
import open3d
import time
import numba.cuda as cuda
import warnings

#Occulusion Removal-----------------------------------------------------
# refer: Zhao, Yiming, et al. "A surface geometry model for lidar depth completion." IEEE Robotics and Automation Letters 6.3 (2021): 4457-4464.
#GPU version
@cuda.jit
def create_occlusion_mask(lidar_coords, coords, depths, dx, dy, dz,mask):
    N = lidar_coords.shape[0]
    i, j = cuda.grid(2)

    if i < N and j < N:
        xi, yi = coords[i]
        xj, yj = coords[j]
        di = depths[i]
        dj = depths[j]
        x0i, y0i, _ = lidar_coords[i]
        x0j, y0j, _ = lidar_coords[j]

        if abs(xi - xj) < dx and abs(yi - yj) < dy and di > dj + dz and (
                (x0i - x0j) * (xi - xj) < 0 or (y0i - y0j) * (yi - yj) < 0):
            mask[i] = False

def get_occlusion_mask(lidar_coords, coords, depths, dx, dy, dz):
    N = lidar_coords.shape[0]
    block_dim = (16, 16)
    grid_dim = ((N - 1) // block_dim[0] + 1, (N - 1) // block_dim[1] + 1)

    mask = np.ones(N, dtype=np.bool_)
    d_lidar_coords = cuda.to_device(lidar_coords)
    d_coords = cuda.to_device(coords)
    d_depths = cuda.to_device(depths)
    d_mask = cuda.to_device(mask)

    create_occlusion_mask[grid_dim, block_dim](d_lidar_coords, d_coords, d_depths, dx, dy, dz, d_mask)
    d_mask.copy_to_host(mask)

    return mask
#Occulusion Removal-----------------------------------------------------

def project_depth(
    coords,
    intrinsic_matrix,
    extrinsic_matrix,
    image_size,
    interp_mode="bilinear",
    scatter_mode="minimum",
    dx=10,
    dy=10,
    dz=10
):
    assert interp_mode in ("nearest", "bilinear")
    assert scatter_mode in ("minimum", "addition")
    lidar_coords=coords[:,:3]
    coords = coords @ extrinsic_matrix.T @ intrinsic_matrix.T
    coords = coords[coords[..., -1] > 0, ...]
    lidar_coords=lidar_coords[coords[..., -1] > 0, ...]

    assert coords.size

    depths = coords[..., -1] #shape: (point_num,)
    coords = coords[..., :-1] / coords[..., -1:] #shape: (point_num, 2), Homogeneous division

    #dx = (np.max(coords[:, 0], axis=0) - np.min(coords[:, 0], axis=0)) / image_size[0]
    #dy = (np.max(coords[:, 1], axis=0) - np.min(coords[:, 1], axis=0)) / image_size[1]

    if interp_mode == "nearest":

        coords = np.round(coords).astype(np.long)

        lower_bound_masks = 0.0 <= coords
        upper_bound_masks = coords < image_size[::-1]

        valid_masks = np.all(lower_bound_masks & upper_bound_masks, axis=-1)

        depths = depths[valid_masks]
        coords = coords[valid_masks, ...]
        lidar_coords = lidar_coords[np.where(valid_masks)[0]]

        # occlusion removal-----------------------------------------------------
        occlusion_mask=get_occlusion_mask(lidar_coords,coords,depths,dx,dy,dz)
        depths = depths[occlusion_mask]
        coords = coords[occlusion_mask, ...]
        # occlusion removal-----------------------------------------------------

        coords_x, coords_y = coords.T
        indices = coords_y * image_size[-1] + coords_x
        weights = np.ones_like(depths)

    if interp_mode == "bilinear":

        min_coords = np.floor(coords).astype(np.long)
        max_coords = min_coords + 1

        lower_bound_masks = 0.0 <= min_coords
        upper_bound_masks = max_coords < image_size[::-1]

        valid_masks = np.all(lower_bound_masks & upper_bound_masks, axis=-1)
        lidar_coords = lidar_coords[np.where(valid_masks)[0]]
        depths = depths[valid_masks, ...]
        coords = coords[valid_masks, ...]
        min_coords = min_coords[valid_masks, ...]
        max_coords = max_coords[valid_masks, ...]

        # occlusion removal-----------------------------------------------------
        occlusion_mask=get_occlusion_mask(lidar_coords,coords,depths,dx,dy,dz)
        depths = depths[occlusion_mask]
        coords = coords[occlusion_mask, ...]
        min_coords = min_coords[occlusion_mask, ...]
        max_coords = max_coords[occlusion_mask, ...]
        # occlusion removal-----------------------------------------------------

        coords_x, coords_y = coords.T
        min_coords_x, min_coords_y = min_coords.T
        max_coords_x, max_coords_y = max_coords.T

        minmax_coords_x = np.stack([min_coords_x, max_coords_x], axis=0)
        minmax_coords_y = np.stack([min_coords_y, max_coords_y], axis=0)

        minmax_weights_x = 1.0 - np.abs(minmax_coords_x - coords_x)
        minmax_weights_y = 1.0 - np.abs(minmax_coords_y - coords_y)

        indices = (np.expand_dims(minmax_coords_y, axis=1) * image_size[-1] + np.expand_dims(minmax_coords_x, axis=0)).ravel()
        weights = (np.expand_dims(minmax_weights_y, axis=1) * np.expand_dims(minmax_weights_x, axis=0)).ravel()

        depths = np.broadcast_to(depths, (4, *depths.shape)).ravel()

    if scatter_mode == "minimum":

        depth_map = np.full(np.prod(image_size), np.inf)
        np.minimum.at(depth_map, indices, depths)

        depth_map = np.where(np.isfinite(depth_map), depth_map, 0.0)

    depth_map = depth_map.reshape(*image_size)


    return depth_map


def make_depth_map(
    root,
    cloud_filename,
    image_filename,
    intrinsic_filename,
    extrinsic_filename,
    image_size=[1280, 3840],
    overlay=True,
):
    intrinsic_filename = os.path.join(root, intrinsic_filename)
    with open(intrinsic_filename) as file:
        intrinsic_params = json.load(file)
        intrinsic_matrix = np.array(intrinsic_params["intrinsic_matrix"])

    extrinsic_filename, = glob.glob(os.path.join(root, extrinsic_filename))
    extrinsic_params = cv.FileStorage(extrinsic_filename, cv.FileStorage_READ)
    R = extrinsic_params.getNode("R").mat()
    t = extrinsic_params.getNode("t").mat() * 1e-3
    extrinsic_matrix = np.concatenate([R, t], axis=-1)
    extrinsic_matrix = extrinsic_matrix @ [
        [ 0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [ 0.0, 0.0, 1.0, 0.0],
        [ 0.0, 0.0, 0.0, 1.0],
    ]

    point_cloud = open3d.io.read_point_cloud(os.path.join(root, cloud_filename))
    point_cloud = np.asanyarray(point_cloud.points)
    point_cloud = np.pad(
        array=point_cloud,
        pad_width=((0, 0), (0, 1)),
        mode="constant",
        constant_values=1.0,
    )

    depth_map = project_depth(
        coords=point_cloud,
        intrinsic_matrix=intrinsic_matrix,
        extrinsic_matrix=extrinsic_matrix,
        image_size=image_size,
    )

    depth_map = (depth_map * (1 << 8)).clip(0, (1 << 16) - 1).astype(np.uint16)

    depth_filename = os.path.join(root, "depth_maps/"+image_filename)
    os.makedirs(os.path.dirname(depth_filename), exist_ok=True)
    cv.imwrite(depth_filename, depth_map)

    if overlay:
        image = cv.imread(os.path.join(root, "images/"+image_filename))
        depth_map = np.cbrt(depth_map / depth_map.max())
        depth_map = (depth_map * (1 << 8)).astype(np.uint8)
        depth_map = np.broadcast_to(depth_map[..., None], (*depth_map.shape, 3))
        color_map = cv.applyColorMap(depth_map, cv.COLORMAP_JET)
        overlaid_image = np.where(depth_map > 0.0, color_map, image)
        overlay_filename = os.path.join(root, "overlaid_images/"+image_filename)
        os.makedirs(os.path.dirname(overlay_filename), exist_ok=True)
        cv.imwrite(os.path.join(root, "overlaid_images/"+image_filename), overlaid_image)



start_time = time.time()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    make_depth_map(root="root_dir", cloud_filename="couldpoint_file_path", image_filename="image_file_path", intrinsic_filename="intrinsic_filename_file_path",extrinsic_filename="extrinsic_filename_file_path")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.3f} seconds")