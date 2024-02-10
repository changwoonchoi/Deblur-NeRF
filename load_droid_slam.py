import os
import glob
import numpy as np
import torch

def load_droid_slam_data(args):

    focal_length = np.load(os.path.join(args.datadir, "focal_length.npy"))
    principal_point = np.load(os.path.join(args.datadir, "principal_point.npy"))
    resolution = np.load(os.path.join(args.datadir, "resolution.npy"))
    height = int(resolution[1])
    width = int(resolution[0])

    image_paths = sorted(glob.glob(os.path.join(args.datadir, '*_input_image.npy')))
    pose_paths = sorted(glob.glob(os.path.join(args.datadir, '*_input_pose.npy')))
    depth_paths = sorted(glob.glob(os.path.join(args.datadir, '*_input_depth.npy')))  # Only for estimating near and far planes
    depth_cov_paths = sorted(glob.glob(os.path.join(args.datadir, '*_depth_cov.npy')))

    frame_idx = []
    for _, path in enumerate(image_paths):
        frame_idx.append(int(path.split('/')[-1].split('_')[0]))
    frame_idx = np.array(frame_idx)

    num_frame = len(image_paths)
    max_frame = frame_idx[-1]

    fl_x, fl_y = focal_length
    cx, cy = principal_point

    K = np.array([
        [fl_x, 0, cx],
        [0, fl_y, cy],
        [0, 0, 1]
        ])
    
    poses = []
    for pose_path in pose_paths:
        pose = np.load(pose_path)
        poses.append(pose)
    poses = np.stack(poses, axis=0)
    # follow the original code
    poses[:, :3, 1:3] *= -1
    poses = poses[:, [1, 0, 2, 3], :]
    poses[:, 2] *= -1
    
    images = []
    for image_path in image_paths:
        image = np.load(image_path)
        image = image[..., :-1]
        images.append(image)
    images = np.stack(images, axis=0)
    # images = torch.from_numpy(images)

    depth_covs = []
    for depth_cov_path in depth_cov_paths:
        depth_cov = np.load(depth_cov_path)
        depth_covs.append(depth_cov)
    depth_covs = np.stack(depth_covs, axis=0)
    depth_covs = torch.from_numpy(depth_covs)

    depths = []
    for depth_path in depth_paths:
        depth = np.load(depth_path)
        depths.append(depth)
    depths = np.stack(depths, axis=0)
    depths = torch.from_numpy(depths)

    cam_near_far = []
    for i in range(len(depths)):
        depth = depths[i]
        depth_cov = depth_covs[i]
        cam_near_far.append(torch.stack((depth[depth_cov<depth_cov.median()].min(), depth[depth_cov<depth_cov.median()].max())))
    cam_near_far = torch.stack(cam_near_far, dim=0)
    near = cam_near_far[:, 0].min().item()
    far = cam_near_far[:, 1].max().item()

    del depth_covs, depths, cam_near_far

    return images, poses, [height, width], K, near, far