import numpy as np
import os, imageio
import json
from img_utils import load_img, tonemap


def load_blender_video_data(args):
    imgs = []
    poses = []
    exposures = []
    start_frame = args.start_frame
    end_frame = args.end_frame

    with open(os.path.join(args.datadir, 'transform.json'), 'r') as fp:
        meta = json.load(fp)
    
    if not args.autoexp:
        hdr = True
        if args.blur:
            print("\033[96m Loading motion blur + fixed exposure images \033[0m")
            img_dir = os.path.join(args.datadir, 'HDR_images')
            global_tonemap = True
        else:
            print("\033[96m Loading sharp + fixed exposure images \033[0m")
            img_dir = os.path.join(args.datadir, 'GT_images')
            global_tonemap = True
    else:
        hdr = False
        if args.blur:
            print("\033[96m Loading motion blur + auto exposure images \033[0m")
            img_dir = os.path.join(args.datadir, 'LDR_images')
            global_tonemap = False
        else:
            print("\033[96m Loading sharp + auto exposure images \033[0m")
            img_dir = os.path.join(args.datadir, 'sharp_LDR_images')
            global_tonemap = False

    img_extension = "exr" if hdr else "png"
    
    for i in range(start_frame, end_frame + 1):
        imgs.append(load_img(os.path.join(img_dir, f"{i:04d}.{img_extension}"), hdr=hdr))
        poses.append(np.array(meta['frames'][i]['transform_matrix']))
        exposures.append(meta['frames'][i]['exposure_value'])

    imgs = np.array(imgs)
    poses = np.array(poses).astype(np.float32)
    exposures = np.array(exposures)

    if global_tonemap:
        imgs = tonemap(imgs, exposure=np.mean(exposures), gamma=2.2)
        print(f"\033[96m average exposure: {np.mean(exposures)}\033[0m")

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    # return imgs, poses, [H, W, focal], i_split
    return imgs, poses, [H, W, focal]
