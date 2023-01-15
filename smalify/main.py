# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import os.path as osp
import time
import torch
import sys
sys.path.append(os.path.abspath(os.getcwd()))
import smplx_smal
import cv2
import numpy as np
from cmd_parser import parse_config
from data_parser import create_dataset
from fit_single_frame import fit_single_frame
from camera import create_camera
from prior import create_prior

def main(**args):
    output_folder = 'output'
    result_folder = output_folder+'/results'
    mesh_folder = output_folder+'/meshes'
    dataset_obj = create_dataset(**args)
    start = time.time()
    betas = torch.Tensor([[float(el) for el in args['zebra_betas']]])
    model_params = dict(model_path=args.get('model_folder'),
                        create_global_orient=not args.get('yaw_only'), 
                        create_body_pose=not args.get('use_vposer'),
                        create_betas=True,
                        create_transl=True,
                        betas=betas,
                        **args)
    body_model = smplx_smal.create(**model_params)
    cameras = []
    num_cameras = len(dataset_obj[0]["cam_names"][0])
    for _ in range(num_cameras):
        camera = create_camera(focal_length_x=args['focal_length'],
                               focal_length_y=args['focal_length'],
                               **args)
        camera.rotation_aa.requires_grad = False
        cameras.append(camera)
    body_pose_prior = create_prior(
        prior_type=args.get('body_prior_type'),
        **args)
    shape_prior = create_prior(
        prior_type=args.get('shape_prior_type', 'mahalanobis_shape'),
        **args)
    angle_prior = create_prior(prior_type='angle', dtype=args['dtype'])
    cam_pose_prior = create_prior(prior_type=args.get('cam_prior_type', 'l2'),**args)

    use_median_betas = args.get('use_median_betas', False)
    if use_median_betas:
        visualize = args.get('visualize', False)
        args['visualize'] = False
        all_betas = []

    for fix_betas in range(1+use_median_betas):
        if fix_betas:
            betas = torch.tensor(np.median(all_betas, axis=0))
            args['visualize'] = visualize
            dataset_obj = create_dataset(**args)

        losses = []
        prev_params = None
        for _, data in enumerate(dataset_obj):
            imgs = data['imgs'][0]
            keypoints = data['keypoints'][0]
            cam_poses = data['cam_poses'][0]
            img_scaled_w = 1280
            img_scaling_factor = imgs[0].shape[1] / img_scaled_w
            img_scaled_h = round(imgs[0].shape[0] / img_scaling_factor)

            for i, _ in enumerate(imgs):
                imgs[i] = cv2.resize(imgs[i], dsize=(img_scaled_w, img_scaled_h), interpolation=cv2.INTER_CUBIC)
                keypoints[i][0][:,:2] = keypoints[i][0][:,:2] / img_scaling_factor
            snapshot_name = data['snapshot_name']
            curr_image_folder = osp.join(output_folder, "images/", snapshot_name)
            print('Processing: {}'.format(snapshot_name))
            if not osp.exists(result_folder):
                os.makedirs(result_folder)
            if not osp.exists(mesh_folder):
                os.makedirs(mesh_folder)
            if not osp.exists(curr_image_folder):
                os.makedirs(curr_image_folder)

            results = fit_single_frame(imgs, keypoints, cam_poses,
                            body_model=body_model,
                            cameras=cameras,
                            snapshot_name=snapshot_name,
                            output_dir=output_folder,
                            shape_prior=shape_prior,
                            body_pose_prior=body_pose_prior,
                            angle_prior=angle_prior,
                            cam_pose_prior=cam_pose_prior,
                            betas=betas,
                            init_params=prev_params,
                            fix_betas=fix_betas,
                            **args)
            losses.append(results[0]['loss'])

            if args.get('init_prev'):
                assert len(results) <= 1, 'Can not use init_prev with more than a single subject'
                if results:
                    prev_params = results[0]['result']
                else:
                    prev_params = None

            if not fix_betas and use_median_betas:
                all_betas.append(prev_params['betas'])

        print(f'Mean loss: {np.mean(losses):.3e}')
                         
    elapsed = time.time() - start
    time_msg = time.strftime('%H hours, %M minutes, %S seconds',time.gmtime(elapsed))
    print('Processing the data took: {}'.format(time_msg))

if __name__ == "__main__":
    args = parse_config()
    main(**args)




