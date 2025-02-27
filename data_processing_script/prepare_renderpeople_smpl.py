import os
import sys
import cv2
import imageio
import torch
import numpy as np

import argparse
import time
import json
from smpl.smpl_numpy import SMPL

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument('-root_dir', '--root_dir', type=str, required=True)
parser.add_argument('-s', '--start_id', type=int, default=0)
parser.add_argument('-e', '--end_id', type=int, default=1)
args = parser.parse_args()

start_id = args.start_id
end_id = args.end_id

view_index_lst = [i for i in range(36)]

# root_dir = 'data/RenderPeople/test'
files_lst = sorted(os.listdir(root_dir))

def get_mask(mask_path):
    msk = imageio.imread(mask_path)
    msk[msk!=0]=255
    return msk

def prepare_smpl_params(smpl_path, pose_index):
    params_ori = dict(np.load(smpl_path, allow_pickle=True))['smpl'].item()
    params = {}
    params['shapes'] = np.array(params_ori['betas']).astype(np.float32)
    params['poses'] = np.zeros((1,72)).astype(np.float32)
    params['poses'][:, :3] = np.array(params_ori['global_orient'][pose_index]).astype(np.float32)
    params['poses'][:, 3:] = np.array(params_ori['body_pose'][pose_index]).astype(np.float32)
    params['R'] = np.eye(3).astype(np.float32)
    params['Th'] = np.array(params_ori['transl'][pose_index:pose_index+1]).astype(np.float32)
    return params

def prepare_input(smpl_path, pose_index, smpl_model):

    params = prepare_smpl_params(smpl_path, pose_index)
    xyz, _ = smpl_model(params['poses'], params['shapes'].reshape(-1))
    xyz = (np.matmul(xyz, params['R'].transpose()) + params['Th']).astype(np.float32)
    vertices = xyz

    # obtain the original bounds for point sampling
    min_xyz = np.min(xyz, axis=0)
    max_xyz = np.max(xyz, axis=0)
    min_xyz -= 0.05
    max_xyz += 0.05
    world_bounds = np.stack([min_xyz, max_xyz], axis=0)

    return world_bounds, vertices, params


for subject_name in files_lst[start_id:end_id]:
    print(subject_name)
    smpl_model = SMPL(sex='neutral', model_dir='assets/SMPL_NEUTRAL_renderpeople.pkl')

    vert_dct = {}

    for view_index in view_index_lst:

        smpl_overlay_lst = []
        for pose_index in range(21):

            # load reference image and mask
            # img_path = os.path.join(root_dir, subject_name + '/camera' + str(view_index).zfill(4) + '/images/' + str(pose_index).zfill(4)+'.jpg')            
            # mask_path = img_path.replace('images', 'msk').replace('jpg', 'png')

            # img = np.array(imageio.imread(img_path))
            # msk = np.array(get_mask(mask_path)) / 255.
            # img[msk == 0] = 0

            # Load reference K, R, T
            camera_file = root_dir + f'/{subject_name}/cameras.json'
            camera = json.load(open(camera_file))
            K = np.array(camera[f'camera{str(view_index).zfill(4)}']['K']).astype(np.float32)
            R = np.array(camera[f'camera{str(view_index).zfill(4)}']['R']).astype(np.float32)
            T = np.array(camera[f'camera{str(view_index).zfill(4)}']['T']).reshape(-1, 1).astype(np.float32)

            if pose_index not in vert_dct.keys():
                # load smplx data

                # prepare smpl at the reference view
                smpl_path = root_dir + f'/{subject_name}/outputs_re_fitting/refit_smpl_2nd.npz'
                _, world_vertex, smpl_param = prepare_input(smpl_path, pose_index, smpl_model)

                smpl_data = {}

                vert = np.array(world_vertex).reshape(-1,3).astype(np.float32)

                vert_dct[pose_index] = vert
            else:
                vert = vert_dct[pose_index]

            vert_cam = (np.matmul(vert, R.transpose()) + T.reshape(1,3)).astype(np.float32)

            # project vertex points to image space
            # xyz_c = torch.from_numpy(vert_cam)[None, None, :,:, None]
            # xyz = torch.matmul(torch.tensor(K)[None, None][:, :, None].float(), xyz_c)[..., 0]
            # xy = xyz[..., :2] / (xyz[..., 2:] + 1e-5)
            # src_uv = xy.view(-1, *xy.shape[2:])

            # test_image = torch.from_numpy(img.copy())
            # src_uv[0,:,1][src_uv[0,:,1]<0] = 0
            # src_uv[0,:,1][src_uv[0,:,1]>=test_image.shape[0]] = 0
            # src_uv[0,:,0][src_uv[0,:,0]<0] = 0
            # src_uv[0,:,0][src_uv[0,:,0]>=test_image.shape[1]] = 0
            # test_image[src_uv[0,:,1].type(torch.LongTensor), src_uv[0,:,0].type(torch.LongTensor)] = 1
            # output_smpl_overlay_dir = f'data/DNA_Rendering/Part_1/{smc_file_name}/camera{str(view_index).zfill(4)}/smpl_overlay'
            # os.makedirs(output_smpl_overlay_dir, exist_ok=True)
            # smpl_overlay_lst.append((255*test_image).numpy().astype(np.uint8))
            # cv2.imwrite(output_smpl_overlay_dir + "/{:06}.jpg".format(pose_index), (255*test_image).numpy().astype(np.uint8))
            # cv2.imwrite('test.jpg', (255*test_image).numpy().astype(np.uint8))

            rendering_dict = {}
            rendering_dict['verts'] = [vert_cam] 
            rendering_dict['cam_t'] = [np.zeros(3).astype(np.float32)]
            rendering_dict['render_res'] = np.array([512, 512])
            # rendering_dict['scaled_focal_length'] = K[0,0]
            rendering_dict['K'] = [K]
            # rendering_dict['faces'] = smpl_model.faces
            # save smpl results
            output_smpl_dir = f'{root_dir}/{subject_name}/camera{str(view_index).zfill(4)}/smpl_results'
            os.makedirs(output_smpl_dir, exist_ok=True)
            np.save(output_smpl_dir + "/{:04}.npy".format(pose_index), rendering_dict)

        # img_all = np.stack(smpl_overlay_lst, axis=0)
        # output_smpl_overlay_dir = f'{output_dir}/smpl_overlay'
        # os.makedirs(output_smpl_overlay_dir, exist_ok=True)
        # imageio.mimsave(output_smpl_overlay_dir + f'/{smc_file_name}_camera{str(view_index).zfill(4)}.mp4', img_all, fps=30, quality=8)