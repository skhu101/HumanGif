import os
import sys
import cv2
import imageio
import torch
import numpy as np
from smplx.body_models import SMPLX
from data.DNA_Rendering.dna_rendering_sample_code.SMCReader import SMCReader

import argparse

import time

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--start_id', type=int, default=0)
parser.add_argument('-e', '--end_id', type=int, default=1)
args = parser.parse_args()

start_id = args.start_id
end_id = args.end_id

output_dir = 'data/DNA_Rendering/Part_1'
smc_dir = output_dir + '/dna_rendering_part1_main/'
smc_files_lst = sorted(os.listdir(smc_dir))
print('total num: ', len(smc_files_lst))

smc_files_lst = []
with open(output_dir + '/smc_lst.txt') as f:
    for line in f.readlines():
        line = line.strip()
        smc_files_lst.append(line + '.smc')
print('total num: ', len(smc_files_lst))

view_index_lst = [i for i in range(0,48)]

for smc_file in smc_files_lst[start_id:end_id]:
    path = smc_dir + smc_file
    smc_reader = SMCReader(path)
    annots_file_path = path.replace('main', 'annotations').split('.')[0] + '_annots.smc'
    smc_annots_reader = SMCReader(annots_file_path)
    print(smc_file, smc_reader.Camera_5mp_info)

    smpl_model = {}
    gender = smc_reader.actor_info['gender']
    smpl_model[gender] = SMPLX('assets/models/smplx/', smpl_type='smplx',
                                gender=gender, use_face_contour=True, flat_hand_mean=False, use_pca=False, 
                                num_pca_comps=24, num_betas=10,
                                num_expression_coeffs=10,
                                ext='npz')

    vert_dct = {}

    for view_index in view_index_lst:

        smc_file_name = smc_file.split('.')[0]
        output_image_dir = f'{output_dir}/data_render/{smc_file_name}/camera{str(view_index).zfill(4)}'
        # if os.path.exists(output_image_dir):
        if os.path.exists(output_image_dir + '/images'):
            img_num = len(os.listdir(output_image_dir + '/images'))
            print('smc: ', smc_file, ' view_index: ', view_index, " total frame: ", smc_reader.Camera_5mp_info['num_frame'], " acutal frame: ", img_num)
            if img_num == smc_reader.Camera_5mp_info['num_frame']:
                continue

        # load image 
        image_original = smc_reader.get_img('Camera_5mp', int(view_index), Image_type='color')
        # laod smplx 
        smpl_dict = smc_annots_reader.get_SMPLx(Frame_id=None)
        # if smpl_dict['fullpose'].shape[0] != smc_reader.Camera_5mp_info['num_frame'] or smpl_dict['transl'].shape[0] != smc_reader.Camera_5mp_info['num_frame'] or smpl_dict['betas'].shape[0] != smc_reader.Camera_5mp_info['num_frame'] or smpl_dict['expression'].shape[0] != smc_reader.Camera_5mp_info['num_frame']:
        #     continue

        smpl_overlay_lst = []
        for pose_index in range(smc_reader.Camera_5mp_info['num_frame']):

            smc_file_name = smc_file.split('.')[0]
            output_image_dir = f'{output_dir}/data_render/{smc_file_name}/camera{str(view_index).zfill(4)}/images'
            output_smpl_dir = f'{output_dir}/data_render/{smc_file_name}/camera{str(view_index).zfill(4)}/smpl_results'
            if os.path.exists(output_image_dir + "/{:06}.jpg".format(pose_index)) and os.path.exists(output_smpl_dir + "/{:06}.npy".format(pose_index)):
                continue

            print('smc: ', smc_file, ' view_index: ', view_index, " pose_index: ", pose_index, " actual num: ", smc_reader.Camera_5mp_info['num_frame'])

            # Load K, R, T
            cam_params = smc_annots_reader.get_Calibration(view_index)
            K = cam_params['K']
            D = cam_params['D'] # k1, k2, p1, p2, k3
            RT = cam_params['RT']
            R = RT[:3, :3]
            T = RT[:3, 3]

            # load camera 
            c2w = np.eye(4)
            c2w[:3,:3] = R
            c2w[:3,3:4] = T.reshape(-1, 1)
            w2c = np.linalg.inv(c2w)

            # load mask 
            # msk = smc_annots_reader.get_mask(view_index, Frame_id=pose_index)
            # msk[msk!=0] = 255
            # msk = np.array(msk) / 255.

            image = np.array(image_original[pose_index]) 

            image = image / 255. # cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.

            # undistort image and mask
            image = cv2.undistort(image, K, D)
            # msk = cv2.undistort(msk, K, D)

            ratio = 0.375
            if ratio != 1.:
                H, W = int(image.shape[0] * ratio), int(image.shape[1] * ratio)
                image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
                # msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
                K[:2] = K[:2] * ratio

            if pose_index not in vert_dct.keys():
                # load smplx data

                smpl_data = {}
                smpl_data['global_orient'] = smpl_dict['fullpose'][pose_index, 0].reshape(-1)
                smpl_data['body_pose'] = smpl_dict['fullpose'][pose_index, 1:22].reshape(-1)
                smpl_data['jaw_pose'] = smpl_dict['fullpose'][pose_index, 22].reshape(-1)
                smpl_data['leye_pose'] = smpl_dict['fullpose'][pose_index, 23].reshape(-1)
                smpl_data['reye_pose'] = smpl_dict['fullpose'][pose_index, 24].reshape(-1)
                smpl_data['left_hand_pose'] = smpl_dict['fullpose'][pose_index, 25:40].reshape(-1)
                smpl_data['right_hand_pose'] = smpl_dict['fullpose'][pose_index, 40:55].reshape(-1)
                smpl_data['transl'] = smpl_dict['transl'][pose_index].reshape(-1)
                smpl_data['betas'] = smpl_dict['betas'][pose_index].reshape(-1)
                smpl_data['expression'] = smpl_dict['expression'][pose_index].reshape(-1)

                smpl_param = {
                    'global_orient': np.expand_dims(smpl_data['global_orient'].astype(np.float32), axis=0),
                    'transl': np.expand_dims(smpl_data['transl'].astype(np.float32), axis=0),
                    'body_pose': np.expand_dims(smpl_data['body_pose'].astype(np.float32), axis=0),
                    'jaw_pose': np.expand_dims(smpl_data['jaw_pose'].astype(np.float32), axis=0),
                    'betas': np.expand_dims(smpl_data['betas'].astype(np.float32), axis=0),
                    'expression': np.expand_dims(smpl_data['expression'].astype(np.float32), axis=0),
                    'leye_pose': np.expand_dims(smpl_data['leye_pose'].astype(np.float32), axis=0),
                    'reye_pose': np.expand_dims(smpl_data['reye_pose'].astype(np.float32), axis=0),
                    'left_hand_pose': np.expand_dims(smpl_data['left_hand_pose'].astype(np.float32), axis=0),
                    'right_hand_pose': np.expand_dims(smpl_data['right_hand_pose'].astype(np.float32), axis=0),
                    }

                smpl_param['R'] = np.eye(3).astype(np.float32)
                smpl_param['Th'] = smpl_param['transl'].astype(np.float32)
                smpl_param_tensor = {}
                for key in smpl_param.keys():
                    smpl_param_tensor[key] = torch.from_numpy(smpl_param[key])
                body_model_output = smpl_model[gender](
                    global_orient=smpl_param_tensor['global_orient'],
                    betas=smpl_param_tensor['betas'],
                    body_pose=smpl_param_tensor['body_pose'],
                    jaw_pose=smpl_param_tensor['jaw_pose'],
                    left_hand_pose=smpl_param_tensor['left_hand_pose'],
                    right_hand_pose=smpl_param_tensor['right_hand_pose'],
                    leye_pose=smpl_param_tensor['leye_pose'],
                    reye_pose=smpl_param_tensor['reye_pose'],
                    expression=smpl_param_tensor['expression'],
                    transl=smpl_param_tensor['transl'],
                    return_full_pose=True,
                )

                # smpl_param['poses'] = body_model_output.full_pose.detach()
                # smpl_param['shapes'] = np.concatenate([smpl_param['betas'], smpl_param['expression']], axis=-1)
                vert = np.array(body_model_output.vertices.detach()).reshape(-1,3).astype(np.float32)

                vert_dct[pose_index] = vert
            else:
                vert = vert_dct[pose_index]

            vert_cam = (np.matmul(vert, w2c[:3,:3].transpose()) + w2c[:3,3]).astype(np.float32)

            # # project vertex points to image space
            # # RT = torch.cat([torch.tensor(w2c[:3,:3].transpose()), torch.tensor(w2c[:3,3]).reshape(3,1)], -1)[None, None]
            # # xyz_repeat = torch.repeat_interleave(torch.tensor(vert)[None, None], repeats=RT.shape[1], dim=1) #[bs, view_num, , 3]
            # # xyz_c = torch.matmul(RT[:, :, None, :, :3].float(), xyz_repeat[..., None].float()) + RT[:, :, None, :, 3:].float()
            # xyz_c = torch.from_numpy(vert_cam)[None, None, :,:, None]
            # xyz = torch.matmul(torch.tensor(K)[None, None][:, :, None].float(), xyz_c)[..., 0]
            # xy = xyz[..., :2] / (xyz[..., 2:] + 1e-5)
            # src_uv = xy.view(-1, *xy.shape[2:])

            # smc_file_name = smc_file.split('.')[0]
            # test_image = torch.from_numpy(image.copy())
            # src_uv[0,:,1][src_uv[0,:,1]<0] = 0
            # src_uv[0,:,1][src_uv[0,:,1]>=test_image.shape[0]] = 0
            # src_uv[0,:,0][src_uv[0,:,0]<0] = 0
            # src_uv[0,:,0][src_uv[0,:,0]>=test_image.shape[1]] = 0
            # test_image[src_uv[0,:,1].type(torch.LongTensor), src_uv[0,:,0].type(torch.LongTensor)] = 1
            # # output_smpl_overlay_dir = f'data/DNA_Rendering/Part_1/{smc_file_name}/camera{str(view_index).zfill(4)}/smpl_overlay'
            # # os.makedirs(output_smpl_overlay_dir, exist_ok=True)
            # smpl_overlay_lst.append((255*test_image).numpy().astype(np.uint8))
            # # cv2.imwrite(output_smpl_overlay_dir + "/{:06}.jpg".format(pose_index), (255*test_image).numpy().astype(np.uint8))

            rendering_dict = {}
            rendering_dict['verts'] = [vert_cam] 
            rendering_dict['cam_t'] = [np.zeros(3).astype(np.float32)]
            rendering_dict['render_res'] = smc_reader.Camera_5mp_info['resolution'] * ratio
            rendering_dict['K'] = [K]
            rendering_dict['faces'] = smpl_model[gender].faces
            # # save smpl results
            smc_file_name = smc_file.split('.')[0]
            output_smpl_dir = f'{output_dir}/data_render/{smc_file_name}/camera{str(view_index).zfill(4)}/smpl_results'
            os.makedirs(output_smpl_dir, exist_ok=True)
            np.save(output_smpl_dir + "/{:06}.npy".format(pose_index), rendering_dict)
            # save image results
            output_image_dir = f'{output_dir}/data_render/{smc_file_name}/camera{str(view_index).zfill(4)}/images'
            os.makedirs(output_image_dir, exist_ok=True)
            cv2.imwrite(output_image_dir + "/{:06}.jpg".format(pose_index), (image*255).astype('uint8'))
            # del rendering_dict, image

        # img_all = np.stack(smpl_overlay_lst, axis=0)
        # output_smpl_overlay_dir = f'{output_dir}/smpl_overlay'
        # os.makedirs(output_smpl_overlay_dir, exist_ok=True)
        # imageio.mimsave(output_smpl_overlay_dir + f'/{smc_file_name}_camera{str(view_index).zfill(4)}.mp4', img_all, fps=30, quality=8)