import cv2 
import os
import numpy as np 
from PIL import Image
import shutil

with open('data/RenderPeople/human_list.txt') as f:
    for subject_name in f.readlines()[:450]:
        subject_name = subject_name.strip()
        os.makedirs(f'data/RenderPeople/train/{subject_name}', exist_ok=True)
        shutil.copytree(f'data/RenderPeople/{subject_name}/outputs_re_fitting', f'data/RenderPeople/train/{subject_name}/outputs_re_fitting')
        shutil.copy(f'data/RenderPeople/{subject_name}/cameras.json', f'data/RenderPeople/train/{subject_name}/cameras.json')
        for camera_id in range(36):
            os.makedirs(f'data/RenderPeople/train/{subject_name}/camera{str(camera_id).zfill(4)}', exist_ok=True)
            shutil.copytree(f'data/RenderPeople/{subject_name}/img/camera{str(camera_id).zfill(4)}', f'data/RenderPeople/train/{subject_name}/camera{str(camera_id).zfill(4)}/images')
            shutil.copytree(f'data/RenderPeople/{subject_name}/mask/camera{str(camera_id).zfill(4)}', f'data/RenderPeople/train/{subject_name}/camera{str(camera_id).zfill(4)}/msk')

with open('data/RenderPeople/human_list.txt') as f:
    for subject_name in f.readlines()[450:]:
        subject_name = subject_name.strip()
        os.makedirs(f'data/RenderPeople/test/{subject_name}', exist_ok=True)
        shutil.copytree(f'data/RenderPeople/{subject_name}/outputs_re_fitting', f'data/RenderPeople/test/{subject_name}/outputs_re_fitting')
        shutil.copy(f'data/RenderPeople/{subject_name}/cameras.json', f'data/RenderPeople/test/{subject_name}/cameras.json')
        for camera_id in range(36):
            os.makedirs(f'data/RenderPeople/test/{subject_name}/camera{str(camera_id).zfill(4)}', exist_ok=True)
            shutil.copytree(f'data/RenderPeople/{subject_name}/img/camera{str(camera_id).zfill(4)}', f'data/RenderPeople/test/{subject_name}/camera{str(camera_id).zfill(4)}/images')
            shutil.copytree(f'data/RenderPeople/{subject_name}/mask/camera{str(camera_id).zfill(4)}', f'data/RenderPeople/test/{subject_name}/camera{str(camera_id).zfill(4)}/msk')
