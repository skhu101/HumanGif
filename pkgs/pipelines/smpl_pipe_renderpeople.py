import argparse
import asyncio
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable
import torch
import numpy as np 

sys.path.append(str(Path(__file__).parent.parent))

from flow.pool.base import TaskPool
from flow.pool.fifo_pool import FIFOPool
from flow.pool.process_pool import ProcessPool
from flow.task import Task

from tqdm import tqdm

# from modules.smpl.generate_smpls import generate_smpl, load_model

def render_smpls(video_dir: str, skip_render=False):
    if not skip_render:
        cmd = [
            "../blender-3.6.10-linux-x64/blender", "pkgs/modules/smpl/blend/smpl_rendering.blend", "--background", "--python", "pkgs/modules/smpl/render_condition_maps.py",
            "--driving_path", os.path.join(video_dir, "smpl_results"),
        ]
        print(f"{' '.join(cmd)}")
        p = subprocess.Popen(cmd)
        ret = p.wait()
        if ret != 0:
            print(f"ERROR: transfer_smpls video_dir={video_dir} ret={ret}")
    else:
        pass

def generate_smpls(video_dir: str, device, model=None, model_cfg=None, detector=None, skip_fit=False):
    image_dir = os.path.join(video_dir, "images")
    images = sorted(os.listdir(image_dir))
    # if not skip_fit:
    #     for image in tqdm(images):
    #         generate_smpl(
    #             image_file=os.path.join(image_dir, image),
    #             output_dir=os.path.join(video_dir, "smpl_results"),
    #             device=device,
    #             model=model,
    #             model_cfg=model_cfg,
    #             detector=detector,
    #         )
    return [video_dir]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="input videos dir")
    parser.add_argument("-d", "--device", default=0, help="device")
    
    parser.add_argument('--skip_fit', action='store_true', default=False, help='skip smpl fitting stage')
    parser.add_argument('--skip_render', action='store_true', default=False, help='skip smpl render stage')

    parser.add_argument("--cpu_count", type=int, help="output dir")
    parser.add_argument("-r", "--rank", type=int, default=0, help="rank")
    parser.add_argument("-p", "--parallelism", type=int, default=1, help="并发数量")

    parser.add_argument("-s", "--start_id", type=int, default=0)
    parser.add_argument("-e", "--end_id", type=int, default=1)

    args = parser.parse_args()

    # 按文件从小到大排序
    # original_files = map(
    #     lambda item: os.path.join(args.input, item),
    #     sorted(os.listdir(args.input)),
    # )
    # original_files = sorted(original_files, key=lambda f: os.stat(f).st_size)
    # original_files = [args.input]

    root_dir = args.input
    files_lst = sorted(os.listdir(root_dir))

    view_index_lst = [i for i in range(0,36)]
    original_files = []
    for subject_name in files_lst:
        for view_index in view_index_lst:
            original_files.append(args.input + f'/{subject_name}/camera{str(view_index).zfill(4)}')
    original_files = original_files[args.start_id*len(view_index_lst):args.end_id*len(view_index_lst)]

    # 根据rank分批
    pick = 0
    files = []
    for f in original_files:
        if pick % args.parallelism == args.rank:
            files.append(f)
        pick += 1

    cpu_count = args.cpu_count if args.cpu_count else os.cpu_count()

    pipeline = [
        {
            "task_fn": generate_smpls,
            "pool": FIFOPool(name="GENERATE_SMPL", parallelism=2),
        },
        {
            "task_fn": render_smpls,
            "pool": ProcessPool(name="RENDER_SMPL", parallelism=max(cpu_count // 3, 1)),
        },
    ]

    def on_next_step(next_pool: TaskPool, next_fn: Callable):
        def process(_pool: TaskPool, _task: Task, results):
            next_pool.schedule(Task(
                f"{next_pool.name} - {results[0]}",
                next_fn,
                *results
            ))
        return process
    reversed_pipe = list(reversed(pipeline))

    # model, model_cfg, detector = load_model(args.device)
    model, model_cfg, detector = None, None, None

    # smpl_file_lst = sorted(os.listdir(args.input+'/smpl_results'))
    # for smpl_file in smpl_file_lst:
    #     smpl_path = args.input #.replace('driving_videos_smooth', 'driving_videos')
    #     smpl_data = np.load(f'{smpl_path}/smpl_results/{smpl_file}', allow_pickle=True).item()
        # smpl_parameter = smpl_data['smpls']

        # smpl_path_2 = args.input.replace('driving_videos_smooth', 'driving_videos')
        # smpl_data_2 = np.load(f'{smpl_path_2}/smpl_results/{smpl_file}', allow_pickle=True).item()
        # smpl_parameter_2 = smpl_data_2['smpls']

        # smpl_output = model.smpl(**{k: torch.from_numpy(v).float().cuda() for k, v in smpl_parameter.items()}, pose2rot=False,)
        # pred_vertices = smpl_output.vertices
        # verts = pred_vertices.detach().cpu().numpy().reshape(-1,3)
        # smpl_data['verts'] = [verts]
        # np.save(f'{args.input}/smpl_results/{smpl_file}', smpl_data)

    for i, step in enumerate(reversed_pipe):
        pool: TaskPool = step["pool"]
        print(f"{i}: {pool.name}")

        if i > 0:
            next_step = reversed_pipe[i - 1]
            next_pool: TaskPool = next_step["pool"]
            pool.on_progress(on_next_step(next_pool, next_step["task_fn"]))

        # 第一个step 启动任务
        if i == len(pipeline) - 1:
            for video_dir in files:
                video_name = Path(video_dir).name.split(".")[0]

                pool.schedule(Task(
                    name=f"{pool.name} - {video_dir}",
                    fn=step["task_fn"],
                    video_dir=video_dir,
                    device=args.device,
                    model=model,
                    model_cfg=model_cfg,
                    detector=detector,
                    skip_fit=args.skip_fit,
                ))
    for step in pipeline:
        pool: TaskPool = step["pool"]
        asyncio.run(pool.wait_for())
