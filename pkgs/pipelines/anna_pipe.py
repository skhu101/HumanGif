
import argparse
import asyncio
import datetime
import os
import sys
from pathlib import Path
from typing import Any

from pkgs.flow.pool.base import TaskPool
from pkgs.flow.pool.fifo_pool import FIFOPool
from pkgs.flow.pool.process_pool import ProcessPool
from pkgs.flow.task import Task
from pkgs.modules.audios.overlap_detector import OverlapDetector
from pkgs.modules.audios.single_voice_splitter import SingleVoiceSplitter
from pkgs.modules.human import detect_face
from pkgs.modules.video import detect_scene, has_audio, split_video, split_wav_audio
from pkgs.modules.videos.crop_video import crop_video
from pkgs.utils.fs import traverse_folder
from pkgs.utils.parallelism import get_data_frames

# 1、判断视频是否有声音（过滤掉没有声音的视频）
# 3、分离audio
# 4、parse_audio_overlap 检测音频的overlap片段 (G)
# 5、split_video 把4中检测出来的overlap其余部分切成clip
# 6、split_audio 把5中视频分离出audio
# 7、split_diarization 将一个人说话的片段单独截取出来 (G)
# 8、detect_scene 拆分场景
# 9、split_video 把8中的场景切分成clip
# 11、detect_face 检测视频中的人脸
# 12、crop_face 将视频片段中的人脸截取出来
# 13、face_forward 判断视频中的人脸朝向是否满足要求
# 15、lip_sync 对高分辨率的原视频片段进行声音-嘴唇匹配 过滤掉低匹配度的clips



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="/mnt/c/Users/leeway.zlw/Downloads/test_data/Bili", help="input dir")
    parser.add_argument("-o", "--output", default=".cache/output", help="output dir")
    parser.add_argument("--cpu_count", type=int, help="output dir")
    parser.add_argument("-r", "--rank", type=int, default=0, help="rank")
    parser.add_argument("-p", "--parallelism", type=int, default=1, help="并发数量")

    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    cpu_count = args.cpu_count if args.cpu_count else os.cpu_count()

    parallelism = args.parallelism
    rank = args.rank

    os.makedirs(output_dir, exist_ok=True)

    # 按文件从小到大排序
    video_files = traverse_folder(input_dir, True, lambda x: x.endswith(".mp4"))
    files = get_data_frames(video_files, parallelism, rank)

    print(f"cpu count: {cpu_count}")

    s1_filter_with_audio = ProcessPool(
        parallelism=max(cpu_count // 3, 1),
        name="filter_with_audio",
    )

    s3_split_audio = ProcessPool(
        parallelism=max(cpu_count // 3, 1),
        name="split_audio",
    )
    s3_dir = os.path.join(args.output, "stage3")
    os.makedirs(s3_dir, exist_ok=True)

    s4_parse_audio_overlap = FIFOPool(
        parallelism=1,
        name="parse_audio_overlap",
    )
    s4_parse_audio_overlap.pre_hook(lambda : {
        "overlap_detector": OverlapDetector(
            checkpoint_path="pretrained_models/overlap_detect/overlap_detect.yaml",
        ).load_model("cuda:0")
    })
    s4_dir = os.path.join(args.output, "stage4")
    os.makedirs(s4_dir, exist_ok=True)

    s5_split_no_overlap = ProcessPool(
        parallelism=max(cpu_count // 3, 1),
        name="split_no_overlap",
    )
    s5_dir = os.path.join(args.output, "stage5")
    os.makedirs(s5_dir, exist_ok=True)

    s6_split_audio = ProcessPool(
        parallelism=max(cpu_count // 3, 1),
        name="split_audio_no_overlap",
    )
    s6_dir = os.path.join(args.output, "stage6")
    os.makedirs(s6_dir, exist_ok=True)

    s7_parse_single_voice = FIFOPool(
        parallelism=1,
        name="parse_single_voice",
    )
    s7_dir = os.path.join(args.output, "stage7")
    os.makedirs(s7_dir, exist_ok=True)
    s7_parse_single_voice.pre_hook(
        lambda : {
            "single_voice_splitter": SingleVoiceSplitter(
                checkpoint_path="pretrained_models/speaker-diarization/config.yaml",
            ).load_model("cuda:0")
        }
    )

    s8_scene_detect = ProcessPool(
        parallelism=max(cpu_count // 3, 1),
        name="scene_detect",
    )
    s8_dir = os.path.join(args.output, "stage8")
    os.makedirs(s8_dir, exist_ok=True)

    s9_split_video = ProcessPool(
        parallelism=max(cpu_count // 3, 1),
        name="split_video_to_scenes"
    )
    s9_dir = os.path.join(args.output, "stage9")
    os.makedirs(s9_dir, exist_ok=True)

    # s10_detect_motion = ProcessPool(
    #     parallelism=max(cpu_count // 3, 1),
    #     name="detect_motion",
    # )
    # s10_dir = os.path.join(args.output, "stage10")
    # os.makedirs(s10_dir, exist_ok=True)

    s11_detect_face = ProcessPool(
        parallelism=max(cpu_count // 3, 1),
        name="detect_face",
    )
    s11_dir = os.path.join(args.output, "stage11")
    os.makedirs(s11_dir, exist_ok=True)

    s12_crop_face = ProcessPool(
        parallelism=max(cpu_count // 3, 1),
        name="crop_face"
    )
    s12_dir = os.path.join(args.output, "stage12")
    os.makedirs(s12_dir, exist_ok=True)

    # s1 --> s2
    def s1_s3_pipe(_: TaskPool, task: Task, has_audio: bool):
        if has_audio:
            video_file = task.kwargs["video_file"]
            video_name = Path(video_file).name.split(".")[0]
            audio_file = os.path.join(s3_dir, f"{video_name}.wav")
            s3_split_audio.schedule(Task(
                name=f"stage3: {video_file}",
                fn=split_wav_audio,
                video_file=video_file,
                output_file=audio_file,
            ))
    s1_filter_with_audio.on_progress(s1_s3_pipe)

    # # s2 --> s3
    # def s2_s3_pipe(_: TaskPool, _task: Task, video_file: str):
    #     video_name = Path(video_file).name.split(".")[0]
    #     audio_file = os.path.join(s3_dir, f"{video_name}.wav")
    #     s3_split_audio.schedule(Task(
    #         name=f"stage3: {video_file}",
    #         fn=split_wav_audio,
    #         video_file=video_file,
    #         output_file=audio_file,
    #     ))
    # s2_down_res.on_progress(s2_s3_pipe)

    # s3 --> s4
    def s3_s4_pipe(_: TaskPool, task: Task, audio_file: str):
        video_file = task.get_argument("video_file", 0)
        video_name = Path(video_file).name.split(".")[0]
        audio_file = os.path.join(s3_dir, f"{video_name}.wav")
        def parse(overlap_detector: OverlapDetector, *args, **kwargs):
            return overlap_detector.process(*args, **kwargs)
        s4_parse_audio_overlap.schedule(Task(
            name=f"stage4: {video_file}",
            fn=parse,
            audio_file=audio_file,
            output_json_file=os.path.join(s4_dir, f"{video_name}.json")
        ))
    s3_split_audio.on_progress(s3_s4_pipe)

    # s4 --> s5
    def s4_s5_pipe(_: TaskPool, _task: Task, result: dict[str, Any]):
        audio_file: str = result.get("audio_file")
        no_overlap_ranges: list[tuple[float, float]] = result.get("no_overlap_ranges")
        video_name = Path(audio_file).name.split(".")[0]
        video_file = os.path.join(input_dir, f"{video_name}.mp4")
        s5_split_no_overlap.schedule(Task(
            name=f"stage5: {video_file}",
            fn=split_video,
            video_file=video_file,
            segments=no_overlap_ranges,
            output_dir=s5_dir,
            name_template="$NAME-no_overlap-$FRAME_START-$FRAME_END.$EXT"
        ))
    s4_parse_audio_overlap.on_progress(s4_s5_pipe)

    # s5 --> s6
    def s5_s6_pipe(_:TaskPool, _task: Task, video_files: list[str]):
        for video_file in video_files:
            video_name = Path(video_file).name.split(".")[0]
            s6_split_audio.schedule(Task(
                name=f"stage6: {video_file}",
                fn=split_wav_audio,
                video_file=video_file,
                output_file=os.path.join(s6_dir, f"{video_name}.wav")
            ))
    s5_split_no_overlap.on_progress(s5_s6_pipe)

    # s6 --> s7
    def s6_s7_pipe(_: TaskPool, _task: Task, audio_file: str):
        filename = Path(audio_file).name.split(".")[0]
        def parse(single_voice_splitter: SingleVoiceSplitter, *args, **kwargs):
            return single_voice_splitter.process(*args, **kwargs)
        s7_parse_single_voice.schedule(Task(
            name=f"stage7: {audio_file}",
            fn=parse,
            audio_file=audio_file,
            output_json_file=os.path.join(s7_dir, f"{filename}.json")
        ))
    s6_split_audio.on_progress(s6_s7_pipe)

    # s7 --> s8
    def s7_s8_pipe(_: TaskPool, _task: Task, result: dict[str, Any]):
        audio_file = result.get("audio_file")
        segments = result.get("segments")
        video_name = Path(audio_file).name.split(".")[0]
        video_file = os.path.join(s5_dir, f"{video_name}.mp4")
        for seg in segments:
            s8_scene_detect.schedule(Task(
                name=f"stage8: {video_file}",
                fn=detect_scene,
                video_file=video_file,
                start_time=seg[0],
                end_time=seg[1],
                adaptive_threshold=0.5,
                min_scene_duration=2,
                max_scene_duration=100,
                output_json_file=os.path.join(s8_dir, f"{video_name}.json")
            ))
    s7_parse_single_voice.on_progress(s7_s8_pipe)

    # s8 --> s9
    def s8_s9_pipe(_: TaskPool, _task: Task, result: dict[str, Any]):
        video_file = result.get("video_file")
        segments = result.get("segments")

        s9_split_video.schedule(Task(
            name=f"stage9: {video_file}",
            fn=split_video,
            video_file=video_file,
            segments=segments,
            output_dir=s9_dir,
            name_template="$NAME-scene-$FRAME_START-$FRAME_END.$EXT"
        ))
    s8_scene_detect.on_progress(s8_s9_pipe)

    # s9 --> s10
    def s9_s10_pipe(_: TaskPool, _task: Task, video_files: list[str]):
        for video_file in video_files:
            video_name = Path(video_file).name.split(".")[0]
            s11_detect_face.schedule(Task(
                name=f"stage11: {video_file}",
                fn=detect_face,
                video_file=video_file,
                least_frames=20,
                score_threshold=0.6,
                save_path=os.path.join(s11_dir, f"{video_name}.json")
            ))
            # s10_detect_motion.schedule(Task(
            #     name=f"stage10: {video_file}",
            #     fn=detect_motion,
            #     video_file=video_file,
            #     output_json_file=os.path.join(s10_dir, f"{video_name}.json")
            # ))
    s9_split_video.on_progress(s9_s10_pipe)

    # s10 --> s11
    # def s10_s11_pipe(_: TaskPool, _task: Task, result: dict[str, Any]):
    #     video_file = result.get("video_file")
    #     motion_detected = result.get("motion_detected")
    #     video_name = Path(video_file).name.split(".")[0]
    #     if not motion_detected:
    #         s11_detect_face.schedule(Task(
    #             name=f"stage11: {video_file}",
    #             fn=detect_face,
    #             video_file=video_file,
    #             least_frames=20,
    #             score_threshold=0.6,
    #             save_path=os.path.join(s11_dir, f"{video_name}.json")
    #         ))
    # s10_detect_motion.on_progress(s10_s11_pipe)

    def s11_s12_pipe(_: TaskPool, _task: Task, result: dict[str, Any]):
        video_file = result.get("video_file")
        segments = result.get("segments")
        video_name = Path(video_file).name
        output_file = os.path.join(s12_dir, video_name)

        if len(segments) != 1:
            return

        bboxes = list(map(lambda x: (x["bounding_box"]["origin_x"], x["bounding_box"]["origin_y"], x["bounding_box"]["width"], x["bounding_box"]["height"]), segments[0]["detections"]))
        s12_crop_face.schedule(Task(
            name=f"stage12: {video_file}",
            fn=crop_video,
            video_file=video_file,
            output_file=output_file,
            bboxes=bboxes,
            expansion_ratio=1.5,
        ))
    s11_detect_face.on_progress(s11_s12_pipe)

    # start from s1
    for video_file in files:
        video_name = Path(video_file).name
        s1_filter_with_audio.schedule(Task(
            name=f"stage1: {video_file}",
            fn=has_audio,
            video_file=video_file
        ))

    start = datetime.datetime.now()

    asyncio.run(s1_filter_with_audio.wait_for())
    s1_filter_with_audio.close()
    print("stage1 ALL DONE")

    asyncio.run(s3_split_audio.wait_for())
    s3_split_audio.close()
    print("stage3 ALL DONE")

    asyncio.run(s4_parse_audio_overlap.wait_for())
    s4_parse_audio_overlap.close()
    print("stage4 ALL DONE")

    asyncio.run(s5_split_no_overlap.wait_for())
    s5_split_no_overlap.close()
    print("stage5 ALL DONE")

    asyncio.run(s6_split_audio.wait_for())
    s6_split_audio.close()
    print("stage6 ALL DONE")

    asyncio.run(s7_parse_single_voice.wait_for())
    s7_parse_single_voice.close()
    print("stage7 ALL DONE")

    asyncio.run(s8_scene_detect.wait_for())
    s8_scene_detect.close()
    print("stage8 ALL DONE")

    asyncio.run(s9_split_video.wait_for())
    s9_split_video.close()
    print("stage9 ALL DONE")

    # asyncio.run(s10_detect_motion.wait_for())
    # s10_detect_motion.close()
    # print("stage10 ALL DONE")

    asyncio.run(s11_detect_face.wait_for())
    s11_detect_face.close()
    print("stage11 ALL DONE")

    asyncio.run(s12_crop_face.wait_for())
    s12_crop_face.close()
    print("stage12 ALL DONE")

    print(f"COST: {datetime.datetime.now() - start}")
    sys.exit(0)
