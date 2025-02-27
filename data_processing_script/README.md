# 💃 SMPL & Rendering

## Follow SMPL & Rendering in [Champ](https://github.com/fudan-generative-vision/champ/blob/master/docs/data_process.md) to configure the conda environment.

## Install dependencies

1. Install [4D-Humans](https://github.com/shubham-goel/4D-Humans)
    ```shell
    git clone https://github.com/shubham-goel/4D-Humans.git
    conda create --name 4D-humans python=3.10
    conda activate 4D-humans
    pip install -e 4D-Humans
    ```

    or you can install via pip by a simple command
    ```shell
    pip install git+https://github.com/shubham-goel/4D-Humans
    ```

2. Install [detectron2](https://github.com/facebookresearch/detectron2)
    
    gcc and g++ 12 is necessary to build detectron2

    ```shell
    conda install -c conda-forge gcc=12 gxx=12
    ```
    Then
    ```shell
    git clone https://github.com/facebookresearch/detectron2

    pip install -e detectron2
    ```
    or you can install via pip by a simple command
    ```shell
    pip install git+https://github.com/facebookresearch/detectron2
    ```

3. Install [Blender](https://www.blender.org/)

    You can download Blender 3.x version for your operation system from this url [https://download.blender.org/release/Blender3.6](https://download.blender.org/release/Blender3.6/).

## Download models

1. [DWPose for controlnet](https://github.com/IDEA-Research/DWPose?tab=readme-ov-file#-dwpose-for-controlnet)

    First, you need to download our Pose model dw-ll_ucoco_384.onnx ([baidu](https://pan.baidu.com/s/1nuBjw-KKSxD_BkpmwXUJiw?pwd=28d7), [google](https://drive.google.com/file/d/12L8E2oAgZy4VACGSK9RaZBZrfgx7VTA2/view?usp=sharing)) and Det model yolox_l.onnx ([baidu](https://pan.baidu.com/s/1fpfIVpv5ypo4c1bUlzkMYQ?pwd=mjdn), [google](https://drive.google.com/file/d/1w9pXC8tT0p9ndMN-CArp1__b2GbzewWI/view)), then put them into `${PROJECT_ROOT}/annotator/ckpts/`.

2. HMR2 checkpoints

    ```shell
    python -m scripts.pretrained_models.download --hmr2
    ```
3. Detectron2 model

    ```shell
    python -m scripts.pretrained_models.download --detectron2
    ```
4. SMPL model

    Please download the SMPL model from the official site [https://smpl.is.tue.mpg.de/download.php](https://smpl.is.tue.mpg.de/download.php).
    Then move the `.pkl` model to `4D-Humans/data`:
    ```shell
    mkdir -p 4D-Humans/data/
    mv basicModel_neutral_lbs_10_207_0_v1.0.0.pkl 4D-Humans/data/
    ```