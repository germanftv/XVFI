# Easy Video Frame Interpolation


**This is a forked repository of [XVFI (eXtreme Video Frame Interpolation)](https://github.com/JihyongOh/XVFI)**

This repository provides a pruned version of the original source code, designed exclusively to execute frame interpolation in custom videos with pretrained models. Its primary function is to easily increase the FPS of the input video with an off-the-shelf video interpolation method. The choice of [XVFI (ICCV2021)](https://openaccess.thecvf.com/content/ICCV2021/papers/Sim_XVFI_eXtreme_Video_Frame_Interpolation_ICCV_2021_paper.pdf) is motivated by its compelling visual results on custom videos. Notably, training-related components and metric performance utilities are excluded from this repository. Additionally, the code is reorganized as a python package that can be added as a submodule in other projects.


### Visual example of the XVFI (x4 Multi-Frame Interpolation) on custom video
<p align="center">
  <img width="800" src="figures/sample_result.gif">
</p>


## Table of Contents
1. [Installation](#Installation)
1. [Usage](#Usage)
1. [Acknowledgements](#Acknowledgements)
1. [License](#License)

## Installation
The code is implemented using PyTorch 1.9.0 and was tested on Ubuntu 20.04 machine (Python 3.8, CUDA 11.3).  
1. Clone repository
    ```bash
    git clone https://github.com/germanftv/XVFI
    cd XVFI
    ```
1. Create conda environment
    ```bash
    conda env create -n XVFI -f environment.yml
    ```
1. Install pip dependencies
    ```bash
    pip install -r pip_requirements.txt
    cd ..
    ```
1. Download pretrained models and place them as indicated:

    - [X4K1000FPS](https://www.dropbox.com/s/xj2ixvay0e5ldma/XVFInet_X4K1000FPS_exp1_latest.pt?dl=0), which was trained on X-TRAIN dataset must be located in `./XVFI/checkpoint_dir/XVFInet_X4K1000FPS_exp1`
        ```
        XVFI
        └── checkpoint_dir
            └── XVFInet_X4K1000FPS_exp1
                └── XVFInet_X4K1000FPS_exp1_latest.pt           
        ```
    - [Vimeo](https://www.dropbox.com/s/5v4dp81bto4x9xy/XVFInet_Vimeo_exp1_latest.pt?dl=0), which was trained on Vimeo90K dataset must be located in `./XVFI/checkpoint_dir/XVFInet_Vimeo_exp1`
        ```
        XVFI
        └── checkpoint_dir
            └── XVFInet_Vimeo_exp1
                └── XVFInet_Vimeo_exp1_latest.pt           
        ```



## Usage
<!-- ### Quick Start for your own video data ('--input_dir') for any Multi-Frame Interpolation (x M) -->
1. Extract the frames from your input video. With [ffmpeg](https://ffmpeg.org/ffmpeg.html), you can use the following command:
    ```bash
    ffmpeg -i $INPUT_VIDEO -start_number 0 -q:v 0 $INPUT_DIR/%08d.png 
    ```

    where:
    * `INPUT_VIDEO` is the path to your input video.
    * `INPUT_DIR` is the path to the directory that contains the extracted frames.


1. Run XVFI either by using the Command Line Interface (CLI) or importing the python package in your python script:

    **CLI:**
    ```bash
    python -m XVFI.main --input_dir $INPUT_DIR --output_dir $OUTPUT_DIR --multiple $VI_FACTOR --pretrained $PRETRAINED --gpu $GPU --config $CONFIG
    ```

    **Python script:**
     ```python
    import XVFI
    import argparse

    # Create an argument parser
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # Declare required inputs
    args.input_dir = INPUT_DIR
    args.output_dir = OUTPUT_DIR
    args.multiple = VI_FACTOR
    # Optional:
    args.pretrained = PRETRAINED
    args.gpu = GPU
    args.config = CONFIG

    # Add default arguments for XVFI
    args = XVFI.add_default_args(args, parser)

    # Run XVFI
    XVFI.run(args)
    ```

    where:
    * `INPUT_DIR` is the path to the directory that contains the input frames.
    * `OUTPUT_DIR` is the path to the directory that contains the frames for the interpolated video.
    * `VI_FACTOR` is the video interpolation factor. For instance, 8 results in a video with x8 FPS.
    * `PRETRAINED` refers to the pretrained weights to be used. Choices: `'X4K1000FPS'`, `'Vimeo'`. Default: `'X4K1000FPS'`.
    * `GPU` is an integer id to a gpu device. Default: 0.
    * `CONFIG` is the path to the config file. Default: `'./XVFI/configs/default.yaml'`.

        > **NOTE**:
        > In the [config file](configs/default.yaml), you can modify the lowest scale depth by regulating `'S_tst'`. By default, we use the test settings that are used in the original source code. The reader is refered to the [paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Sim_XVFI_eXtreme_Video_Frame_Interpolation_ICCV_2021_paper.pdf) for more details about this hyper-parameter.

1. **(Optional)**
Convert frames to video with ffmpeg using the command:
    ```bash
    ffmpeg -framerate 30 -pattern_type glob -i "$OUTPUT_DIR/*.png" -c:v libx264 -crf 1 -pix_fmt yuv420p $OUTPUT_VIDEO 
    ```
    where:
    * `OUTPUT_DIR` is the path to the directory that contains the frames for the interpolated video.
    * `OUTPUT_VIDEO` is the path interpolated video file with .mp4 extension.

## Acknowledgements
Kudos to the original contributors of XVFI: [Jihyong Oh](https://github.com/JihyongOh), [Hyeonjun Sim](https://github.com/hjSim), and the [Video & Image Computing Lab at KAIST](https://www.viclab.kaist.ac.kr/), who provided a [pytorch implementation](https://github.com/JihyongOh/XVFI) of the [paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Sim_XVFI_eXtreme_Video_Frame_Interpolation_ICCV_2021_paper.pdf).

## License
The source code can be used for research and education only as stated by the original authors.
