#!/bin/bash
# 第一次使用前须在终端运行 chmod +x startup.sh
# pip install uv -i https://pypi.tuna.tsinghua.edu.cn/simple
conda create -n radarPointCloudMap python=3.12
conda activate radarPointCloudMap
conda install opencv
pip install open3d
conda deactivate