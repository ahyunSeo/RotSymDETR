#!/bin/bash
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118 https://download.pytorch.org/whl/cu118/torch-2.1.0%2Bcu118-cp38-cp38-linux_x86_64.whl https://download.pytorch.org/whl/cu118/torchvision-0.16.0%2Bcu118-cp38-cp38-linux_x86_64.whl#sha256=682ae598fccef9cd1935e5d1546138b1a23c8d78adc8f96c899385dee7c7db90
pip install mmcv==2.1.0

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
cd ..

git clone -b main https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -v -e .
cd ..
pip install ftfy regex timm albumentations

