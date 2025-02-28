#!/bin/bash

# install dev-tools
apt install -y libgoogle-glog-dev libboost-all-dev

#########################################################################################

# build Samoyeds-Kernel and its baselines
cd Samoyeds-Kernel
./build.sh
cd ..

#########################################################################################
# build Samoyeds Model Executor

pip install torch==2.1.2

# install dependencies
pip install -r requirements.txt

#########################################################################################
# ColossalAI
pip install ./third_party/ColossalAI
pip install -r ./third_party/ColossalAI/examples/language/openmoe/requirements.txt

#########################################################################################
# patch dependencies
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

# 用cast.h文件替换<path-to-your-site-packages>/site-packages/torch/include/pybind11/cast.h文件
TARGET_DIR="$SITE_PACKAGES/torch/include/pybind11"
cp "patch/cast.h" "$TARGET_DIR/cast.h"

# 用deepseek.py文件替换<path-to-your-site-packages>/vllm/model_executor/models/deepseek.py
TARGET_DIR="$SITE_PACKAGES/vllm/model_executor/models"
cp "patch/deepseek.py" "$TARGET_DIR/deepseek.py"

# 用qwen2_moe.py文件替换<path-to-your-site-packages>/vllm/model_executor/models/qwen2_moe.py
cp "patch/qwen2_moe.py" "$TARGET_DIR/qwen2_moe.py"

#########################################################################################
# VENOM end-to-end
# sten
cd ./Samoyeds-Kernel/benchmark/third_party/venom/end2end/sten/
pip install .
# end-to-end
cd ..
./install_v64.sh
cd ../../../../../

#########################################################################################
# sparseml
cd sparseml && pip install -e . && cd ..

#########################################################################################
# reinstall dependencies for target version
pip install -r requirements.txt

#########################################################################################
# Samoyeds
if [ -d "build" ]; then
    rm -r "build"
fi
python setup.py install
