#!/bin/bash

file_path="build/"

if [ -d "$file_path" ]; then
    rm -r "$file_path"
fi

# 使用pytorch内置的CUDAExtension编译
# pip list中是spatha，但是import的时候是ssmm_mod
#python setup_torch.py install

# 使用cmake编译
# pip list中是ssmm-module，但是import的时候是ssmm_mod
#python setup_cmake.py install

# 默认使用setup.py编译，这是一个软链接，指向setup_cmake.py或者setup_torch.py
python setup.py install
