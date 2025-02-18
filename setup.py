import multiprocessing
import os
import platform
import subprocess
import sys

from setuptools import find_packages
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

def get_cuda_compute_capability():
    try:
        # 调用 nvidia-smi 获取 GPU 计算能力
        gpu_cc = subprocess.check_output(
            ["nvidia-smi", "--id=0", "--query-gpu=compute_cap", "--format=csv,noheader"],
            universal_newlines=True
        ).strip()

        # 根据计算能力设置 CUDA_COMPUTE_CAPABILITY
        if gpu_cc == "8.0":
            return "80"
        elif gpu_cc == "8.6":
            return "86"
        elif gpu_cc == "8.9":
            return "89"
        elif gpu_cc == "9.0":
            return "90"
        else:
            raise RuntimeError(f"Unsupported GPU compute capability: {gpu_cc}")
    except Exception as e:
        raise RuntimeError(f"Failed to detect GPU compute capability: {e}")

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        # don't invoke the original build_ext for this special extension
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            _ = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(ext.name for ext in self.extensions))

        try:
            import torch
        except ImportError:
            sys.stderr.write("Pytorch is required to build this package\n")
            sys.exit(-1)

        self.pytorch_dir = os.path.dirname(torch.__file__)
        self.python_exe = subprocess.check_output(["which", "python"]).decode().strip()

        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext):
        CUDA_COMPUTE_CAPABILITY = get_cuda_compute_capability()
        
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(extdir),
                      "-DCMAKE_PREFIX_PATH={}".format(self.pytorch_dir),
                    #   "-DPYTHON_EXECUTABLE:FILEPATH={}".format(self.python_exe),
                      "-DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0",  # for kenlm - avoid seg fault
                      # "-DPYTHON_EXECUTABLE=".format(sys.executable),
                      f"-DCMAKE_CUDA_ARCHITECTURES={CUDA_COMPUTE_CAPABILITY}"
                      ]

        config = "Debug" if self.debug else "Release"
        build_args = ["--config", config]

        if platform.system() == "Darwin":
            cmake_args += ["-DCMAKE_OSX_DEPLOYMENT_TARGET=10.9"]

        if platform.system() == "Windows":
            cmake_args += ["-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(config.upper(), extdir)]
            if sys.maxsize > 2 ** 32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + config]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        cwd = os.getcwd()
        os.chdir(os.path.dirname(extdir))
        self.spawn(["cmake", ext.sourcedir] + cmake_args)
        if not self.dry_run:
            self.spawn(["cmake", "--build", ".", "--", "-j{}".format(multiprocessing.cpu_count())])
        os.chdir(cwd)


setup(
    name="samoyeds_module",
    version="1.0.0",
    description="Kernel Library for Dual-side Sparse Matrix Multiplication with N:M Structured Sparsity",
    author="Chenpeng Wu, Qiqi Gu",
    author_email="cpwu_sjtu@sjtu.edu.cn, qiqi.gu@sjtu.edu.cn",
    packages=find_packages(),
    license="MIT",
    ext_modules=[CMakeExtension("samoyeds_kernel")],
    cmdclass={"build_ext": CMakeBuild, }
)