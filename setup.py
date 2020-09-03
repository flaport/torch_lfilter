""" Bring lowpass filtering to PyTorch! """

import torch
import torch_lfilter
from setuptools import setup, Extension
from torch.utils import cpp_extension

try:
    with open("readme.md", "r") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = torch_lfilter.__doc__

torch_lfilter_cpp = Extension(
    name="torch_lfilter_cpp",
    sources=["torch_lfilter.cpp"],
    include_dirs=cpp_extension.include_paths(),
    library_dirs=cpp_extension.library_paths(),
    extra_compile_args=[],
    libraries=[
        "c10",
        "torch",
        "torch_cpu",
        "torch_python",
    ],
    language="c++",
)

setup(
    name="torch_lfilter",
    version=torch_lfilter.__version__,
    author=torch_lfilter.__author__,
    author_email="floris.laporte@gmail.com",
    description=torch_lfilter.__doc__.splitlines()[0],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/flaport/torch_lfilter",
    py_modules=["torch_lfilter"],
    ext_modules=[torch_lfilter_cpp],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
