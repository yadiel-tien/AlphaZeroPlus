from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension
import numpy as np


ext_modules = [
    Extension(
        name="chess_cython",          # 生成的 .so 文件名
        sources=["chess_cython.pyx"], # 你的 Cython 源文件
        include_dirs=[np.get_include()]
    )
]

setup(
    name="chess_cython",
    ext_modules=cythonize(ext_modules, annotate=True),
)