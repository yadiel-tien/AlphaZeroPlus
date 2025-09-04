from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension
import numpy as np


ext_modules = [
    Extension(
        name="chess_cython",          # 生成的 .so 文件名
        sources=["chess_cython.pyx"], # 你的 Cython 源文件
        include_dirs=[np.get_include()],
        # 这会告诉编译器，这个模块是为无GIL环境准备的
        define_macros=[("Py_GIL_DISABLED", "1")]
    )
]

setup(
    name="chess_cython",
    ext_modules=cythonize(ext_modules, annotate=True),
)