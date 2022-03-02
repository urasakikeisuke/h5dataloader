# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='h5dataloader',
    version='1.0.0',
    description='`https://github.com/shikishima-TasakiLab/h5dataloader`',
    long_description='`https://github.com/shikishima-TasakiLab/h5dataloader`',
    author='Junya Shikishima',
    author_email='160442065@ccalumni.meijo-u.ac.jp',
    url='https://github.com/shikishima-TasakiLab/h5dataloader',
    license='',
    packages=find_packages(),
    install_requires=[
        "scikit-build", "cmake", "ninja", "numpy", "h5py", "opencv-python-headless"
    ],
    python_requires='>=3.6'
)
