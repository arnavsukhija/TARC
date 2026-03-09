#!/usr/bin/env python
from setuptools import setup, find_packages

required = [
    'jax>=0.4.13',
    'brax',
    'chex',
    'wandb',
    'pandas',
    'jaxtyping>=0.2.20',
    'matplotlib>=3.5.1',
    'mujoco',
    'mujoco_mjx',
    'playground',
    'cloudpickle'
]

extras = {}
setup(
    name='tarc',
    version='0.0.1',
    license="MIT",
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=required,
    extras_require=extras,
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
    ],
)
