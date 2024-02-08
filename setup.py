#!/usr/bin/env python

"""The setup script."""

import re
from setuptools import setup, find_packages


def get_long_description():
    return "See https://github.com/aicell-lab/triton_launcher"


def get_version():
    with open("triton_launcher/__init__.py") as f:
        for line in f.readlines():
            m = re.match("__version__ = '([^']+)'", line)
            if m:
                return m.group(1)
        raise IOError("Version information can not found.")


def get_install_requirements():
    requirements = [
        "aiohttp", "requests", "tqdm", "fire",
        "loguru", "executor-engine", "hypha",
        "uvicorn", "imjoy-rpc", "pyotritonclient"
    ]
    return requirements


requires_test = ['pytest', 'pytest-cov', 'flake8', 'mypy']
packages_for_dev = ["pip", "setuptools", "wheel", "twine", "ipdb"]

requires_dev = packages_for_dev + requires_test
requires_doc = []
with open("docs/requirements.txt") as f:
    for line in f:
        p = line.strip()
        if p:
            requires_doc.append(p)

requires_dev += requires_doc

setup(
    author="Weize Xu, Wei Ouyang",
    author_email='vet.xwz@gmail.com',
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    description="Run triton server on HPC",
    install_requires=get_install_requirements(),
    license="MIT license",
    long_description=get_long_description(),
    include_package_data=True,
    keywords='triton_launcher',
    name='triton_launcher',
    packages=find_packages(include=['triton_launcher', 'triton_launcher.*']),
    url='https://github.com/Nanguage/triton_launcher',
    version=get_version(),
    zip_safe=False,
    extras_require={
        'test': requires_test,
        'doc': requires_doc,
        'dev': requires_dev,
    },
    entry_points={
        'console_scripts': [
            'triton_launcher=triton_launcher.__main__:main',
        ],
    },
)
