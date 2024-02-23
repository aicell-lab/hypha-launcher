<div align="center">
<h1> hypha-launcher </h1>

<p> Run triton server on HPC </p>

<p>
  <a href="https://pypi.org/project/hypha-launcher/">
    <img src="https://img.shields.io/pypi/v/hypha-launcher.svg" alt="Install with PyPi" />
  </a>
  <a href="https://github.com/aicell-lab/hypha-launcher/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/aicell-lab/hypha-launcher" alt="MIT license" />
  </a>
</p>
</div>

**Work In Progress**

## Features

+ CLI for downloading model from s3 and pulling docker image of triton server
+ Support different container engines
  - Docker
  - Apptainer
+ Support different compute environments
  - Local
  - Slurm

## Installation

```bash
pip install hypha-launcher
```

## Usage

```bash
$ hypha-launcher
NAME
    hypha-launcher

SYNOPSIS
    hypha-launcher - GROUP | COMMAND | VALUE

GROUPS
    GROUP is one of the following:

     container_engine
       Container engine abstraction. Provides a common interface to container engines, such as docker, apptainer, podman, etc.

COMMANDS
    COMMAND is one of the following:

     download_models_from_s3
       Download models from S3

     pull_image

     run_launcher_server
       Start a launcher server, run in the login node of HPC.

     run_worker
       Run a worker server, run in the compute node of HPC.

VALUES
    VALUE is one of the following:

     debug

     store_dir
```

### Download model from s3

```bash
$ hypha-launcher --store-dir="./triton_store" - download_models_from_s3 bioengine-model-runner.* --n_parallel=5
```

### Pull docker image of triton server

```bash
$ hypha-launcher --store-dir="./triton_store" - pull_image
```

### Start launcher

On local machine:

```bash
$ hypha-launcher --store-dir="./triton_store/" - run_launcher_server 
```

On HPC(Slurm):

```bash
$ hypha-launcher --store-dir="./triton_store/" - run_launcher_server --slurm-settings='{"account": "your-account", "gpus_per_node": "V100:1", "time": "01:00:00"}'
```

## TODO

* [x] Download model from s3
* [x] Pull docker image of triton server
* [x] Run triton server
* [x] Register service on hypha
* [x] Conmmunicate with triton server
* [x] Test on HPC
* [ ] Job management(Auto stop and restart)
* [ ] Load balancing
* [ ] Documentation

## Other information

This package was created with Cookiecutter and the `Nanguage/cookiecutter-pypackage` project template.

+ Cookiecutter: https://github.com/audreyr/cookiecutter
+ `Nanguage/cookiecutter-pypackage`: https://github.com/Nanguage/cookiecutter-pypackage
