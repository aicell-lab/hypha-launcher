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

+ CLI/API for:
  - downloading model from s3 and pulling docker image of triton server
  - launch s3 server
  - launch triton server
  - ...
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

## CLI Usage

```bash
$ hypha-launcher --help
```

### Launch the bioimage.io backend

Download all models from s3 and launch triton server.

```bash
$ python -m hypha_launcher launch_bioimageio_backend --service-id my-triton
```

Launch on slurm cluster.

```bash
# Please replace the slurm settings with your own settings
$ export SLURM_ACCOUNT=Your-Slurm-Account
$ export SLURM_TIME=03:00:00
$ export SLURM_GPUS_PER_NODE=A100:1
$ python -m hypha_launcher launch_bioimageio_backend --service-id my-triton
```

### Download model from s3

```bash
$ python -m hypha-launcher - download_models_from_s3 bioengine-model-runner.* --n_parallel=5
```

### Pull docker image of triton server

```bash
$ python -m hypha-launcher - pull_image
```

## TODO

* [x] Download model from s3
* [x] Pull docker image of triton server
* [x] Run triton server
* [x] Register service on hypha
* [x] Conmmunicate with triton server
* [x] Test on HPC
* [ ] Support run on local machine without GPU
* [ ] Support launch containers inside a container (For support run inside the podman-desktop)
* [ ] Job management(Auto stop and restart)
* [ ] Load balancing
* [ ] Documentation


## Development
Install the package in editable mode with the following command:

```bash
pip install -e .
pip install -r requirements-dev.txt
```

