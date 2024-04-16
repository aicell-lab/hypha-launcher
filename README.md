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
hypha-launcher --help
```

### Launch the BioEngine Worker on HPC

BioEngine consists of a set of services that are used to serve AI models from bioimage.io. We provide the model test run feature accessible from https://bioimage.io and a dedicated bioengine web client: https://bioimage-io.github.io/bioengine-web-client/. While our public instance is openly accessible for testing and evaluation, you can run your own instance of the BioEngine worker to serve the models, e.g. with your own HPC computing resources.

Download all models from s3 and launch triton server.

Launch on HPC cluster. You need to set the job command template via the `HYPHA_HPC_JOB_TEMPLATE` environment variable for your own HPC cluster.

For example, here is an example for launching the BioEngine on a Slurm cluster:

```bash
# Please replace the job command with your own settings
export HYPHA_HPC_JOB_TEMPLATE="srun -A Your-Slurm-Account -t 03:00:00 --gpus-per-node A100:1 {cmd}"
python -m hypha_launcher launch_bioengine_worker --service-id my-triton
```

In the above example, the job command template is set to use the Slurm scheduler with the specified account and time limit. The `{cmd}` placeholder will be replaced with the actual command to launch jobs.

Optionally, you can also set the store path for storing the models and the triton server configuration via the `HYPHA_LAUNCHER_STORE_DIR` environment variable. By default, the store path is set to `.hypha-launcher`.

```bash
export HYPHA_LAUNCHER_STORE_DIR=".hypha-launcher"
```

### Download model from s3

```bash
python -m hypha-launcher - download_models_from_s3 bioengine-model-runner.* --n_parallel=5
```

### Pull docker image of triton server

```bash
python -m hypha-launcher - pull_image
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

