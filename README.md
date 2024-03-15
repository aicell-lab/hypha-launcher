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
  - launch other services inside container
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

### Download model from s3

```bash
$ python -m hypha-launcher - download_models_from_s3 bioengine-model-runner.* --n_parallel=5
```

### Pull docker image of triton server

```bash
$ python -m hypha-launcher - pull_image
```

### Launch a Triton server

Launch a triton worker and register it to a upstream hypha server(https://ai.imjoy.io) as service.

```bash
$ python -m hypha-launcher - launch_triton https://ai.imjoy.io --upstream-mode=True --worker-service-id=my-triton-server
```

Launch triton worker on slurm cluster.

```bash
# Please replace the slurm settings with your own settings
$ python -m hypha_launcher launch_triton https://ai.imjoy.io --upstream-mode=True --slurm-settings='{"account": "Your-Slurm-Account", "time": "03:00:00", "gpus_per_node": "A100:1"}' --worker-service-id=my-hpc-triton-server
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

