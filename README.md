<div align="center">
<h1> triton-launcher </h1>

<p> Run triton server on HPC </p>

<p>
  <a href="https://pypi.org/project/triton_launcher/">
    <img src="https://img.shields.io/pypi/v/triton_launcher.svg" alt="Install with PyPi" />
  </a>
  <a href="https://github.com/aicell-lab/triton-launcher/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/aicell-lab/triton-launcher" alt="MIT license" />
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

## TODO

* [x] Download model from s3
* [x] Pull docker image of triton server
* [x] Run triton server
* [x] Register service on hypha
* [x] Conmmunicate with triton server
* [ ] Test on HPC
* [ ] Documentation

## Other information

This package was created with Cookiecutter and the `Nanguage/cookiecutter-pypackage` project template.

+ Cookiecutter: https://github.com/audreyr/cookiecutter
+ `Nanguage/cookiecutter-pypackage`: https://github.com/Nanguage/cookiecutter-pypackage
