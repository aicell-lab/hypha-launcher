import typing as T
from pathlib import Path

from .log import get_logger
from .misc import run_cmd

logger = get_logger()


class ContainerEngine():
    """Container engine abstraction.
    Provides a common interface to container engines,
    such as docker, apptainer, podman, etc.
    """
    supported_engines = ["docker", "apptainer"]

    def __init__(
            self,
            store_dir: T.Optional[str] = "~/.triton_launcher/containers",
            engine_type: T.Optional[str] = None
            ):
        self.store_dir = Path(store_dir).expanduser()
        if not self.store_dir.exists():
            self.store_dir.mkdir(parents=True)
        if engine_type is not None:
            self.engine_type = engine_type
        else:
            self.engine_type = self.detect_engine_type()
        self.sif_files = {}

    def detect_engine_type(self):
        for engine in self.supported_engines:
            try:
                run_cmd([engine, "--version"], check=True)
                logger.info(f"Found supported container engine: {engine}")
                return engine
            except FileNotFoundError:
                pass
        else:
            raise RuntimeError(
                "Cannot find supported container engine: "
                f"{self.supported_engines}")

    def pull_image(self, image_name: str):
        logger.info(f"Pulling image {image_name}")
        if self.engine_type == "docker":
            if image_name.startswith("docker://"):
                image_name = image_name[len("docker://"):]
            run_cmd(["docker", "pull", image_name], check=True)
        elif self.engine_type == "apptainer":
            sif_prefix = image_name.split("//")[1] \
                .replace("/", "-").replace(":", "_")
            sif_path = self.store_dir / f"{sif_prefix}.sif"
            if not Path(sif_path).exists():
                logger.info(f"Pull and saving image to {sif_path}")
                run_cmd(
                    ["apptainer", "pull", image_name, sif_path], check=True)
            else:
                logger.info(f"Image exists: {sif_path}")
            self.sif_files[image_name] = sif_path
        else:
            raise NotImplementedError


if __name__ == "__main__":
    import fire
    fire.Fire(ContainerEngine)
