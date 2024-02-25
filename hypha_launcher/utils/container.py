import typing as T
from pathlib import Path

from .log import get_logger
from .misc import run_cmd

logger = get_logger()


class ContainerEngine:
    """Container engine abstraction.
    Provides a common interface to container engines,
    such as docker, apptainer, podman, etc.
    """

    supported_engines = ["docker", "apptainer", "podman"]

    def __init__(
        self,
        store_dir: str = "~/.hypha_launcher/containers",
        engine_type: T.Optional[str] = None,
    ):

        self.store_dir = Path(store_dir).expanduser()
        if not self.store_dir.exists():
            self.store_dir.mkdir(parents=True)
        if engine_type is not None:
            self.engine_type = engine_type
        else:
            self.engine_type = self.detect_engine_type()
        self.sif_files: T.Dict[str, Path] = {}

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
                "Cannot find supported container engine: " f"{self.supported_engines}"
            )

    @staticmethod
    def process_image_name_for_docker(image_name: str):
        if image_name.startswith("docker://"):
            return image_name[len("docker://") :]
        return image_name

    @staticmethod
    def process_image_name_for_podman(image_name: str):
        return f"{ContainerEngine.process_image_name_for_docker(image_name)}"

    def pull_image(self, image_name: str):
        logger.info(f"Pulling image {image_name}")
        if self.engine_type == "docker":
            image_name = self.process_image_name_for_docker(image_name)
            run_cmd(["docker", "pull", image_name], check=True)
        elif self.engine_type == "apptainer":
            sif_prefix = image_name.split("//")[1].replace("/", "-").replace(":", "_")
            sif_path = self.store_dir / f"{sif_prefix}.sif"
            if not Path(sif_path).exists():
                logger.info(f"Pull and saving image to {sif_path}")
                run_cmd(["apptainer", "pull", str(sif_path), image_name], check=True)
            else:
                logger.info(f"Image exists: {sif_path}")
            self.sif_files[image_name] = sif_path
        elif self.engine_type == "podman":
            image_name = self.process_image_name_for_podman(image_name)
            run_cmd(["podman", "pull", image_name], check=True)
        else:
            raise NotImplementedError

    def run_command(
        self,
        cmd: str,
        image_name: str,
        volumes: T.Optional[dict] = None,
        ports: T.Optional[dict] = None,
    ):
        """Start container with a command,
        supporting volume and port mapping.

        Args:
            cmd (str): command to run in the container
            image_name (str): image name
            volumes (dict, optional): volume mapping
                The key is the host path and the value is the container path
            ports (dict, optional): port mapping
                The key is the host port and the value is the container port
        """
        # Initialize volume and port mappings as empty strings
        volume_mapping = ""
        port_mapping = ""
        # If volumes are provided, construct volume mapping options
        if volumes:
            for host_path, container_path in volumes.items():
                volume_mapping += f"-v {host_path}:{container_path} "
        # If ports are provided, construct port mapping options
        if ports:
            for host_port, container_port in ports.items():
                port_mapping += f"-p {host_port}:{container_port} "
        # Construct the command based on the engine type
        if self.engine_type == "docker":
            image_name = self.process_image_name_for_docker(image_name)
            run_cmd(
                f"docker run {volume_mapping} {port_mapping} {image_name} {cmd}",
                check=True,
            )  # noqa
        elif self.engine_type == "apptainer":
            # add tmp dir mapping when using apptainer
            host_tmp_dir = self.store_dir / "apptainer_tmp"
            host_tmp_dir.mkdir(exist_ok=True)
            volumes[host_tmp_dir.as_posix()] = "/tmp"
            # Note: Apptainer (formerly Singularity) has different options
            sif_path = self.sif_files[image_name]
            # For Apptainer, bind options are used for volume mapping
            bind_option = ""
            if volumes:
                binds = [f"{host}:{container}" for host, container in volumes.items()]
                bind_option = f"--bind {','.join(binds)} "
            run_cmd(
                f"apptainer run --contain {bind_option} {sif_path} {cmd}", check=True
            )  # noqa
        elif self.engine_type == "podman":
            image_name = self.process_image_name_for_docker(image_name)
            run_cmd(
                f"podman run {volume_mapping} {port_mapping} {image_name} {cmd}",
                check=True,
            )  # noqa
        else:
            raise NotImplementedError


if __name__ == "__main__":
    import fire

    fire.Fire(ContainerEngine)
