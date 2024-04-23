import os
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
        store_dir: str = ".hypha_launcher/containers",
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
            return image_name[len("docker://"):]
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

    def get_command(
        self,
        cmd: str,
        image_name: str,
        cmd_template: T.Optional[str] = None,
        volumes: T.Optional[dict] = None,
        ports: T.Optional[dict] = None,
        gpu: T.Optional[bool] = False,
        envs: T.Optional[dict] = None,
    ) -> str:
        """Get the command for run process in the container

        Args:
            cmd (str): command to run in the container
            image_name (str): image name
            cmd_template (str, optional): command template,
                used to construct the command.
                If not provided, construct the command based on the engine type.
            volumes (dict, optional): volume mapping
                The key is the host path and the value is the container path
            ports (dict, optional): port mapping
                The key is the host port and the value is the container port
            gpu (bool, optional): whether to use GPU
            envs (dict, optional): environment variables
                The key is the environment variable name and the value is the value
        """
        cmd_template = cmd_template or os.environ.get("HYPHA_CONTAINER_CMD_TEMPLATE")
        if cmd_template is not None:
            return cmd_template.format(cmd=cmd, image_name=image_name)
        # Initialize volume and port mappings as empty strings
        volume_mapping = ""
        port_mapping = ""
        env_options = ""
        if volumes is None:
            volumes = {}
        # If volumes are provided, construct volume mapping options
        for host_path, container_path in volumes.items():
            volume_mapping += f"-v {host_path}:{container_path} "
        # If ports are provided, construct port mapping options
        if ports:
            for host_port, container_port in ports.items():
                port_mapping += f"-p {host_port}:{container_port} "
        # If environment variables are provided, construct environment variable options
        if envs:
            for env_name, env_value in envs.items():
                env_options += f"-e {env_name}={env_value} "
        # Construct the command based on the engine type
        if self.engine_type == "docker":
            image_name = self.process_image_name_for_docker(image_name)
            gpu_option = "--gpus all" if gpu else ""
            return f"docker run --rm {gpu_option} {env_options} {volume_mapping} {port_mapping} {image_name} {cmd}"
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
            gpu_option = "--nv" if gpu else ""
            if envs:
                env_options = " ".join([f"--env {k}={v}" for k, v in envs.items()])
            return f"apptainer run --contain {gpu_option} {env_options} {bind_option} {sif_path} {cmd}"
        elif self.engine_type == "podman":
            image_name = self.process_image_name_for_docker(image_name)
            return f"podman run {env_options} {volume_mapping} {port_mapping} {image_name} {cmd}"
        else:
            raise NotImplementedError


if __name__ == "__main__":
    import fire

    fire.Fire(ContainerEngine)
