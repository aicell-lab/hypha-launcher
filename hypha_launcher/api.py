import os
import typing as T
from pathlib import Path
from functools import partial
import secrets

from executor.engine import Engine
from executor.engine.job.extend import SubprocessJob
from executor.engine.job import Job, ProcessJob
from executor.engine.utils import PortManager
from imjoy_rpc.hypha import connect_to_server

from .utils.download import download_files, parse_s3_xml, download_content
from .utils.log import get_logger
from .utils.container import ContainerEngine
from .utils.hpc import HPCManger
from .constants import S3_MODELS_URL, S3_CONDA_ENVS_URL, TRITON_IMAGE, S3_IMAGE
from .bridge import HyphaBridge


logger = get_logger()


class HyphaLauncher:
    def __init__(
            self,
            store_dir: str = ".hypha_launcher",
            debug: bool = False,
            container_engine_kwargs: T.Optional[T.Dict[str, T.Any]] = None,
            hpc_manager_kwargs: T.Optional[T.Dict[str, T.Any]] = None,
            executor_engine_kwargs: T.Optional[T.Dict[str, T.Any]] = None,
            ):
        self.store_dir = Path(store_dir).expanduser().absolute()
        if not self.store_dir.exists():
            self.store_dir.mkdir(parents=True)
        logger.info(f"Store dir: {self.store_dir}")
        self.debug = debug
        if container_engine_kwargs is None:
            container_engine_kwargs = {
                "store_dir": str(self.store_dir / "containers"),
            }
        self.container_engine = ContainerEngine(**container_engine_kwargs)
        if hpc_manager_kwargs is None:
            hpc_manager_kwargs = {}
        self.hpc_manager = HPCManger(**hpc_manager_kwargs)
        if executor_engine_kwargs is None:
            executor_engine_kwargs = {}
        self.engine = Engine(**executor_engine_kwargs)

    def get_free_port(self) -> int:
        return PortManager.get_port()

    def get_jobs_ids(self) -> T.List[str]:
        return [job.id for job in self.engine.jobs]

    async def stop_job(self, job_id: str) -> bool:
        job = self.engine.jobs.get_job_by_id(job_id)
        if job is not None:
            await job.cancel()
            return True
        return False

    async def download_from_s3(
        self,
        pattern: str,
        dest_dir: T.Optional[str] = None,
        n_parallel: int = 5,
        s3_base_url=S3_MODELS_URL,
    ):
        """Download files from S3

        Args:
            pattern (str): pattern to match model files
            dest_dir (str): destination directory
            s3_base_url (str, optional): base url of S3.
        """
        if dest_dir is None:
            dest_dir = str(self.store_dir / "models")
        logger.info(
            f"Downloading models from {s3_base_url}"
            f" with pattern {pattern}"
            f" to {dest_dir}"
        )
        xml_content = download_content(s3_base_url)
        items = parse_s3_xml(xml_content, pattern)
        urls = [s3_base_url + item for item in items if not item.endswith("/")]
        await download_files(
            urls, dest_dir, n_parallel=n_parallel, base_url=s3_base_url
        )

    async def download_models_from_s3(
        self,
        pattern: str,
        dest_dir: T.Optional[str] = None,
        n_parallel: int = 5,
        s3_base_url=S3_MODELS_URL,
    ):
        """Download models from S3"""
        await self.download_from_s3(
            pattern, dest_dir, n_parallel, s3_base_url
        )

    async def download_conda_envs_from_s3(
        self,
        pattern: str,
        dest_dir: T.Optional[str] = None,
        n_parallel: int = 5,
        s3_base_url=S3_CONDA_ENVS_URL,
    ):
        """Download conda envs from S3"""
        await self.download_from_s3(
            pattern, dest_dir, n_parallel, s3_base_url
        )

    def pull_image(self, image_name: str = TRITON_IMAGE):
        self.container_engine.pull_image(image_name)

    async def launch_s3_server(
            self,
            minio_root_user: T.Optional[str] = None,
            minio_root_password: T.Optional[str] = None):
        """Run a worker server, run in the login node of HPC. """
        engine = self.engine
        self.container_engine.pull_image(S3_IMAGE)

        data_dir = self.store_dir / "s3_data"
        data_dir.mkdir(exist_ok=True, parents=True)
        if minio_root_user is None:
            minio_root_user = "minio"
        if minio_root_password is None:
            minio_root_password = secrets.token_urlsafe(16)

        async def start_s3_server():

            port = PortManager.get_port()
            console_port = PortManager.get_port()
            cmd = self.container_engine.get_command(
                f'server /data --console-address ":{port}" --address ":{console_port}"',  # noqa
                S3_IMAGE,
                ports={
                    port: port,
                    console_port: console_port,
                },
                volumes={data_dir.as_posix(): "/data"},
            )
            env = os.environ.copy()
            env["MINIO_ROOT_USER"] = minio_root_user
            env["MINIO_ROOT_PASSWORD"] = minio_root_password
            job = SubprocessJob(
                cmd, base_class=ProcessJob,
                popen_kwargs={"env": env},
            )
            await engine.submit_async(job)
            await job.wait_until_status("running")
            return job, port, console_port

        job, port, console_port = await start_s3_server()
        return {
            "job_id": job.id,
            "console_port": port,
            "port": console_port,
            "stop": job.cancel
        }

    async def launch_job(self, job: Job):
        """Launch an executor job."""
        await self.engine.submit_async(job)
        return {
            "job_id": job.id,
            "stop": job.cancel
        }

    async def launch_command(
            self,
            cmd: str,
            image_name: T.Optional[str] = None,
            container_kwargs: T.Optional[T.Dict[str, str]] = None,
            hpc_kwargs: T.Optional[T.Dict[str, str]] = None,
            cmd_kwargs: T.Optional[T.Dict[str, str]] = None,
            ):
        if image_name is not None:
            self.container_engine.pull_image(image_name)
            container_kwargs = container_kwargs or {}
            cmd = self.container_engine.get_command(cmd, image_name, **container_kwargs)
        cmd = self.hpc_manager.get_command(cmd, **(hpc_kwargs or {}))
        job = SubprocessJob(cmd, base_class=ProcessJob, **(cmd_kwargs or {}))
        await self.engine.submit_async(job)
        return {
            "job_id": job.id,
            "stop": job.cancel
        }

    async def launch_server_app(self, server, app_code: str):
        from .constants import IMJOY_APP_TEMPLATE
        controller = await server.get_service("server-apps")
        imjoy_app_code = IMJOY_APP_TEMPLATE.format(app_code=app_code)
        config = await controller.launch(
            source=imjoy_app_code,
            config={"type": "web-python"},
        )
        return config

    async def launch_bridge(
            self,
            server,
            service_id: str = "hypha-bridge",
            worker_types: T.Optional[T.Dict] = None,
            slurm_settings: T.Optional[T.Dict[str, str]] = None,
            upstream_mode: bool = False,
        ):
        """Launch a bridge."""
        bridge = HyphaBridge(
            server=server,
            service_id=service_id,
            engine=self.engine,
            store_dir=str(self.store_dir),
            slurm_settings=slurm_settings,
            debug=self.debug,
            upstream_mode=upstream_mode,
        )
        await bridge.run(worker_types=worker_types)

    async def _get_server(self, server):
        if isinstance(server, str):
            server = await connect_to_server({"server_url": server})
        return server

    async def launch_bridge_worker(
            self,
            server,
            worker_type: str,
            bridge_service_id: str = "hypha-bridge",
            hpc_type: T.Optional[str] = None,
            worker_service_id: T.Optional[str] = None,
        ):
        """Launch a bridge worker."""
        server = await self._get_server(server)
        bridge = await server.get_service(bridge_service_id)
        worker_dict = await bridge.launch_worker(worker_type, hpc_type, worker_service_id)
        worker_dict['stop'] = partial(bridge.stop_worker, worker_dict['worker_id'])
        return worker_dict

    async def launch_triton(
            self,
            server,
            bridge_service_id: str = "hypha-bridge",
            hpc_type: T.Optional[str] = None,
            worker_service_id: T.Optional[str] = None,
            **kwargs,
        ):
        """Launch a Triton worker."""
        server = await self._get_server(server)
        try:
            await server.get_service(bridge_service_id)
        except Exception as e:
            logger.warning(f"Cannot find bridge service: {bridge_service_id}")
            await self.launch_bridge(server, bridge_service_id, **kwargs)
        return await self.launch_bridge_worker(
            server, "triton", bridge_service_id, hpc_type,
            worker_service_id,
        )

    async def launch_hello_world(self):
        """Detect in which environment, docker/k8s/apptainer"""
        # detect env
        # print env name
        # launch a ubuntu, echo hello world from {ENV_NAME}
        pass


def create_service(config):
    launcher = HyphaLauncher(config)
    return {
        "service_id": "hypha-launcher",
        "pull_image": launcher.pull_image,
        "download_models_from_s3": launcher.download_models_from_s3,
        "download_conda_envs_from_s3": launcher.download_conda_envs_from_s3,
        "launch_s3_server": launcher.launch_s3_server,
        "launch_triton": launcher.launch_triton,
        "launch_command": launcher.launch_command,
        "stop_job": launcher.stop_job,
        "get_jobs_ids": launcher.get_jobs_ids,
    }
