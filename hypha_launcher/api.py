import os
import json
import asyncio
import typing as T
from pathlib import Path
from functools import partial

from executor.engine import Engine
from executor.engine.job.extend import WebappJob, SubprocessJob
from executor.engine.job import Job, ProcessJob
from executor.engine.utils import PortManager

from .utils.download import download_files, parse_s3_xml, download_content
from .utils.log import get_logger
from .utils.container import ContainerEngine
from .constants import S3_MODELS_URL, S3_CONDA_ENVS_URL, TRITON_IMAGE, S3_IMAGE
from .bridge import HyphaBridge


logger = get_logger()


class HyphaLauncher:
    def __init__(self, store_dir: str = ".hypha_launcher", debug: bool = False):
        self.store_dir = Path(store_dir).expanduser().absolute()
        if not self.store_dir.exists():
            self.store_dir.mkdir(parents=True)
        logger.info(f"Store dir: {self.store_dir}")
        self.debug = debug
        self.container_engine = ContainerEngine(self.store_dir / "containers")
        self.engine = Engine()

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
        s3_base_url=S3_BASE_URL,
    ):
        """Download files from S3

        Args:
            pattern (str): pattern to match model files
            dest_dir (str): destination directory
            s3_base_url (str, optional): base url of S3.
        """
        if dest_dir is None:
            dest_dir = self.store_dir / "models"
        logger.info(
            f"Downloading models from {s3_base_url}"
            f" with pattern {pattern}"
            f" to {dest_dir}"
        )
        xml_content = download_content(s3_base_url)
        items = parse_s3_xml(xml_content, pattern)
        urls = [s3_base_url + item for item in items]
        await download_files(
            urls, dest_dir, n_parallel=n_parallel, base_url=s3_base_url
        )
    
    download_models_from_s3 = partial(download_from_s3, s3_base_url=S3_MODELS_URL)
    download_conda_envs_from_s3 = partial(download_from_s3, s3_base_url=S3_CONDA_ENVS_URL)

    def pull_image(self, image_name: str = TRITON_IMAGE):
        self.container_engine.pull_image(image_name)

    async def launch_s3_server(self):
        """Run a worker server, run in the login node of HPC. """
        engine = self.engine
        self.container_engine.pull_image(S3_IMAGE)

        data_dir = self.store_dir / "s3_data"
        data_dir.mkdir(exist_ok=True, parents=True)

        async def start_s3_server():

            port = PortManager.get_port()
            console_port = PortManager.get_port()
            cmd = self.container_engine.get_command(
                f'server /data --console-address ":{console_port}" --address ":{port}"',  # noqa
                S3_IMAGE,
                ports={
                    port: 9000,
                    console_port: console_port,
                },
                volumes={str(data_dir): "/data"},
            )
            job = SubprocessJob(cmd, base_class=ProcessJob)
            await engine.submit_async(job)
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

    async def launch_server_app(self, server, app_code: str):
        from .constants import IMJOY_APP_TEMPLATE
        controller = await server.get_service("server-apps")
        imjoy_app_code = IMJOY_APP_TEMPLATE.format(app_code=app_code)
        config = await controller.launch(
            source=imjoy_app_code,
            config={"type": "web-python"},
        )
        return config

    async def launch_triton_server(
            self,
            server,
            worker_type: T.Optional[str] = None,
            slurm_settings: T.Optional[T.Dict[str, str]] = None,
            ):
        """Launch a Triton server."""
        bridge = HyphaBridge(
            server=server,
            engine=self.engine,
            store_dir=str(self.store_dir),
            upstream_hypha_url=None,
            upstream_service_id=None,
            worker_type=worker_type,
            slurm_settings=slurm_settings,
            debug=self.debug,
        )
        return await bridge.run()
