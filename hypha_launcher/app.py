import os
import json
import asyncio
import typing as T
from pathlib import Path

from executor.engine import Engine
from executor.engine.job.extend.webapp import WebappJob

from .utils.download import download_files, parse_s3_xml, download_content
from .utils.log import get_logger
from .utils.container import ContainerEngine
from .constants import S3_BASE_URL, TRITON_IMAGE


logger = get_logger()


class App:
    def __init__(self, store_dir: str = ".hypha_launcher", debug: bool = False):
        self.store_dir = Path(store_dir).expanduser().absolute()
        if not self.store_dir.exists():
            self.store_dir.mkdir(parents=True)
        logger.info(f"Store dir: {self.store_dir}")
        self.debug = debug
        self.container_engine = ContainerEngine(self.store_dir / "containers")

    async def download_models_from_s3(
        self,
        pattern: str,
        dest_dir: T.Optional[str] = None,
        n_parallel: int = 5,
        s3_base_url=S3_BASE_URL,
    ):
        """Download models from S3

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

    def pull_image(self, image_name: str = TRITON_IMAGE):
        self.container_engine.pull_image(image_name)

    def run_launcher_server(
        self,
        upstream_hypha_url: str = "https://ai.imjoy.io",
        upstream_service_id: str = "hypha-launcher",
        worker_type: T.Optional[str] = None,
        slurm_settings: T.Optional[T.Dict[str, str]] = None,
        enable_server_apps: bool = True,
    ):
        """Start a launcher server, run in the login node of HPC."""

        launcher_settings = {
            "store_dir": str(self.store_dir),
            "upstream_hypha_url": upstream_hypha_url,
            "upstream_service_id": upstream_service_id,
            "worker_type": worker_type,
            "slurm_settings": slurm_settings,
            "debug": self.debug,
        }
        os.environ["LAUNCHER_SETTINGS"] = json.dumps(launcher_settings)
        engine = Engine()

        async def run_hypha_server():
            """Run a hypha server in the login node."""
            hypha_server_cmd = "python -m hypha.server --host={ip} --port={port}"  # noqa
            if enable_server_apps:
                hypha_server_cmd += " --enable-server-apps"
            hypha_server_cmd += " --startup-function=hypha_launcher.hypha_startup:main"
            hypha_server_job = WebappJob(
                hypha_server_cmd,
                ip="0.0.0.0")
            await engine.submit_async(hypha_server_job)
            while True:
                if hypha_server_job.port:
                    break
                await asyncio.sleep(0.5)
            logger.info(
                "Hypha server started at: " f"http://localhost:{hypha_server_job.port}"
            )

        loop = asyncio.get_event_loop()
        loop.create_task(run_hypha_server())
        loop.run_forever()
