import asyncio
import typing as T
from pathlib import Path

from executor.engine import Engine
from executor.engine.job.extend.webapp import WebappJob
from imjoy_rpc.hypha import connect_to_server

from .utils.download import download_files, parse_s3_xml, download_content
from .utils.log import get_logger
from .utils.container import ContainerEngine
from .utils.misc import get_ip_address


S3_BASE_URL = "https://uk1s3.embassy.ebi.ac.uk/model-repository/"
TRITON_IMAGE = "docker://nvcr.io/nvidia/tritonserver:23.03-py3"
logger = get_logger()


class App():
    def __init__(self, store_dir: str = "~/.triton_launcher"):
        self.store_dir = Path(store_dir).expanduser()
        if not self.store_dir.exists():
            self.store_dir.mkdir(parents=True)
        logger.info(f"Store dir: {self.store_dir}")

    async def download_models_from_s3(
            self, pattern: str,
            dest_dir: T.Optional[str] = None,
            n_parallel: int = 5,
            s3_base_url=S3_BASE_URL):
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
            urls, dest_dir, n_parallel=n_parallel,
            base_url=s3_base_url)

    def pull_image(self, image_name: str = TRITON_IMAGE):
        contanier_engine = ContainerEngine(self.store_dir / "containers")
        contanier_engine.pull_image(image_name)

    def run_launcher_server(
            self, upstream_hypha_server: str = "https://ai.imjoy.io"):
        """Start a launcher server, run in the login node of HPC. """
        login_node_ip = get_ip_address()
        engine = Engine()

        async def run_hypha_server():
            """Run a hypha server in the login node."""
            hypha_server_cmd = "python -m hypha.server --host={ip} --port={port}" # noqa
            hypha_server_job = WebappJob(hypha_server_cmd, ip="0.0.0.0")
            await engine.submit_async(hypha_server_job)
            logger.info(
                "Hypha server started at: "
                f"{login_node_ip}:{hypha_server_job.port}")

        async def link_to_upstream():
            """Link to the upstream hypha server."""
            server = await connect_to_server({"server_url": upstream_hypha_server}) # noqa
            await server.register_service({
                "name": "triton_launcher",
                "id": "launcher",
                "config": {
                    "visibility": "public"
                },
                "run_worker": self.run_worker,
            })

        loop = asyncio.get_event_loop()
        loop.create_task(run_hypha_server())
        loop.create_task(link_to_upstream())
        loop.run_forever()

    def run_worker(
            self, hypha_server_url: str):
        async def start_worker_server(server_url: str):
            server = await connect_to_server({"server_url": server_url})

            def hello():
                print("Hello from worker")

            await server.register_service({
                "name": "hello",
                "id": "hello",
                "config": {
                    "visibility": "public"
                },
                "hello": hello,
            })

        loop = asyncio.get_event_loop()
        loop.create_task(start_worker_server(hypha_server_url))
        loop.run_forever()
