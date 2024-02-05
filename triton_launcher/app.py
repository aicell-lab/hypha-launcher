import asyncio
import typing as T
from pathlib import Path

from executor.engine import Engine
from executor.engine.job.extend.webapp import WebappJob
from executor.engine.job.extend.subprocess import SubprocessJob
from executor.engine.job import ProcessJob
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
        login_hypha_port = None
        engine = Engine()
        worker_count = 0
        workers_jobs: T.Dict[str, SubprocessJob] = {}

        def get_login_hypha_url():
            return f"http://{login_node_ip}:{login_hypha_port}"

        async def run_hypha_server():
            """Run a hypha server in the login node."""
            hypha_server_cmd = "python -m hypha.server --host={ip} --port={port}" # noqa
            hypha_server_job = WebappJob(hypha_server_cmd, ip="0.0.0.0")
            await engine.submit_async(hypha_server_job)
            while True:
                if hypha_server_job.port:
                    break
                await asyncio.sleep(0.5)
            logger.info(
                "Hypha server started at: "
                f"{login_node_ip}:{hypha_server_job.port}")
            nonlocal login_hypha_port
            login_hypha_port = hypha_server_job.port

        async def link_to_upstream():
            """Link to the upstream hypha server."""
            server = await connect_to_server({"server_url": upstream_hypha_server}) # noqa
            logger.info(f"Linking to upstream hypha server: {upstream_hypha_server}") # noqa

            async def hello(worker_id: str):
                server = await connect_to_server({"server_url": get_login_hypha_url()}) # noqa
                worker = await server.get_service(worker_id)
                return await worker.hello()

            async def start_worker():
                nonlocal worker_count
                worker_count += 1
                worker_id = f"worker_{worker_count}"
                cmd_job = SubprocessJob(
                    f"python -m triton_launcher run_worker {worker_id} {get_login_hypha_url()}",  # noqa
                    base_class=ProcessJob
                )
                await engine.submit_async(cmd_job)
                workers_jobs[worker_id] = cmd_job
                return worker_id

            async def stop_worker(worker_id: str):
                if worker_id in workers_jobs:
                    job = workers_jobs[worker_id]
                    await job.cancel()
                    del workers_jobs[worker_id]
                    return True
                return False

            await server.register_service({
                "name": "triton_launcher",
                "id": "triton_launcher",
                "config": {
                    "visibility": "public"
                },
                "hello": hello,
                "start_worker": start_worker,
                "stop_worker": stop_worker,
            })

        loop = asyncio.get_event_loop()
        loop.create_task(run_hypha_server())
        loop.create_task(link_to_upstream())
        loop.run_forever()

    def run_worker(
            self,
            worker_id: str,
            hypha_server_url: str):
        async def start_worker_server(server_url: str):
            server = await connect_to_server({"server_url": server_url})

            def hello():
                print("Hello from worker")
                return "Hello from worker"

            await server.register_service({
                "name": "worker",
                "id": worker_id,
                "config": {
                    "visibility": "public"
                },
                "hello": hello,
            })

        loop = asyncio.get_event_loop()
        loop.create_task(start_worker_server(hypha_server_url))
        loop.run_forever()
