import asyncio
import typing as T
from pathlib import Path

from executor.engine import Engine
from executor.engine.job.extend.webapp import WebappJob
from executor.engine.job.extend.subprocess import SubprocessJob
from executor.engine.job import ProcessJob
from executor.engine.utils import PortManager
from imjoy_rpc.hypha import connect_to_server
from pyotritonclient import get_config, execute

from .utils.download import download_files, parse_s3_xml, download_content
from .utils.log import get_logger
from .utils.container import ContainerEngine
from .utils.misc import get_ip_address


S3_BASE_URL = "https://uk1s3.embassy.ebi.ac.uk/model-repository/"
TRITON_IMAGE = "docker://nvcr.io/nvidia/tritonserver:23.03-py3"
logger = get_logger()


class App():
    def __init__(
            self,
            store_dir: str = "~/.triton_launcher",
            upstream_hypha_url: str = "https://ai.imjoy.io",
            service_id: str = "triton_launcher",
            debug: bool = False):
        self.store_dir = Path(store_dir).expanduser().absolute()
        self.upstream_hypha_url = upstream_hypha_url
        self.service_id = service_id
        if not self.store_dir.exists():
            self.store_dir.mkdir(parents=True)
        logger.info(f"Store dir: {self.store_dir}")
        logger.info(f"Upstream hypha server: {self.upstream_hypha_url}")
        logger.info(f"Service id: {self.service_id}")
        self.debug = debug

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

    def run_launcher_server(self):
        """Start a launcher server, run in the login node of HPC. """
        login_node_ip = get_ip_address()
        login_hypha_port = None
        engine = Engine()
        worker_count = 0  # only increase
        workers_jobs: T.Dict[str, SubprocessJob] = {}
        current_worker_id: T.Union[int, None] = None

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

        async def main():
            """Start hypha server and link with the upstream hypha server."""
            await run_hypha_server()

            server = await connect_to_server({"server_url": self.upstream_hypha_url}) # noqa
            logger.info(f"Linking to upstream hypha server: {self.upstream_hypha_url}") # noqa

            async def hello(worker_id: str):
                server = await connect_to_server({"server_url": get_login_hypha_url()}) # noqa
                worker = await server.get_service(worker_id)
                return await worker.hello()

            async def start_worker():
                nonlocal worker_count
                worker_count += 1
                worker_id = f"worker_{worker_count}"
                cmd = f"python -m triton_launcher --store_dir={self.store_dir.as_posix()} - run_worker {worker_id} {get_login_hypha_url()}"  # noqa
                logger.info(f"Starting worker: {worker_id}")
                logger.info(f"Command: {cmd}")
                cmd_job = SubprocessJob(
                    cmd,
                    base_class=ProcessJob
                )
                nonlocal current_worker_id
                current_worker_id = worker_id
                await engine.submit_async(cmd_job)
                workers_jobs[worker_id] = cmd_job
                return worker_id

            async def stop_worker(worker_id: str):
                if worker_id in workers_jobs:
                    job = workers_jobs[worker_id]
                    await job.cancel()
                    del workers_jobs[worker_id]
                    nonlocal current_worker_id
                    if worker_id == current_worker_id:
                        if worker_count > 0:
                            new_worker_id = workers_jobs.keys()[0]
                            current_worker_id = new_worker_id
                        else:
                            current_worker_id = None
                    return True
                return False

            async def get_config(
                    model_name: str, verbose: bool = False):
                server = await connect_to_server({"server_url": get_login_hypha_url()})  # noqa
                if current_worker_id is None:
                    logger.error("No worker is running.")
                    return None
                worker = await server.get_service(current_worker_id)
                res = await worker.get_config(model_name, verbose=verbose)
                return res

            async def execute(
                    inputs: T.Union[T.Any, None] = None,
                    model_name: T.Union[str, None] = None,
                    cache_config: bool = True,
                    **kwargs
                    ):
                server = await connect_to_server({"server_url": get_login_hypha_url()})  # noqa
                if current_worker_id is None:
                    logger.error("No worker is running.")
                    return None
                worker = await server.get_service(current_worker_id)
                res = await worker.execute(
                    inputs=inputs,
                    model_name=model_name,
                    cache_config=cache_config,
                    **kwargs
                )
                return res

            service = {
                "name": "triton_launcher",
                "id": self.service_id,
                "config": {
                    "visibility": "public"
                },
                "get_config": get_config,
                "execute": execute,
            }
            if self.debug:
                service["start_worker"] = start_worker
                service["stop_worker"] = stop_worker
                service["hello"] = hello
            await server.register_service(service)
            await start_worker()  # start a default worker

        loop = asyncio.get_event_loop()
        loop.create_task(main())
        loop.run_forever()

    def run_worker(
            self,
            worker_id: str,
            hypha_server_url: str):
        """Run a worker server, run in the compute node of HPC. """
        host_triton_port: T.Union[int, None] = None

        async def start_worker_server(server_url: str):
            server = await connect_to_server({"server_url": server_url})

            def hello():
                print("Hello from worker")
                return "Hello from worker"

            async def get_triton_config(model_name: str, verbose: bool = False):
                if host_triton_port is not None:
                    try:
                        res = await get_config(
                            f"http://127.0.0.1:{host_triton_port}",
                            model_name=model_name,
                            verbose=verbose
                        )
                        return res
                    except Exception as e:
                        logger.error(f"Error: {e}")
                        return None
                else:
                    logger.error("Triton server is not started yet.")
                    return None

            async def execute_triton(
                    inputs: T.Union[T.Any, None] = None,
                    model_name: T.Union[str, None] = None,
                    cache_config: bool = True,
                    **kwargs
                    ):
                if host_triton_port is not None:
                    try:
                        res = await execute(
                            inputs=inputs,
                            server_url=f"http://127.0.0.1:{host_triton_port}",
                            model_name=model_name,
                            cache_config=cache_config,
                            **kwargs
                        )
                        return res
                    except Exception as e:
                        logger.error(f"Error: {e}")
                        return None
                else:
                    logger.error("Triton server is not started yet.")
                    return None

            await server.register_service({
                "name": "worker",
                "id": worker_id,
                "config": {
                    "visibility": "public"
                },
                "hello": hello,
                "get_config": get_triton_config,
                "execute": execute_triton,
            })

        engine = Engine()
        container_engine = ContainerEngine(self.store_dir / "containers")

        async def start_triton_server():
            def run_triton_server(host_port: int):
                container_engine.run_command(
                    f"bash -c \"tritonserver --model-repository=/models --log-verbose=3 --log-info=1 --log-warning=1 --log-error=1 --model-control-mode=poll --exit-on-error=false --repository-poll-secs=10\"",  # noqa
                    TRITON_IMAGE,
                    ports={
                        host_port: 8000
                    },
                    volumes={
                        str(self.store_dir / "models"): "/models"
                    }
                )
            nonlocal host_triton_port
            host_triton_port = PortManager.get_port()
            triton_job = ProcessJob(
                run_triton_server, args=(host_triton_port,)
            )
            await engine.submit_async(triton_job)

        loop = asyncio.get_event_loop()
        loop.create_task(start_worker_server(hypha_server_url))
        loop.create_task(start_triton_server())
        loop.run_forever()
