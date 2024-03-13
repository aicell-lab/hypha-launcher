import asyncio
import typing as T
from pathlib import Path

from executor.engine import Engine
from executor.engine.job.extend.subprocess import SubprocessJob
from executor.engine.job.base import Job
from executor.engine.job import ProcessJob
from executor.engine.utils import PortManager
from imjoy_rpc.hypha import connect_to_server
from pyotritonclient import get_config, execute
from cloudpickle import dumps, loads

from .utils.log import get_logger
from .utils.misc import get_all_ips, check_ip_port, run_cmd
from .utils.hpc import HPCManger
from .utils.container import ContainerEngine
from .constants import TRITON_IMAGE

logger = get_logger()


class BridgeWorker:
    def __init__(
            self,
            worker_id: str,
            store_dir: Path = Path(".hypha_launcher_store")
            ):
        self.worker_id = worker_id
        self.store_dir = store_dir

    def run(self, container_engine: ContainerEngine, **kwargs):
        pass

    def register_service(self, server):
        pass


class TritonWorker(BridgeWorker):
    def run(
            self,
            container_engine: ContainerEngine,
            **kwargs):
        container_engine.pull_image(TRITON_IMAGE)
        host_port = PortManager.get_port()
        store_dir = self.store_dir
        model_dir = store_dir / "models"
        if not model_dir.exists():
            model_dir.mkdir(parents=True)
        cmd = container_engine.get_command(
            f'bash -c "tritonserver --model-repository=/models --log-verbose=3 --log-info=1 --log-warning=1 --log-error=1 --model-control-mode=poll --exit-on-error=false --repository-poll-secs=10 --allow-grpc=False --http-port={host_port}"',  # noqa
            TRITON_IMAGE,
            ports={host_port: host_port},
            volumes={str(store_dir / "models"): "/models"},
        )
        self.host_port = host_port
        run_cmd(cmd, check=True)

    async def register_service(self, server):
        async def get_triton_config(model_name: str, verbose: bool = False):  # noqa
            if self.host_port is not None:
                try:
                    res = await get_config(
                        f"http://127.0.0.1:{self.host_port}",
                        model_name=model_name,
                        verbose=verbose,
                    )
                    return res
                except Exception as e:
                    logger.error(f"Error: {e}")
                    return {"error": str(e)}
            else:
                logger.error("Triton server is not started yet.")
                return {"error": "Triton server is not started yet."}

        async def execute_triton(
            inputs: T.Union[T.Any, None] = None,
            model_name: T.Union[str, None] = None,
            cache_config: bool = True,
            **kwargs,
        ):
            if self.host_port is not None:
                try:
                    res = await execute(
                        inputs=inputs,
                        server_url=f"http://127.0.0.1:{self.host_port}",
                        model_name=model_name,
                        cache_config=cache_config,
                        **kwargs,
                    )
                    return res
                except Exception as e:
                    logger.error(f"Error: {e}")
                    return {"error": str(e)}
            else:
                logger.error("Triton server is not started yet.")
                return {"error": "Triton server is not started yet."}

        await server.register_service(
            {
                "name": self.worker_id,
                "id": self.worker_id,
                "config": {"visibility": "public"},
                "get_config": get_triton_config,
                "execute": execute_triton,
            }
        )


class HyphaBridge:
    def __init__(
            self,
            server: T.Optional[T.Union[T.Dict, str]] = None,
            engine: T.Optional[Engine] = None,
            store_dir: str = ".hypha_launcher_store",
            slurm_settings: T.Optional[T.Dict[str, str]] = None,
            debug: bool = False,
            ):
        self.server = server
        self.engine = engine
        self.store_dir = Path(store_dir)
        self.debug = debug

        self.hpc_manager = HPCManger()

        self.slurm_settings = slurm_settings
        self.container_engine = ContainerEngine(
            str(self.store_dir / "containers"))

    @property
    def server_port(self) -> int:
        assert isinstance(self.server, dict)
        public_url = self.server['config']["public_base_url"]
        hypha_port = int(public_url.split(":")[-1])
        return hypha_port

    async def run(
            self,
            worker_types: T.Optional[T.Dict[str, T.Type[BridgeWorker]]] = None,  # noqa
            ):
        if isinstance(self.server, str):
            logger.info(f"Connecting to hypha server: {self.server}")
            self.server = await connect_to_server({"server_url": self.server})
            assert isinstance(self.server, dict)
            logger.info(f"Connected to hypha server: {self.server['config']['public_base_url']}")  # noqa
        assert self.server is not None, "Server is not provided."
        self._record_hypha_server_port(self.server_port)
        self._record_login_node_ips()

        worker_types = worker_types or {
            "triton": TritonWorker,
        }
        self._store_worker_types(worker_types)
        logger.info(f"Worker types: {worker_types}")

        assert self.engine is not None, "Engine is not provided."
        engine = self.engine
        worker_count = 0  # only increase
        workers_jobs: T.Dict[str, Job] = {}
        current_worker_id: T.Union[str, None] = None

        async def launch_worker(
                worker_type: str,
                hpc_type: T.Optional[str] = None,
                worker_id: T.Optional[str] = None,
                ) -> dict:
            nonlocal worker_count
            worker_count += 1
            if worker_id is None:
                worker_id = f"hypha_bridge_worker_{worker_count}"
            cmd = "python -m hypha_launcher.bridge " + \
                  f"--store_dir={self.store_dir.as_posix()} - " + \
                  f"run_worker {worker_type} {worker_id}"
            logger.info(f"Starting worker: {worker_id}")
            logger.info(f"Command: {cmd}")
            if hpc_type is None:
                hpc_type = self.hpc_manager.hpc_type
            logger.info(f"Computational environment(worker type): {hpc_type}")  # noqa
            if hpc_type == "slurm":
                logger.info(f"Slurm settings: {self.slurm_settings}")
                if self.slurm_settings is None:
                    logger.error("Slurm settings is not provided.")
                    raise ValueError("Slurm settings is not provided.")
                assert (
                    "account" in self.slurm_settings
                ), "account is required in slurm settings"  # noqa
                cmd = self.hpc_manager.get_slurm_command(cmd, **self.slurm_settings)  # noqa

            cmd_job = SubprocessJob(cmd, base_class=ProcessJob)
            nonlocal current_worker_id
            current_worker_id = worker_id
            await engine.submit_async(cmd_job)
            await cmd_job.wait_until_status("running")
            workers_jobs[worker_id] = cmd_job
            while True:
                logger.info(f"Waiting for worker: {worker_id}")
                try:
                    service = await self.server.get_service(worker_id)
                    break
                except Exception as e:
                    logger.debug(f"Error: {e}")
                    await asyncio.sleep(2)
            return {
                "worker_id": worker_id,
                "service": service
            }

        async def stop_worker(worker_id: str):
            if worker_id in workers_jobs:
                job = workers_jobs[worker_id]
                await job.cancel()
                del workers_jobs[worker_id]
                nonlocal current_worker_id
                if worker_id == current_worker_id:
                    if len(workers_jobs) > 0:
                        new_worker_id = list(workers_jobs.keys())[0]
                        current_worker_id = new_worker_id
                    else:
                        current_worker_id = None
                return True
            return False

        async def get_all_workers():
            return list(workers_jobs.keys())

        await self.server.register_service(
            {
                "name": "Hypha Bridge",
                "id": "hypha-bridge",
                "launch_worker": launch_worker,
                "stop_worker": stop_worker,
                "get_all_workers": get_all_workers,
            }
        )

    def run_worker(
            self,
            worker_type: str,
            worker_id: str,
            hypha_server_url: T.Optional[str] = None):
        """Run a worker server, run in the compute node of HPC. """
        if hypha_server_url is None:
            hypha_server_url = self._find_connectable_hypha_address()
            if hypha_server_url is None:
                raise ValueError("Cannot connect to hypha server.")

        worker_types = self._load_worker_types()
        worker = worker_types[worker_type](worker_id, self.store_dir)

        async def register_service(server_url: str):
            server = await connect_to_server({"server_url": server_url})
            await worker.register_service(server)

        engine = Engine()
        self.container_engine.pull_image(TRITON_IMAGE)

        async def start_run_worker():
            triton_job = ProcessJob(
                worker.run,
                args=(self.container_engine,)
            )
            await engine.submit_async(triton_job)

        async def main():
            f1 = register_service(hypha_server_url)
            f2 = start_run_worker()
            await asyncio.gather(f1, f2)

        loop = asyncio.get_event_loop()
        loop.create_task(main())
        loop.run_forever()

    def _store_worker_types(self, worker_types: T.Dict[str, T.Type[BridgeWorker]]):  # noqa
        worker_types_file = self.store_dir / "worker_types.pkl"
        with open(worker_types_file, "wb") as f:
            f.write(dumps(worker_types))
        logger.info(f"Worker types are stored to {worker_types_file}")

    def _load_worker_types(self) -> T.Dict[str, T.Type[BridgeWorker]]:
        worker_types_file = self.store_dir / "worker_types.pkl"
        with open(worker_types_file, "rb") as f:
            worker_types = loads(f.read())
        return worker_types

    def _record_login_node_ips(self):
        ips = get_all_ips()
        logger.info("Possible login node IPs:", str(ips))
        ip_record_file = self.store_dir / "login_node_ips.txt"
        with open(ip_record_file, "w") as f:
            for ip in ips:
                f.write(f"{ip[0]}\t{ip[1]}\n")
        logger.info(f"Login node IPs are recorded to {ip_record_file}")
        return ips

    def _record_hypha_server_port(self, port: int):
        with open(self.store_dir / "hypha_server_port.txt", "w") as f:
            f.write(str(port))
        logger.info(f"Hypha server port is recorded to {port}")

    def _get_all_hypha_addresses(self) -> T.List[T.Tuple[str, int]]:
        with open(self.store_dir / "login_node_ips.txt") as f:
            lines = f.readlines()
        ips = [line.strip().split("\t") for line in lines]
        with open(self.store_dir / "hypha_server_port.txt") as f:
            port = int(f.read().strip())
        return [(ip[1], port) for ip in ips]

    def _find_connectable_hypha_address(self) -> T.Union[str, None]:
        hypha_server_url = None
        hypha_addresses = self._get_all_hypha_addresses()
        for ip, port in hypha_addresses:
            logger.info(f"Checking hypha server: {ip}:{port}")
            if check_ip_port(ip, port):
                hypha_server_url = f"http://{ip}:{port}"
                logger.info(f"Found connectable hypha server: {hypha_server_url}")  # noqa
                break
        return hypha_server_url


if __name__ == "__main__":
    import fire
    fire.Fire(HyphaBridge)
