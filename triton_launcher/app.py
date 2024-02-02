import typing as T
from pathlib import Path

from executor.engine import Engine
from executor.engine.job.extend.webapp import WebappJob

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

    def run_launcher_server(self, upstream_hypha_server: str):
        """Start a launcher server, run in the login node of HPC. """
        pass
        #login_node_ip = get_ip_address()
        #engine = Engine()
        #hypha_server_cmd = "python -m hypha.server"
        #engine.submit(
        #    WebappJob("python -m hypha.server")
        #)
