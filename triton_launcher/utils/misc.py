import typing as T
import subprocess as subp
from .log import get_logger


logger = get_logger()


def run_cmd(cmd: T.List[str], check: bool = True):
    logger.info(f"Running command: {' '.join(cmd)}")
    subp.run(cmd, check=check)


def get_ip_address():
    cmd = ["hostname", "-I"]
    result = subp.run(cmd, check=True, stdout=subp.PIPE)
    return result.stdout.decode("utf-8").strip().split()[0]
