import typing as T
import subprocess as subp
import socket
from .log import get_logger


logger = get_logger()


def run_cmd(cmd: T.Union[str, T.List[str]], check: bool = True):
    if isinstance(cmd, list):
        logger.info(f"Running command: {' '.join(cmd)}")
        subp.run(cmd, check=check)
    else:
        logger.info(f"Running command: {cmd}")
        subp.run(cmd, shell=True, check=check)


def get_ip_address():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        IP = s.getsockname()[0]
    finally:
        s.close()
    return IP
