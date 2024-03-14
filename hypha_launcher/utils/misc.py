import typing as T
import os
import subprocess as subp
import socket

import psutil

from .log import get_logger


logger = get_logger()


def run_cmd(cmd: T.Union[str, T.List[str]], check: bool = True, **kwargs):
    if isinstance(cmd, list):
        logger.info(f"Running command: {' '.join(cmd)}")
        subp.run(cmd, check=check, **kwargs)
    else:
        logger.info(f"Running command: {cmd}")
        subp.run(cmd, shell=True, check=check, **kwargs)


def get_all_ips() -> T.List[T.Tuple[str, str]]:
    ip_info = []
    for interface, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET:  # IPv4
                ip_info.append((interface, addr.address))
    return ip_info


def check_ip_port(ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10)  # 设置超时时间
    result = sock.connect_ex((ip, port))
    sock.close()
    return result == 0


def detect_runtime_environment():
    # Check for Docker and Podman through /proc/self/cgroup
    try:
        with open('/proc/self/cgroup', 'rt') as ifh:
            cgroup_contents = ifh.read()
            if 'docker' in cgroup_contents or '/docker/' in cgroup_contents:
                return 'Docker'
            elif 'podman' in cgroup_contents or '/libpod/' in cgroup_contents:
                return 'Podman'
    except FileNotFoundError:
        pass  # /proc/self/cgroup does not exist, not in a container, or not allowed to read

    # Check for Kubernetes by looking for specific environment variables
    if os.getenv('KUBERNETES_SERVICE_HOST'):
        return 'Kubernetes'

    # Check for Apptainer/Singularity by looking for environment variables set by it
    if os.getenv('SINGULARITY_CONTAINER') or os.getenv('APPTAINER_CONTAINER'):
        return 'Apptainer/Singularity'

    # Check for .dockerenv file
    if os.path.exists('/.dockerenv'):
        return 'Docker'

    # No indicators found
    return 'Unknown'
