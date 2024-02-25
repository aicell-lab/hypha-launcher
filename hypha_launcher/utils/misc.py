import typing as T
import subprocess as subp
import socket

import netifaces as ni

from .log import get_logger


logger = get_logger()


def run_cmd(cmd: T.Union[str, T.List[str]], check: bool = True):
    if isinstance(cmd, list):
        logger.info(f"Running command: {' '.join(cmd)}")
        subp.run(cmd, check=check)
    else:
        logger.info(f"Running command: {cmd}")
        subp.run(cmd, shell=True, check=check)


def get_all_ips(
        ip_type: T.Literal["ipv4", "ipv6"] = "ipv4"
        ) -> T.List[T.Tuple[str, str]]:
    ip_info = []
    for interface in ni.interfaces():
        if interface == "lo":
            continue
        addresses = ni.ifaddresses(interface)
        if ip_type == "ipv4":
            ip = addresses.get(ni.AF_INET, [{}])[0].get('addr')
        else:
            ip = addresses.get(ni.AF_INET6, [{}])[0].get('addr')
        if ip:
            ip_info.append((interface, ip))
    return ip_info


def check_ip_port(ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10)  # 设置超时时间
    result = sock.connect_ex((ip, port))
    sock.close()
    return result == 0
