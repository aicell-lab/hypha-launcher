import typing as T
from executor.engine.job.extend.subprocess import SubprocessJob

from .log import get_logger
from .misc import run_cmd

logger = get_logger()


def SlurmSubprocess(
        cmd: str,
        account: str,
        time: T.Optional[str] = None,
        gpus_per_node: T.Optional[str] = None,
        additonal_options: T.Optional[str] = None,
        **attrs) -> SubprocessJob:
    options_str = f"--account={account} "
    if time is not None:
        options_str += f"--time={time} "
    if gpus_per_node is not None:
        options_str += f"--gpus-per-node={gpus_per_node} "
    if additonal_options is not None:
        options_str += additonal_options
    new_cmd = f"srun {options_str} {cmd}"
    logger.info(f"Slurm command: {new_cmd}")
    p = SubprocessJob(new_cmd, **attrs)
    return p


def detect_hpc_type() -> T.Optional[str]:
    try:
        run_cmd(["sinfo"], check=True)
        return "slurm"
    except FileNotFoundError:
        pass
    try:
        run_cmd(["qstat"], check=True)
        return "pbs"
    except FileNotFoundError:
        pass
    return None
