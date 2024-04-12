import os
import typing as T
from .log import get_logger
from .misc import run_cmd

logger = get_logger()


def detect_hpc_type() -> str:
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
    return "local"


class HPCManger:
    def __init__(self, hpc_type: T.Optional[str] = None):
        if hpc_type is None:
            hpc_type = detect_hpc_type()
        self.hpc_type = hpc_type

    def get_command(self, cmd: str, **attrs) -> str:
        if self.hpc_type == "slurm":
            return self.get_slurm_command(cmd, **attrs)
        elif self.hpc_type == "local":
            return cmd
        else:
            raise NotImplementedError(f"Unsupported HPC type: {self.hpc_type}")

    def get_slurm_command(
            self,
            cmd: str,
            account: T.Optional[str] = None,
            time: T.Optional[str] = None,
            gpus_per_node: T.Optional[str] = None,
            additonal_options: T.Optional[str] = None,
            ) -> str:
        if account is None:
            account = os.environ.get("SLURM_ACCOUNT")
            if account is None:
                raise ValueError("SLURM_ACCOUNT is not set.")
        if time is None:
            time = os.environ.get("SLURM_TIME")
        if gpus_per_node is None:
            gpus_per_node = os.environ.get("SLURM_GPUS_PER_NODE")
        if additonal_options is None:
            additonal_options = os.environ.get("SLURM_ADDITIONAL_OPTIONS")

        options_str = f"--account={account} "
        if time is not None:
            options_str += f"--time={time} "
        if gpus_per_node is not None:
            options_str += f"--gpus-per-node={gpus_per_node} "
        if additonal_options is not None:
            options_str += additonal_options
        new_cmd = f"srun {options_str} {cmd}"
        return new_cmd
