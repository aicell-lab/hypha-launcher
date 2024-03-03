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
    def __init__(self):
        self.hpc_type = detect_hpc_type()
        self.logger = get_logger()

    def get_command(self, cmd: str, **attrs) -> str:
        if self.hpc_type == "slurm":
            return self.get_slurm_command(cmd, **attrs)
        elif self.hpc_type == "local":
            return cmd
        else:
            raise NotImplementedError(f"Unsupported HPC type: {self.hpc_type}")

    def get_slurm_command(
            cmd: str,
            account: str,
            time: T.Optional[str] = None,
            gpus_per_node: T.Optional[str] = None,
            additonal_options: T.Optional[str] = None,
            ) -> str:
        options_str = f"--account={account} "
        if time is not None:
            options_str += f"--time={time} "
        if gpus_per_node is not None:
            options_str += f"--gpus-per-node={gpus_per_node} "
        if additonal_options is not None:
            options_str += additonal_options
        new_cmd = f"srun {options_str} {cmd}"
        return new_cmd
