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
    def __init__(self, hpc_type: T.Optional[str] = None, hpc_job_command: T.Optional[str] = None):
        if hpc_type is None:
            hpc_type = detect_hpc_type()
        self.hpc_type = hpc_type
        self.hpc_job_command = hpc_job_command or os.environ.get("HYPHA_HPC_JOB_COMMAND")

    def get_command(self, cmd: str, **attrs) -> str:
        if self.hpc_job_command is not None:
            if "{cmd}" in self.hpc_job_command:
                return self.hpc_job_command.format(cmd=cmd)
            return f"{self.hpc_job_command} {cmd}"
        if self.hpc_type == "slurm":
            return self.get_slurm_command(cmd, **attrs)
        elif self.hpc_type == "local":
            return cmd
        else:
            raise NotImplementedError(f"Unsupported HPC type: {self.hpc_type}")

    def get_slurm_command(
            self,
            cmd: str,
            **attrs,
            ) -> str:
        options_str = ""
        for key, value in attrs.items():
            if value is not None:
                options_str += f"--{key}={value} "
        new_cmd = f"srun {options_str} {cmd}"
        return new_cmd
