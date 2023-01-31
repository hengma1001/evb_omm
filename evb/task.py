import os
import signal
import GPUtil
import subprocess
import logging
from typing import List, Dict, Tuple, Set

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)


class InsufficientResources(BaseException): 
    pass

class GPUManager:
    def __init__(self, maxLoad=.2, maxMemory=.2):
        self.maxLoad = maxLoad
        self.maxMemory = maxMemory
        self.gpus = GPUtil.getGPUs()

    def request(self, num_gpus:int) -> Set[int]:
        try: 
            request_gpus = GPUtil.getAvailable(self.gpus, limit=num_gpus,
                    maxLoad=self.maxLoad, maxMemory=self.maxMemory)
        except IndexError: 
            raise InsufficientResources("Not enough resource available for the request. ")
        return request_gpus


class RunTemplate:
    @staticmethod
    def _env_str(envs):
        envstrs = (f'export {var}="{val}" && ' for var, val in envs.items())
        return " ".join(envstrs)

    @staticmethod
    def render(
        command_line: str,
        gpu_ids=None,
        envs_dict=None,
    ):

        if envs_dict is None:
            envs_dict = {}
        if gpu_ids:
            envs_dict["CUDA_VISIBLE_DEVICES"] = ",".join(str(id) for id in gpu_ids)
        envs = RunTemplate._env_str(envs_dict)

        return f"{envs} {command_line}"


class Run:

    def __init__(
        self,
        cmd_line: str,
        # num_ranks: int,
        output_file,
        gpu_ids=None,
        cwd=None,
        envs_dict: Dict[str, str] = None,
    ):
        self.gpu_ids = gpu_ids
        self.outfile = open(output_file, 'wb') if isinstance(output_file, str) else output_file

        command = RunTemplate.render(
            command_line=cmd_line,
            gpu_ids=gpu_ids,
            envs_dict=envs_dict,
        )

        logger.info(f"Popen: {command}")
        self.process = subprocess.Popen(
            command,
            shell=True,
            cwd=cwd,
            stdout=self.outfile,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid
        )

    def poll(self):
        retcode = self.process.poll()
        if retcode is not None:
            self.outfile.close()
        return retcode

    def kill(self): 
        os.killpg(os.getpgid(self.process.pid), 9)



if __name__ == "__main__":
    gpu_manager = GPUManager()
    print(gpu_manager.gpus)
    gpus = gpu_manager.request(num_gpus=1)
    print(gpus)
    gpus = gpu_manager.request(num_gpus=2)
    print(gpus)
    gpus = gpu_manager.request(num_gpus=4)
    print(gpus)
    gpus = gpu_manager.request(num_gpus=4)
    print(gpus)
    gpus = gpu_manager.request(num_gpus=5)
    print(gpus)
    gpus = gpu_manager.request(num_gpus=6)
    print(gpus)
    gpus = gpu_manager.request(num_gpus=7)
    print(gpus)
