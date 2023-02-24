import os
import logging
import subprocess
from typing import List, Dict, Tuple, Set

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)


class InsufficientResources(BaseException): 
    pass

class GPUManager:
    def __init__(self, hostfile=None):
        self.nranks_per_node = int(subprocess.check_output("nvidia-smi -L | wc -l", shell=True))
        self.hosts = self.get_node_list(hostfile=hostfile)
        self.n_gpus = self.nranks_per_node * len(self.hosts)

    @classmethod
    def get_node_list(cls, hostfile=None):
        if hostfile is None:
            hostfile = os.environ["PBS_NODEFILE"]
            print(hostfile)
        with open(hostfile, 'r') as fp:
            data = fp.read()
        splitter = ',' if ',' in data else None
        return [node_id for node_id in data.split(splitter)]

    def request(self, num_gpus:int) -> Set[int]:
        if self.n_gpus > num_gpus:
            return list(range(num_gpus))
        else: 
            return list(range(self.n_gpus))

    def loc_gpu(self, gpu_id): 
        host = self.hosts[ gpu_id // self.nranks_per_node ]
        gpu_id = gpu_id % self.nranks_per_node
        return host, gpu_id


class RunTemplate:
    @staticmethod
    def _env_str(envs):
        envstrs = (f"export {var}='{val}' && " for var, val in envs.items())
        return " ".join(envstrs)

    @staticmethod
    def render(
        command_line: str,
        cwd=None,
        gpu_ids=None,
        envs_dict=None,
        host=None
    ):

        if envs_dict is None:
            envs_dict = {}
        if gpu_ids:
            envs_dict["CUDA_VISIBLE_DEVICES"] = ','.join(str(id) for id in gpu_ids)
        envs = RunTemplate._env_str(envs_dict)
        cmd = f"{envs} {command_line}"
        if cwd:
            cmd =  f"cd {cwd} && {cmd}"
        if host: 
            cmd = f'ssh {host} "{cmd}"'
        return cmd


class Run:

    def __init__(
        self,
        cmd_line: str,
        # num_ranks: int,
        output_file,
        gpu_ids=None,
        cwd=None,
        host=None,
        envs_dict: Dict[str, str] = None,
    ):
        self.gpu_ids = gpu_ids
        self.outfile = open(output_file, 'wb') if isinstance(output_file, str) else output_file

        command = RunTemplate.render(
            command_line=cmd_line,
            cwd=cwd,
            gpu_ids=gpu_ids,
            host=host,
            envs_dict=envs_dict,
        )

        logger.info(f"Popen: {command}")
        self.process = subprocess.Popen(
            command,
            shell=True,
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
    print(gpu_manager.hosts)
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
    gpus = gpu_manager.request(num_gpus=9)
    print(gpus)
    for gpu in gpus: 
        print(gpu_manager.loc_gpu(gpu))
