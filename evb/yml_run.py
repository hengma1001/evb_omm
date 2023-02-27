import os
import time
from .task import Run, GPUManager
from .utils import build_logger 

logger = build_logger()

class yml_run(object): 
    """
    Set up the ddmd run 

    Parameters
    ----------
    cfg_yml : ``str``
        Main yaml file for setting up runs 
    
    md_only : ``bool``
        Whether to run ml and infer
    """
    def __init__(self, ymls:list) -> None:
        self.ymls = ymls

    def submit_job(self, 
            yml_file, work_path,
            n_gpus=1, job_type='md', 
            type_ind=-1): 

        run_cmd = f"{self.evb_exe} run_{job_type} -c {yml_file}"
        # setting up output log file
        label = os.path.basename(yml_file)[:-4]
        output_file = f"./{work_path}/../run_logs/{label}"
        # get gpu ids for current job 
        gpu_ids = [self.gpu_ids.pop() for _ in range(n_gpus)]
        host = self.gpu_host.hosts[gpu_ids[0] // self.gpu_host.nranks_per_node]
        gpu_ids = [i % self.gpu_host.nranks_per_node for i in gpu_ids]
        
        run = Run(
            cmd_line=run_cmd,
            gpu_ids=gpu_ids,
            output_file=output_file,
            cwd=work_path, # can be a different working directory
            host=host,
            envs_dict=None, # can be a dictionary of environ vars to add
        )
        return run
        
    def run(self):
        "create and submit md jobs "
        # manage GPUs
        n_runs = len(self.ymls)
        logger.info(f"Running {n_runs} simulations...")

        self.gpu_host = GPUManager()
        self.gpu_ids = self.gpu_host.request(num_gpus=n_runs)
        logger.info(f"Available {len(self.gpu_ids)} GPUs: {self.gpu_ids}")
        if len(self.gpu_ids) < n_runs:
            n_gpus = len(self.gpu_ids)
            n_runs = n_gpus
            logger.info(f"New configuration: {n_runs} simulations, ")
            logger.info(f"using {n_gpus} GPUs.")
        
        md_ymls = self.ymls
        runs = []
        # md
        for i in range(self.n_sims): 
            md_yml = md_ymls.pop()
            ind = len(md_ymls)
            md_run = self.submit_job(
                    md_yml, os.path.dirname(md_yml), n_gpus=1, 
                    job_type='md', type_ind=ind)
            runs.append(md_run)
        
        try:
            # loop through all the yamls
            while md_ymls != []:
                runs_done = [run for run in runs if run.poll() is not None]
                if len(runs_done) > 0: 
                    for run in runs_done:
                        runs.remove(run)
                        self.gpu_ids.extend(run.gpu_ids)
                        logger.info(f"Finished run on gpu {run.gpu_ids}")

                        i += 1
                        md_yml = md_ymls.pop()
                        ind = len(md_ymls)
                        md_run = self.submit_job(
                                md_yml, os.path.dirname(md_yml), n_gpus=1, 
                                job_type='md', type_ind=ind)
                        runs.append(md_run)
            # waiting for runs to finish
            while runs:
                # Clean up runs which finished (poll() returns process code)
                runnings = [run for run in runs if run.poll() is None]
                if len(runs) != len(runnings):
                    logger.info(f"waiting on {len(runnings)} runs to finish...")
                    runs = runnings 
                time.sleep(5)
            logger.info("All done!")
        except KeyboardInterrupt: 
            for p in runs: 
                p.kill()
            logger.info("cleaned up!")
