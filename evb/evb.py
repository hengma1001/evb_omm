import os
import time
import glob
import shutil
import numpy as np
from .task import Run, GPUManager
from .utils import build_logger, create_path
from .utils import dict_from_yaml, dict_to_yaml

logger = build_logger()

class evb_run(object): 
    """
    Set up the ddmd run 

    Parameters
    ----------
    cfg_yml : ``str``
        Main yaml file for setting up runs 
    
    md_only : ``bool``
        Whether to run ml and infer
    """
    def __init__(self, cfg_yml) -> None:
        self.cfg_yml = os.path.abspath(cfg_yml)
        self.yml_dir = os.path.dirname(self.cfg_yml)
        self.evb_setup = dict_from_yaml(self.cfg_yml)
        print(self.evb_setup)
        
        work_dir = self.evb_setup['output_dir']
        cont_run =self.evb_setup['continue'] if 'continue' in self.evb_setup else False
        if os.path.exists(work_dir):
            if cont_run:
                md_previous = glob.glob(f"{work_dir}/md_run/md_run_*")
                md_unfinished = [i for i in md_previous if not os.path.exists(f"{i}/DONE")]
                for md in md_unfinished: 
                    shutil.move(md, f"{os.path.dirname(md)}/_{os.path.basename(md)}")
            else: 
                bkup_dir = work_dir + f'_{int(time.time())}'
                shutil.move(work_dir, bkup_dir)
                logger.info(f"Back up old {work_dir} to {bkup_dir}")
                os.makedirs(work_dir)
        else:
            os.makedirs(work_dir)
        os.chdir(work_dir)

        # logging 
        self.log_dir = 'run_logs'
        os.makedirs(self.log_dir, exist_ok=True)
        
        # manage GPUs
        self.n_sims=self.evb_setup['n_sims']
        n_runs = self.n_sims 
        logger.info(f"Running only {self.n_sims} simulations...")

        self.gpu_ids = GPUManager().request(num_gpus=n_runs)
        logger.info(f"Available {len(self.gpu_ids)} GPUs: {self.gpu_ids}")
        if len(self.gpu_ids) < n_runs:
            n_gpus = len(self.gpu_ids)
            self.n_sims = n_gpus
            logger.info(f"New configuration: {self.n_sims} simulations, ")
            logger.info(f"using {n_gpus} GPUs.")

    def build_tasks(self): 
        evb_cfg = self.evb_setup['evb_cfg']
        print(evb_cfg)
        rc_min, rc_max = evb_cfg['rc_min'], evb_cfg['rc_max']
        rc_inc = evb_cfg['rc_inc']
        rc0_list = np.arange(rc_min, rc_max+rc_inc, rc_inc)
        
        md_setup = self.evb_setup['md_setup']
        # correcting file path
        input_files = ['pdb_file', 'mp_top', 'mr_top']
        for input in input_files: 
            if input in md_setup and md_setup[input]: 
                if not os.path.isabs(md_setup[input]): 
                    md_setup[input] = os.path.join(self.yml_dir, md_setup[input])
                    logger.debug(f"updated entry{input} to {md_setup[input]}.")
        self.md_path = create_path(dir_type='md', time_stamp=False)
        
        md_ymls= []
        for rc0 in rc0_list: 
            for run in ['mr', 'mp']: 
                md_yml = f"{self.md_path}/md_{rc0:.5f}_{run}.yml"
                md_setup_copy = md_setup.copy()
                md_setup_copy['top_file'] = md_setup_copy[f'{run}_top']
                # umb setup
                dbonds_umb = evb_cfg['dbonds_umb'].copy()
                dbonds_umb['rc0'] = float(rc0)
                dbonds_umb['atom_i'] = int(evb_cfg['mr_atom']) - 1
                dbonds_umb['atom_j'] = int(evb_cfg['mp_atom']) - 1
                dbonds_umb['atom_k'] = int(evb_cfg['h_atom']) - 1
                md_setup_copy['dbonds_umb'] = dbonds_umb
                # morse setup
                morse_bond = evb_cfg['morse_bond'].copy()
                morse_bond['atom_i'] = int(evb_cfg[f'{run}_atom']) - 1
                morse_bond['atom_j'] = int(evb_cfg[f'h_atom']) - 1
                md_setup_copy['morse_bond'] = morse_bond
                dict_to_yaml(md_setup_copy, md_yml)
                md_ymls.append(md_yml)

        return md_yml

    def submit_job(self, 
            yml_file, work_path,
            n_gpus=1, job_type='md', 
            type_ind=-1): 
        run_cmd = f"ddmd run_{job_type} -c {yml_file}"
        # setting up output log file
        output_file = f"./{self.log_dir}/{job_type}"
        if type_ind >= 0: 
            output_file = f"{output_file}_{type_ind}"
        # get gpu ids for current job 
        gpu_ids = [self.gpu_ids.pop() for _ in range(n_gpus)]
        run = Run(
            cmd_line=run_cmd,
            gpu_ids=gpu_ids,
            output_file=output_file,
            cwd=work_path, # can be a different working directory
            envs_dict=None, # can be a dictionary of environ vars to add
        )
        return run
        
    def run(self):
        "create and submit ddmd jobs "
        md_ymls = self.build_tasks()
        runs = []
        # md
        for i in range(self.n_sims): 
            md_yml = md_ymls.pop()
            ind = int(os.path.basename(md_yml)[:-4].split('_')[1])
            md_run = self.submit_job(
                    md_yml, self.md_path, n_gpus=1, 
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
                        ind = int(os.path.basename(md_yml)[:-4].split('_')[1])
                        md_run = self.submit_job(
                                md_yml, self.md_path, n_gpus=1, 
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
