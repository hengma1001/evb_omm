import os
import yaml
import json
import shutil
import parmed as pmd

import openmm as omm
import openmm.app as app
import openmm.unit as u

from .utils import build_logger
from .utils import yml_base
from .utils import create_path
from .reporter import RCReporter
#, get_dir_base

logger = build_logger()

class Simulate(yml_base):
    """
    Run simulation with OpenMM

    Parameters
    ----------
    pdb_file : ``str``
        Coordinates file (.gro, .pdb, ...), This file contains
        the atom positions and possibly PBC (periodic boundary
        condition) box in the system.

    top_file : ``str``
        Topology file contains the interactions in the system

    check_point : ``str``
        Checkpoint file from previous simulations to continue a
        run. Default evaluates to `None`, meaning to run fresh
        simulation

    GPU_id : ``str`` or ``int``
        The device ids of GPU to use for running the simulation.
        Use strings, '0,1' for example, to use more than 1 GPU,

    output_traj : ``str``
        The output trajectory file (.dcd), This is the file
        stores all the coordinates information of the MD
        simulation results. 

    output_log : ``str``
        The output log file (.log). This file stores the MD
        simulation status, such as steps, time, potential energy,
        temperature, speed, etc.

    output_cm : ``str``, optional
        The h5 file contains contact map information. 
        Default is None

    report_time : ``int``
        The frequency to write an output in ps. Default is 10

    sim_time : ``int``
        The length of the simulation trajectory in ns. Default is 10

    dt : ``float``
        The time step of the simulation in fs. Default is 2 

    explicit_sol : ``bool``
        Whether the system contains explicit water model

    temperature : ``float``
        Simulation temperature in K, default is 300

    pressure : ``float``
        Simulation pressure in bar, default is 1

    nonbonded_cutoff: ``float``
        Cutoff distance for nonbonded interactions in nm, default is 
        1
    init_vel : ``bool``
        Initializing velocity, default is False
    """

    def __init__(self,
            pdb_file,
            top_file=None, 
            checkpoint=None,
            gpu_id=0,
            output_traj="output.dcd",
            output_log="output.log", 
            output_rc="output.rc",
            report_time=10,
            log_report=0,
            sim_time=10,
            dt=2.,
            skip_step=0,
            explicit_sol=True,
            temperature=300., 
            pressure=1.,
            nonbonded_cutoff=1.,
            init_vel=False,
            dbonds_umb={},
            morse_bond={},
            forcefield='amber14-all.xml', 
            sol_model='implicit/gbn2.xml',
            local_ssd=None,
            **args) -> None:

        super().__init__()
        # inputs
        self.pdb_file = pdb_file
        self.top_file = top_file
        # self.add_sol = add_sol
        self.checkpoint = checkpoint
        self.gpu_id = str(gpu_id)
        # outputs
        self.output_traj = output_traj
        self.output_log = output_log
        self.output_rc = output_rc
        self.report_time = report_time * u.picoseconds
        self.log_report = log_report * u.picoseconds
        self.sim_time = sim_time * u.nanoseconds
        self.skip_step = skip_step
        # sim setup 
        self.dt = dt * u.femtoseconds
        self.explicit_sol = explicit_sol
        self.temperature = temperature
        self.pressure = pressure
        self.nonbonded_cutoff = nonbonded_cutoff * u.nanometers
        self.init_vel = init_vel

        # 
        self.dbonds_umb = dbonds_umb
        self.morse_bond = morse_bond

        # force field
        self.forcefield = forcefield
        self.sol_model = sol_model
        self.base_dir = os.getcwd()

        # run path 
        self.local_ssd = local_ssd

    def get_setup(self): 
        return {'r0': self.dbonds_umb['rc0'],
                'pdb_file': self.pdb_file, 
                'top_file': self.top_file, 
                'checkpoint': self.checkpoint}

    def build_system(self): 
        system_setup = {
                "nonbondedMethod": app.PME if self.explicit_sol 
                                        else app.CutoffNonPeriodic, 
                # "constraints": app.HBonds, 
                 }
        if self.nonbonded_cutoff:
            system_setup["nonbondedCutoff"] = self.nonbonded_cutoff
        
        if self.top_file: 
            pdb = pmd.load_file(self.top_file, xyz = self.pdb_file)
            if not self.explicit_sol: 
                system_setup['implicitSolvent'] = app.GBn2
            system = pdb.createSystem(**system_setup)
        else: 
            # only supporting implicit runs without topology file 
            # for now
            pdb = pmd.load_file(self.pdb_file)
            forcefield = app.ForceField(
                           self.forcefield,  self.sol_model)
            system = forcefield.createSystem(pdb.topology, **system_setup)

        if self.pressure and self.explicit_sol: 
            system.addForce(omm.MonteCarloBarostat(
                            self.pressure*u.bar, 
                            self.temperature*u.kelvin)
                            )
        ###hm add umbrella potential here 
        if self.dbonds_umb:
            morse_bond = morse_bond_force(**self.morse_bond)
            system.addForce(morse_bond)
            ddbonds_umb = ddbonds_umbforce(**self.dbonds_umb)
            system.addForce(ddbonds_umb)
        
        self.system = system 
        self.top = pdb

    def build_simulation(self): 
        self.build_system() 
        if self.temperature: 
            integrator = omm.LangevinMiddleIntegrator(
                        self.temperature * u.kelvin, 
                        1 / u.picosecond, self.dt)
        else: 
            integrator = omm.VerletIntegrator(self.dt)
        
        try:
            platform = omm.Platform_getPlatformByName("CUDA")
            properties = {'DeviceIndex': str(self.gpu_id), 
                            'CudaPrecision': 'mixed'}
        except Exception:
            platform = omm.Platform_getPlatformByName("OpenCL")
            properties = {'DeviceIndex': str(self.gpu_id)}

        simulation = app.Simulation(
            self.top.topology, self.system, integrator, platform, properties)
        self.simulation = simulation

    def minimizeEnergy(self): 
        self.simulation.context.setPositions(self.top.positions)
        self.simulation.minimizeEnergy()

    def add_reporters(self): 
        report_freq = int(self.report_time / self.dt)
        self.simulation.reporters.append(
                    app.DCDReporter(self.output_traj, report_freq))
        self.simulation.reporters.append(
                app.CheckpointReporter('checkpnt.chk', report_freq))

        if self.log_report:
            report_freq = int(self.log_report / self.dt)
        self.simulation.reporters.append(app.StateDataReporter(
                self.output_log, report_freq, 
                step=True, time=True, speed=True,
                potentialEnergy=True, temperature=True, totalEnergy=True))
        self.simulation.reporters.append(
                RCReporter(self.output_rc, report_freq, **self.dbonds_umb)) 

    def run_sim(self, path='./'): 
        if self.local_ssd: 
            run_path = f"{self.local_ssd}/{os.path.basename(path)}"
        else: 
            run_path = path
        
        if not os.path.exists(run_path): 
            os.makedirs(run_path)

        self.build_simulation() 
        # skip minimization if check point exists
        if self.checkpoint: 
            self.simulation.loadCheckpoint(self.checkpoint)
        else: 
            self.minimizeEnergy()
            self.simulation.step(self.skip_step)
            
        os.chdir(run_path)
        self.add_reporters() 
        # clutchy round up method
        nsteps = int(self.sim_time / self.dt + .5)
        logger.info(f"  Running simulation for {nsteps} steps. ")
        self.simulation.step(nsteps)
        shutil.move(run_path, os.path.dirname(path))
        os.chdir(self.base_dir)

    def md_run(self): 
        """ddmd recursive MD runs"""
        path_label = f'{os.path.basename(self.top_file)[:-4]}_{self.dbonds_umb["rc0"]:.3f}'
        omm_path = create_path(sys_label=path_label)
        logger.info(f"Starting simulation at {omm_path}")
        self.dump_yaml(f"{omm_path}/setting.yml")
        self.run_sim(omm_path)


def ddbonds_umbforce(
        atom_i: int, atom_j: int, atom_k: int, 
        k: float = 0, rc0: float = 0):
    force = omm.CustomCompoundBondForce(3, """
            0.5*k*((r13-r23)-rc0)^2; 
            r13=distance(p1, p3); r23=distance(p2, p3);""")
    force.addGlobalParameter('k', k)
    force.addGlobalParameter('rc0', rc0)
    force.addBond([atom_i, atom_j, atom_k])
    return force


def morse_bond_force(
        atom_i:int, atom_j:int, 
        de:float=0, alpha:float=0, r0:float=0):
    force = omm.CustomBondForce("de*(1 - exp(-alpha*(r-r0)))^2")
    force.addGlobalParameter('de', de)
    force.addGlobalParameter('alpha', alpha)
    force.addGlobalParameter('r0', r0)
    force.addBond(atom_i, atom_j)
    return force


# def exclude_vmd_force(atom_i:int, atom_j:int): 
#     force = omm.CustomBondForce("-4*epsilon*((sigma/r)^12-(sigma/r)^6)")
#     force.addGlobalParameter('epsilon', )
#     force.addGlobalParameter('sigma', )
#     force.addBond(atom_i, atom_j)
