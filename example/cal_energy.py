
import os
import sys
import yaml
import glob
import tqdm

import openmm as mm
import parmed as pmd
import pandas as pd
import MDAnalysis as mda

# import matplotlib.pyplot as plt
sys.path.append("../")
from evb.sim import Simulate, ddbonds_umbforce, morse_bond_force
from evb.utils import build_logger

# from mpi4py import MPI
# comm = MPI.COMM_WORLD
# size = comm.Get_size()
# rank = comm.Get_rank()

import numpy as np
from numpy import linalg as LA

logger = build_logger()

def dist(a, b):
    """
    calculate distance between two points
    in PBC box
    """
    dist_vec = np.abs(a - b)
    return LA.norm(dist_vec)

def ddbonds(a, b, c): 
    return dist(a, c) - dist(b, c)

def build_energy_set(pdb, top, morse_bond=[1,2]): 
    sim = Simulate(pdb, top_file=top, 
            explicit_sol=False, nonbonded_cutoff=0)
    sim.build_system()

    mbond = morse_bond_force(*morse_bond, 1491.89, 10.46, 0.1)
    sim.system.addForce(mbond)

    # ddbonds_umb = ddbonds_umbforce(6, 7, 8, 251040., -.06)
    # sim.system.addForce(ddbonds_umb)

    context = mm.Context(sim.system, mm.VerletIntegrator(0.001))
    return sim, context

def cal_energy(sim, context, positions): 
    context.setPositions(positions)
    return pmd.openmm.energy_decomposition(sim.top, context)

runs = glob.glob('./test/md_run/md_run_*')

mr_top = './mr_gmx.top'
mp_top = './mp_gmx.top'
pdb = './mr.crd'

mr_sim, mr_context = build_energy_set(pdb, mr_top, [7,8])
mp_sim, mp_context = build_energy_set(pdb, mp_top, [6,8])

evb_df = []
for run in tqdm.tqdm(runs):
    top_label = os.path.basename(run).split('_')[2]
    rc_0 = float(os.path.basename(run).split('_')[-1])
    top_file = f'./{top_label}.top'
    dcd_file = f"{run}/output.dcd"

    mda_u = mda.Universe(top_file, dcd_file)
    for _ in tqdm.tqdm(mda_u.trajectory[:]):
        dbonds_dist = ddbonds(mda_u.atoms[7].position, mda_u.atoms[6].position, mda_u.atoms[8].position)
        local_df = {"rc0": rc_0, "rc": dbonds_dist}
        if top_label == 'mr': 
            local_df['H11'] = cal_energy(mr_sim, mr_context, mda_u.atoms.positions/10)['total']
            local_df['H22'] = 0
        elif top_label == 'mp':
            local_df['H11'] = 0
            local_df['H22'] = cal_energy(mp_sim, mp_context, mda_u.atoms.positions/10)['total']
        evb_df.append(local_df)
    logger.info(f"Done {top_label}_{rc_0}...")
        
# evb_df = comm.gather(evb_df, root=0)
df = pd.DataFrame(evb_df)
df.to_pickle('evb.pkl')


