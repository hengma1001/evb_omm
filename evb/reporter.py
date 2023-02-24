import openmm as mm
import openmm.unit as unit

import numpy as np
from numpy import linalg as LA

def dist_pbc(a, b, box):
    """
    calculate distance between two points
    in PBC box
    """
    assert len(a) == len(b)
    box = box[:len(a)]
    a = a % box
    b = b % box
    dist_vec = np.abs(a - b)
    dist_vec = np.abs(dist_vec - box * (dist_vec > box/2))
    return LA.norm(dist_vec)
    # print(dist_vec)

def ddbonds(a, b, c, box): 
    return dist_pbc(a, c, box) - dist_pbc(b, c, box)


class RCReporter(object):
    def __init__(self, file, reportInterval, atom_i, atom_j, atom_k, k, rc0):
        self._out = open(file, 'w')
        self._out.write("rc0,rc,dist_mr, dist_mp\n")
        self._reportInterval = reportInterval
        self.atom_inds = [atom_i, atom_j, atom_k]
        self.k = k
        self.rc0 = rc0

    def __del__(self):
        self._out.close()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, True, False, False, False, None)

    def report(self, simulation, state):
        box = state.getPeriodicBoxVectors().value_in_unit(unit.angstrom)
        box = np.array([box[0][0], box[1][1], box[2][2]])
        positions = np.array(state.getPositions().value_in_unit(unit.angstrom))
        # print(*positions[self.atom_inds], box)
        dist_mr = dist_pbc(positions[self.atom_inds[0]], positions[self.atom_inds[2]], box)
        dist_mp = dist_pbc(positions[self.atom_inds[1]], positions[self.atom_inds[2]], box)
        rc = dist_mr - dist_mp
        self._out.write(f"{self.rc0},{rc},{dist_mr},{dist_mp}\n")

