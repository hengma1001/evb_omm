# EVB implementation in OpenMM

## Codes
### Topology building
- Custom Morse bond force
    - Replace the corresponding bond in OMM topology
- Umbrella force 
### Simulation setup 
- Run 2 sets of simulations with mr and mp
### Energy estimation
- md rerun the previous trajectory
    - Make the topology are matching atom-to-atom
    - Load position frame-by-frame
- Energy decomposition in [parmed](https://parmed.github.io/ParmEd/html/openmm.html)
- Energy report in openmm

## Input & Output
### Input
- yml setup file 
    - force constants and $RC_0$ for `ddbonds` umbrella sampling, and three atoms
    - 
- mr_top, mp_top
    1. generate ligand pdbs
    2. generate protein poses, align with pymol 
    3. parameterize pdbs 
    4. merge pdbs
- initial coordinates

### Output format
- A list of energies of $RC$, $H_{11}$, $H_{22}$
    - into more info, $E_g$, $E_{bias}$