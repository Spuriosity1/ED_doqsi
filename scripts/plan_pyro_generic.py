import sys
import lattice
import numpy as np
import pyrochlore
import itertools
from pyro_ham_construct import get_symmetries, perm_order
import json

######################################
#  CONSTANTS AND SETUP
######################################

Jpm_sweep = np.linspace(-0.1, 0.2, 20)

# cell = [[1, 0, 0], [0, 2, 0], [0, 0, 2]]
# cell = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
if len(sys.argv) < 4:
    raise Exception(f"Call sequence: {sys.argv[0]} Z1 Z2 Z3, where the supercell vectors are given by e.g. A1 = Z1[1] a1 + Z1[2] a2 + Z1[3] a3")
cell = [[int(xi) for xi in x.split(',')] for x in sys.argv[1:4]]

full_lat = lattice.Lattice(pyrochlore.primitive, cell)

assert full_lat.num_atoms < 33

coupling_consts = dict(
    Jzz=1.0,
    Jpm=None,
    B=[0, 0, 0]
)


lattice_symmetries = get_symmetries(full_lat)
print(lattice_symmetries)
chosen_symmetries = {k: lattice_symmetries[k]
                     for k in ['T1', 'T2', 'T3', 'I', 'P01']}

sectors = {}
for symlabel in chosen_symmetries:
    N = perm_order(lattice_symmetries[symlabel])
    if N > 1:
        sectors[symlabel] = range(N)


sector_list = [x for x in itertools.product(*sectors.values())]

print(sector_list)


symmetry_eigval_list = []
print()
for sector in sector_list:
    print(list(sectors.keys()), ":", sector)
    # construct set of symmetry eigenvalues
    symmetry_evals = {k: v for k, v in zip(sectors.keys(), sector)}
    symmetry_eigval_list.append(symmetry_evals)

# jpm sweep (change later)
param_sweep = [dict(
    Jzz=1.0,
    Jpm=jpm,
    B=[0, 0, 0]
) for jpm in np.linspace(-0.2, 0.2, 20)]

fout = input("enter name of output file (no extension)")


with open(fout+".json", 'w') as f:
    json.dump({'cell': cell,
               'symmetry_eigval_list': symmetry_eigval_list,
               'param_sweep': param_sweep,
               }, f)



