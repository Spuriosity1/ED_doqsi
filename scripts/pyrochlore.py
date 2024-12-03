import lattice
# import visual
# import matplotlib.pyplot as plt
import numpy as np
from sympy import Matrix

primitive = lattice.PrimitiveCell([[0, 4, 4],
                                   [4, 0, 4],
                                   [4, 4, 0]])

disp = [
    Matrix(v) for v in [
        [0, 0, 0],
        [0, 2, 2],
        [2, 0, 2],
        [2, 2, 0]
    ]]


plaqt = [
    [Matrix(x) for x in [
        [0, -2, 2],
        [2, -2, 0],
        [2, 0, -2],
        [0, 2, -2],
        [-2, 2, 0],
        [-2, 0, 2]]],
    [Matrix(x) for x in [
        [0, 2, -2],
        [2, 2, 0],
        [2, 0, 2],
        [0, -2, 2],
        [-2, -2, 0],
        [-2, 0, -2]]],
    [Matrix(x) for x in [
        [0, -2, -2],
        [-2, -2, 0],
        [-2, 0, 2],
        [0, 2, 2],
        [2, 2, 0],
        [2, 0, -2]]],
    [Matrix(x) for x in [
        [0, 2, 2],
        [-2, 2, 0],
        [-2, 0, -2],
        [0, -2, -2],
        [2, -2, 0],
        [2, 0, 2]]]
]

plaq_locs = [Matrix(x) for x in
             [[4, 4, 4], [4, 2, 2], [2, 4, 2], [2, 2, 4]]]


def get_ringflips(lat: lattice.Lattice):
    retval = []
    for ix in range(lat.periodicity[0]):
        for iy in range(lat.periodicity[1]):
            for iz in range(lat.periodicity[2]):
                dx = lat.primitive.lattice_vectors @ Matrix([ix, iy, iz])
                for mu, plaq_sl_pos in enumerate(plaq_locs):
                    plaq_pos = lat.wrap_coordinate(plaq_sl_pos + dx)

                    retval.append([lat.as_linear_idx(plaq_pos + x)
                                  for x in plaqt[mu]])

    return retval


sublat = [primitive.add_sublattice(str(j), disp[j]) for j in range(4)]

# six bond sublattices, store them like su(4) generators as lower triang mat
c = [[lattice.Coupling("j01", np.zeros((3, 3), dtype=np.complex128))
      for j in range(i)]
     for i in range(4)]

for j, color in zip([1, 2, 3], ['r', 'g', 'b']):
    primitive.add_bond(sublat[0], disp[j], c[j][0], fmt={'color': color})
    primitive.add_bond(sublat[j], disp[j],  c[j][0], fmt={'color': color})

for (i, j), color in zip([(1, 2), (2, 3), (3, 1)], ['r', 'g', 'b']):
    delta = disp[j] - disp[i]
    primitive.add_bond(sublat[i], delta,  c[j][0], fmt={'color': color})
    primitive.add_bond(sublat[i], -delta,  c[j][0], fmt={'color': color})
