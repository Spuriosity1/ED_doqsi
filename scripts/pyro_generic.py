from quspin.basis import spin_basis_general  # Hilbert space spin basis
import numpy as np  # generic math functions
from quspin.operators import quantum_LinearOperator, hamiltonian
from quspin.tools import lanczos
import pyrochlore
import lattice
from sympy import Matrix
import itertools
import json
import sys

def is_identity(p: list):
    return all(x == j for (j, x) in enumerate(p))


def perm_order(p: list):
    order = 1
    tmp = [x for x in p]
    while not is_identity(tmp):
        tmp = [p[x] for x in tmp]
        order += 1
        if order > len(p):
            raise Exception("perm is broken")
    return order


def construct_hamiltonian_spec(lat: lattice.Lattice, Jzz, Jpm, B):
    Ising_opspec = []
    xc_opspec = []

    for bond in lat.bonds:
        Ising_opspec.append([Jzz, bond['from_idx'], bond['to_idx']])
        xc_opspec.append([Jpm, bond['from_idx'], bond['to_idx']])
    # gx_opspec = [[g, i] for i in range(L)]

    return dict(
        static=[["zz", Ising_opspec], ["+-", xc_opspec], ["-+", xc_opspec]],
        lat=lat
    )


def get_symmetries(lat, strip_trivial=False):
    T1, T2, T3 = lat.get_transl_generators()
    I = lat.get_inversion_perm([0, 0, 0])

    syms = {
        "T1": T1,
        "T2": T2,
        "T3": T3,
        "I": I,
        'P01': lat.get_refl_perm(origin=Matrix([1, 1, 1]),
                                 direction=Matrix([0, 1, 1])),
        'P02': lat.get_refl_perm(origin=Matrix([1, 1, 1]),
                                 direction=Matrix([1, 0, 1])),
        'P03': lat.get_refl_perm(origin=Matrix([1, 1, 1]),
                                 direction=Matrix([1, 1, 0])),
        'P12': lat.get_refl_perm(origin=Matrix([1, 1, 1]),
                                 direction=Matrix([0, 1, -1])),
        'P23': lat.get_refl_perm(origin=Matrix([1, 1, 1]),
                                 direction=Matrix([0, 1, -1])),
        'P31': lat.get_refl_perm(origin=Matrix([1, 1, 1]),
                                      direction=Matrix([0, 1, -1])),
    }

    if not strip_trivial:
        return syms

    nontriv_syms = {}
    for s in syms:
        if not is_identity(syms[s]):
            nontriv_syms[s] = syms[s]
    return nontriv_syms


def calc_hamiltonian(hamspec, silent=False, **kwargs):
    """
    Calculates the actual sparse hamiltonian
    @param hamspec -> output of construct_hamiltonian_spec
    kwargs: pass in the form n_T_1 = n
    """
    syms = get_symmetries(hamspec["lat"])

    args_to_spin_basis = {}
    for k in kwargs:
        if type(kwargs[k]) is not int:
            raise "kwargs specify rep, must start with 'k_' and be integer"
        if k in syms:
            args_to_spin_basis[k+"_block"] = (syms[k], kwargs[k])

    basis = spin_basis_general(
        hamspec["lat"].num_atoms,
        **args_to_spin_basis
    )
    if not silent:
        print("basis dim: ", basis.Ns)

    ham_kwargs = {}
    if silent:
        ham_kwargs = dict(check_herm=False, check_symm=False)

    H = quantum_LinearOperator(hamspec["static"], basis=basis,
            dtype=np.complex128,
            **ham_kwargs)

    return basis, H


def diagonalise(basis, H, lat, krylov_dim=50):
    v0 = np.random.normal(0, 1, size=basis.Ns)

    # m_GS = 150  # Krylov subspace dimension
    #
    E, V = H.eigsh(k=krylov_dim, which="SA")
    # compute ground state vector
    return E[0], V[:, 0]


def calc_ringflip(psi_GS_lanczos, lat):
    ringflip_idx = pyrochlore.get_ringflips(lat)

    # ringflip_spec = hamiltonian([["+-+-+-", [1.] +  ringflip_idx[0] ]],[])
    static_rf = [["+-+-+-", [[1/len(ringflip_idx)]+rf for rf in ringflip_idx]]]

    sum_ringflip_op = quantum_LinearOperator(
        static_rf, basis=basis, check_herm=False, check_symm=False)

    return psi_GS_lanczos.dot(sum_ringflip_op.dot(psi_GS_lanczos))


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


# SYMMETRY AWARE DIAGONALISATION

# Diagonalise in each sector
sectors = {}


for symlabel in chosen_symmetries:
    N = perm_order(lattice_symmetries[symlabel])
    if N > 1:
        sectors[symlabel] = range(N)

ringflip_vals = []

sector_list = [x for x in itertools.product(*sectors.values())]

print(sector_list)

chosen_sector_list = sector_list
# update if needed

print()
for sector in chosen_sector_list:
    print(list(sectors.keys()), ":", sector)
    tmp = []
    for jpm in Jpm_sweep:
        print(f"\rSector {sector} Jpm={jpm}")
        coupling_consts["Jpm"] = jpm

        hamspec = construct_hamiltonian_spec(full_lat, **coupling_consts)

        symmetry_evals = {k: v for k, v in zip(sectors.keys(), sector)}
        basis, H = calc_hamiltonian(hamspec, silent=False, **symmetry_evals)

        E, psi_GS = diagonalise(basis, H, full_lat,
                                krylov_dim=min(150, basis.Ns//2))

        rf = calc_ringflip(psi_GS, full_lat)

        tmp.append(dict(
            Jpm=jpm,
            energy=E,
            ringflip={
                're': np.real(rf),
                'im': np.imag(rf)
                }
        ))

    ringflip_vals.append({
        'sector': {k: v for k, v in zip(sectors.keys(), sector)},
        'ring': tmp
        })

out_data = {
    "Jpm_sweep": list(Jpm_sweep),
    "sim_results": ringflip_vals,
    "version": 1
}
fout = f"out_pyro_{';'.join(sys.argv[1:4])}.json"
with open(fout, 'w') as f:
    json.dump(out_data, f)

    print("Dumped to ", fout)
