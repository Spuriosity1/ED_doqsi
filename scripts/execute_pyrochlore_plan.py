from pyro_ham_construct import get_symmetries, construct_hamiltonian_spec
import numpy as np
import pyrochlore
import sys
import json
import lattice


from quspin.basis import spin_basis_general  # Hilbert space spin basis
from quspin.operators import quantum_LinearOperator


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


def calc_ringflip(basis, psi, lat):
    ringflip_idx = pyrochlore.get_ringflips(lat)

    # ringflip_spec = hamiltonian([["+-+-+-", [1.] +  ringflip_idx[0] ]],[])
    static_rf = [["+-+-+-", [[1/len(ringflip_idx)]+rf for rf in ringflip_idx]]]

    sum_ringflip_op = quantum_LinearOperator(
        static_rf, basis=basis, check_herm=False, check_symm=False)

    return psi.dot(sum_ringflip_op.dot(psi))


########################################
# BEGIN MAIN PROGRAM
# Simulates the parameter sweep within a particular sector
########################################
if len(sys.argv) < 3:
    print(f"Call sequence: {sys.argv[0]} PLANFILE INDEX")
    sys.exit(1)

assert sys.argv[1].endswith('json')
with open(sys.argv[1], 'r') as planf:
    spec = json.load(planf)

symmetry_evals = spec['symmetry_eigval_list'][int(sys.argv[2])]

full_lat = lattice.Lattice(pyrochlore.primitive, spec['cell'])

tmp = []
for coupling_consts in spec['param_sweep']:
    hamspec = construct_hamiltonian_spec(full_lat, **coupling_consts)
    basis, H = calc_hamiltonian(hamspec, silent=False, **symmetry_evals)

    E, psi_GS = diagonalise(basis, H, full_lat, krylov_dim=min(150, basis.Ns//2))

    rf = calc_ringflip(psi_GS, full_lat)
    tmp.append(dict(
        coupling_consts=coupling_consts,
        energy=E,
        ringflip={
            're': np.real(rf),
            'im': np.imag(rf)
            }
        ))


#  trim .json
fout = "out_"+sys.argv[1][:-5]

for k in symmetry_evals:
    fout += f"?{k}={symmetry_evals[k]}"

fout += ".json"

out_data = {
    "sweep": spec['param_sweep'],
    "sim_results": tmp,
    "version": 1
}

