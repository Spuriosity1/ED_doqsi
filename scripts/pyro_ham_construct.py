import lattice
from sympy import Matrix


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
