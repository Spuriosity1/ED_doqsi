# import numpy as np
# import numpy.linalg as LA
# import scipy.sparse as sp
import json
from sympy import Matrix, Rational, ZZ
from sympy.matrices.normalforms import smith_normal_decomp
from sympy.functions.elementary.integers import floor
from sympy.matrices.dense import diag

#######################################################################
# ATOMIC STORAGE

# The most basic container, represents a single atom


class Atom:
    def __init__(self, xyz, plot_args={}):
        for x in xyz:
            if not (type(x) is int or (
                hasattr(x, 'is_rational') and x.is_rational)
            ):
                # exclude floats because of weird rounding
                raise ValueError(f"Positions must be rational, got '{
                    type(x)}'")
        self.xyz = Matrix([Rational(x) for x in xyz])
        self.plot_args = plot_args

# An atom-class for unitcellspecifier type calls


class Sublattice(Atom):
    def __init__(self, sl_name: str, xyz, plot_args={}):
        super().__init__(xyz, plot_args)
        self.sl_name = sl_name
        self.bonds_from = []

    def __str__(self):
        return (f"Sublattice {self.sl_name} at {self.xyz}")

    def make_shifted_clone(self, plus: Matrix):
        '''
        Returns an Atom of the same type, offfset by 'plus'
        '''
        x = Atom(self.xyz + plus, **self.plot_args)
        return x


class Coupling:
    def __init__(self, name: str, mat):
        self.name = name
        self.mat = mat


class Bond:
    def __init__(self, from_idx: int,
                 to_idx: int,
                 coupling: Coupling,
                 bond_delta
                 ):
        self.from_idx = from_idx
        self.to_idx = to_idx
        self.coupling = coupling
        self.bond_delta = Matrix(bond_delta)


# returns 'Y' such that Y = A r, for some r \\in [0,1)^3
def _wrap_coordinate(A: Matrix, X: Matrix):
    r = (A.inv() @ X) % 1
    return A @ r


def from_cols(*a):
    # convenience
    return Matrix(a).T


class PrimitiveCell:
    # Represents a primitive cell
    def __init__(self, a, lattice_tolerance=1e-6):
        '''
            a -> a matrix of cell vectors formatted as column vectors
            lattice_tolerance -> currently unused
        '''
        self.sublattices = []

        a = Matrix(a)
        assert a.is_square

        self.lattice_vectors = a
        self.lattice_tolerance = lattice_tolerance

    @property
    def num_atoms(self):
        return len(self.sublattices)

    @property
    def atoms(self):
        return self.sublattices

    def distance(self, r1, r2):
        # check the 27 possible cell displacements
        r1 = self.wrap_coordinate(r1)
        r2 = self.wrap_coordinate(r2)
        D = []
        for ix in [-1, 0, 1]:
            for iy in [-1, 0, 1]:
                for iz in [-1, 0, 1]:
                    D.append((self.lattice_vectors @
                             Matrix([ix, iy, iz])+r1-r2).norm())

        return min(D)

    def in_unitcell(self, xyz: Matrix):
        for i in range(3):
            a = self.lattice_vectors[:, i]
            proj = a.dot(xyz)
            if proj < 0 or proj > a.dot(a):
                return False
        return True

    def add_sublattice(self, sl_label: str, xyz: Matrix,
                       wrap_unitcell: bool = True, plot_args={}):
        '''
        @param sl_label        a name for the sublattice
        @param xyz             a [3,1] vector representing the atom position
        '''
        xyz = Matrix(xyz)

        if wrap_unitcell:
            xyz = self.wrap_coordinate(xyz)
        else:
            # check that it is actually within the primitive cell
            # issue a warning if not
            if not self.in_unitcell(xyz):
                print("WARN: adding an atom outside the primtive cell")

        # check that we are not too close to anyone else
        for a in self.sublattices:
            D = self.distance(xyz, a.xyz)
            if D < self.lattice_tolerance*10:
                print(f"WARN: adding an atom very close ({
                    D}) to existing atom {a}")

        self.sublattices.append(Sublattice(sl_label, xyz, plot_args))

        if isinstance(sl_label, Sublattice):
            self.sublattices[-1].bonds_from = sl_label.bonds_from
            # indices. couplings and directions are still OK

        return self.sublattices[-1]  # for further editing

    def as_sl_idx(self, xyz: Matrix):
        sl = None

        ainv = self.lattice_vectors.inv()
        for i, a in enumerate(self.sublattices):
            if all([x.is_integer for x in (ainv @ (a.xyz - xyz))]):
                sl = i
                break

        if sl is None:
            msg = f"Could not add bond: there is no site at {xyz} \n"
            msg += "Possible sites:\n"
            for x in [f"{a.sl_name} {a.xyz} \n" for a in self.sublattices]:
                msg += x + "\n"
            raise Exception(msg)

    # Bonds are always 'attached' to the 'from' index
    def add_bond(self, sl_from: Sublattice, bond_delta: Matrix,
                 coupling: Coupling,
                 fmt={}):
        '''
        @param sl_from     atom linking from
        @param bond_delta  a vector pointing to the next atom
        @param coupling    a reference to the coupling object
        @param fmt         kwargs to pass to ax.plot
        '''
        bond_delta = Matrix(bond_delta)
        to_xyz = self.wrap_coordinate(Matrix(bond_delta) + sl_from.xyz)

        sl_from.bonds_from.append(
            Bond(from_idx=self.as_sl_idx(sl_from.xyz),
                 to_idx=self.as_sl_idx(to_xyz),
                 coupling=coupling,
                 bond_delta=bond_delta)
        )

        sl_from.bonds_from[-1].fmt = fmt

    @property
    def bonds(self):
        bonds = []
        for from_idx, a in enumerate(self.sublattices):
            for b in a.bonds_from:
                bonds.append(dict(
                    from_idx=from_idx,
                    to_idx=b.to_idx,
                    coupling=b.coupling,
                    bond_delta=b.bond_delta,
                    fmt=b.fmt
                ))
        return bonds

    def reduced_pos(self, xyz: Matrix):
        return self.lattice_vectors.inv() @ xyz

    def wrap_coordinate(self, xyz):
        return _wrap_coordinate(self.lattice_vectors, xyz)


def reshape_primitive_cell(cell: PrimitiveCell, bravais: Matrix):
    """
    Constructs a new primitve unit cell based on the previous one, with new
    lattice vectors b1 b2 b3 constructed from A using bravais
    """
    assert all([x.is_integer for x in bravais])
    assert abs(bravais.det()) == 1
    newcell = PrimitiveCell(cell.lattice_vectors * bravais,
                            lattice_tolerance=cell.lattice_tolerance)

    new_sublats = [
        newcell.add_sublattice(
            a.sl_name, a.xyz, wrap_unitcell=True, plot_args=a.plot_args)
        for a in cell.sublattices]

    for j, a in enumerate(cell.sublattices):
        for bf in a.bonds_from:
            newcell.add_bond(
                new_sublats[j], bf.bond_delta, bf.coupling, bf.fmt)

    return newcell

#######################################################
# The Lattice Objects, for storing supercells


class Lattice:
    def __init__(self, primitive_suggestion: PrimitiveCell, bravais_vectors):
        '''
         @param primitive         A primitive unit cell of the lattice. 
                                  This class may choose a different one to 
                                  better mattch the bravais vectors.
         @param bravais_vectors   The Bravais lattice vectors, in units of the primitives,
                                  of the enlarged cell. This should be a 3x3 of integers, 
                                  such that the enlarged vectors may be expressed as 
                                  (a1 a2 a3) * bravais_vectors := (b1 b2 b3)
        '''
        # set up the index scheme, define self.primitive
        self.establish_primitive_cell(
            primitive_suggestion, Matrix(bravais_vectors))
        # Populate the bonds

        self.atoms = []  # format: {xyz, plot_args}
        self.bonds = []  # format: {from_idx, to_idx, coupling}
        self.populate_atoms()
        self.populate_bonds()

    @property
    def lattice_vectors(self):
        return self.primitive.lattice_vectors @ diag(self.periodicity, unpack=True)

    def establish_primitive_cell(self, primitive_suggestion, bravais_vectors: Matrix):
        # Calculate the Smith decomposition of the Bravais vectors , i.e.
        # invertible S, T and diagonal D s.t.
        # S * bravais_vectors * T = D <=> bravais_vectors = S-1 D T-1
        # so can rewrite the periodicity requirement
        # (b1 b2 b3) * diag[z1, z2, z3] ~ 0
        # for all z1,z2,z3 integer
        # <=> (a1 a2 a3) * S-1 J diag(x1 x2 x3) ~ 0 for all x1 x2 x3 integer;
        # suggests a good way to index it in terms of the A1 A2 A3 := (a1 a2 a3) S-1
        # see INDEXING.md for details

        if not all([x.is_rational for x in bravais_vectors]):
            raise TypeError("Bravais vectors must be integers")

        D, S, T = smith_normal_decomp(bravais_vectors, ZZ)
        self.periodicity = [int(x) for x in D.diagonal()]
        self.primitive = reshape_primitive_cell(primitive_suggestion, S.inv())

    def populate_atoms(self):
        for sl in self.primitive.sublattices:
            for iz in range(self.periodicity[2]):
                for iy in range(self.periodicity[1]):
                    for ix in range(self.periodicity[0]):
                        delta = self.primitive.lattice_vectors @ Matrix(
                            [ix, iy, iz])
                        self.atoms.append(
                            Atom(sl.xyz + delta, sl.plot_args)
                        )

    def populate_bonds(self):
        for a in self.primitive.sublattices:
            for iz in range(self.periodicity[2]):
                for iy in range(self.periodicity[1]):
                    for ix in range(self.periodicity[0]):
                        X0 = self.primitive.lattice_vectors @ Matrix(
                            [ix, iy, iz])
                        for b in a.bonds_from:
                            self.bonds.append(dict(
                                from_idx=self.as_linear_idx(X0 + a.xyz),
                                to_idx=self.as_linear_idx(
                                    X0 + a.xyz + b.bond_delta),
                                bond_delta=b.bond_delta,
                                coupling=b.coupling,
                                fmt=b.fmt
                            ))

    def as_idx(self, xyz: Matrix):
        '''
        Takes in the xyz poisition, returns a site index
        Note: after rolling, all xyz must have the form
        (A1 A2 A3)m + dr
        where m \\in Z^3, dr is strictly within the A1 A2 A2 unit cell
        Each SL has an index 
        Strategy: 
            0. roll xyz using b1 b2 b3
            1. decide which primitive unit cell we are in by solving (A1 A2 A3)m = xyz
            2. successively subtract off the primitive-cell positions, decide which is best
        '''
        xyz = self.wrap_coordinate(xyz)
        m = self.primitive.lattice_vectors.inv() @ xyz
        # intra-cell dr
        sl_idx = None
        cell_idx = Matrix([floor(x) for x in m])
        dr = self.primitive.lattice_vectors@(m - cell_idx)
        cell_idx = [int(j) for j in cell_idx]
        for j, sl in enumerate(self.primitive.sublattices):
            if (sl.xyz - dr).norm() == 0:
                sl_idx = j

        if sl_idx is None:
            print(self.lattice_vectors)
            print(f"xyz = {xyz}\nm={m}\ncell_idx={cell_idx}\ndr = {dr}")
            print([[x for x in s.xyz] for s in self.atoms])
            raise Exception(
                f"Position {xyz} does not appear to lie on the lattice")

        for j in range(3):
            assert (cell_idx[j] >= 0 and cell_idx[j] < self.periodicity[j])
        return (cell_idx, sl_idx)

    def as_linear_idx(self, xyz: Matrix):
        cell_idx, sl_idx = self.as_idx(xyz)
        N = self.periodicity
        return cell_idx[0] + N[0]*(cell_idx[1] +
                                   N[1]*(cell_idx[2] + N[2]*sl_idx))

        # ________________
        # Ix 0123012301230123012301230123012301230123012301230123012301230123012301230
        # Iy 0   1   2   0   1   2   0   1   2   0   1   2   0   1   2   0   1   2   0
        # Iz 0           1           0           1           0           1           0
        # mu 0                       1                       2                       3

    @property
    def num_atoms(self):
        n = self.primitive.num_atoms
        for j in range(3):
            n *= self.periodicity[j]
        return n

    def wrap_coordinate(self, xyz):
        return _wrap_coordinate(self.lattice_vectors, xyz)

# SYMMETRY OPERATIONS

    def get_transl_generators(self):
        """
        Returns a list of gnerators for the three obvious translational
        symmetries
        """
        d = self.lattice_vectors.shape[0]  # ndim

        retval = []

        for i in range(d):
            perm = []
            for (orig_idx, a) in enumerate(self.atoms):
                transl_idx = self.as_linear_idx(
                    a.xyz + self.primitive.lattice_vectors[:, i])
                perm.append(transl_idx)

            retval.append(perm)

        return retval

    def get_inversion_perm(self, inversion_centre_xyz):
        """
        Uses a hand-fed inversion centre to generate an inversion permutation
        """
        inv_xyz = Matrix(inversion_centre_xyz)
        perm = []
        for (orig_idx, a) in enumerate(self.atoms):
            dx = a.xyz-inv_xyz
            transl_idx = self.as_linear_idx(inv_xyz - dx)
            perm.append(transl_idx)

        return perm

    def get_refl_perm(self, origin: Matrix, direction: Matrix):
        """
        Returns permutations corresponding to (possibly trivial) reflection
        in the planes normal to 'direction' passing
        through 'origin'
        """

        perm = []

        unit_direction = direction / direction.norm()
        for (orig_idx, a) in enumerate(self.atoms):
            relpos = a.xyz - origin
            # project on to direction
            delta = relpos.dot(unit_direction) * unit_direction
            transl_idx = self.as_linear_idx(
                a.xyz - 2*delta)
            perm.append(transl_idx)

        return perm
