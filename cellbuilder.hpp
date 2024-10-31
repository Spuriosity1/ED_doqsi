#pragma once
#include <armadillo>
#include <cassert>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "rationalmath.hpp"
struct abstract_site {
	arma::ivec3 pos;
	std::vector<std::pair<arma::ivec3, int>> coboundary;
};


struct cell_specifier {
	arma::imat33 primitive_vectors;
	std::vector<abstract_site> sites;
};

struct bondspec {
	int i;
	int j;
	std::string bond_type;
};

struct periodic_cell {
	std::vector<arma::ivec3> spin_pos;
	std::vector<bondspec> bond_specs;

	periodic_cell(const arma::imat33& supercell, const cell_specifier& primitive);

	arma::ivec3 wrap(const arma::ivec3& x) const;
};


arma::ivec3 periodic_cell::wrap(const arma::ivec3& x) const {
	// Maps x back within the cell

}
