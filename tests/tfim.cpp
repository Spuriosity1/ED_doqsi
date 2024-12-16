#include <cmath>
#include <complex>
#include <vector>
#include <xdiag/algorithms/lanczos/eigvals_lanczos.hpp>
#include <xdiag/all.hpp>
#include <armadillo>

using namespace xdiag;


inline PermutationGroup construct_symmetry_group(int64_t N){
	// Translational symmetries
	std::vector<int64_t> id_perm;
	std::vector<int64_t> rot_perm;

	for (int64_t i =0; i<N; i++){
		id_perm.push_back(i);
		rot_perm.push_back( (i+1) % N);
	}


	Permutation rot(rot_perm);

	std::vector<Permutation> perm_ops;


	Permutation current_perm(id_perm);
	for (int64_t i =0; i<N; i++){
		perm_ops.push_back(current_perm);
		current_perm = current_perm * rot;
	}
	return PermutationGroup(perm_ops);
}


int main(int argc, char** argv) try {
	if (argc < 1){
		std::cout<<"USAGE: "<<argv[0]<<" <N>\n";
		return 1;
	}
	int N = atoi(argv[1]);

	auto permgroup = construct_symmetry_group(N);


	// the Paulis
	//
	const arma::mat sigma_x("0 1; 1 0");
	const arma::cx_mat sigma_y(arma::mat("0 0; 0 0"), arma::mat("0 -1; 1 0"));
	const arma::mat sigma_z("1 0; 0 -1");


	OpSum ops;
	for (int i = 0; i < N; ++i) {
		ops += Op("ISING", "Jzz", {i, (i + 1) % N});
		
		ops += Op("Sx", "hx", i);
	}
	ops["Jzz"] = 1.0;
	ops["hx"] = 0.2 * Coupling(sigma_x);


	// Figure out the representations
	std::map<double, Representation> irreps;
	for (int i=0; i<N; i++) {
		double k = 2*M_PI*i/N;

		std::vector<complex> characters;
		for (int j=0; j<N; j++) {
			characters.push_back(exp(complex(0,1)*j*k));
		}

		auto rep = Representation(characters);
		irreps[k] = rep;
	}
	


	set_verbosity(2);// set verbosity for monitoring progress
					 //
	std::vector<arma::vec> eigvals;

	for (const auto& [k, irrep] : irreps){
		auto block = Spinhalf(N, permgroup, irrep);
		auto lanczos_res = eigvals_lanczos(ops, block, 4); // compute ground state energy
		std::cerr << "k = " << k << ", iter criterion -> " <<  lanczos_res.criterion << std::endl;
		eigvals.push_back(lanczos_res.eigenvalues);
	}

} catch (Error& e) {
	error_trace(e);
}
