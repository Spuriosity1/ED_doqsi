#include <cmath>
#include <complex>
#include <vector>
#include <xdiag/algorithms/lanczos/eigvals_lanczos.hpp>
#include <xdiag/all.hpp>
#include <armadillo>

using namespace xdiag;
using namespace std;

int main(int argc, char** argv) try {
	if (argc < 3){
		std::cout<<"USAGE: "<<argv[0]<<" <L> <hx> <Jzz>=1\n";
		return 1;
	}
	int L = atoi(argv[1]);
	const int n_states = 2;


	// the Paulis
	//
	const arma::mat sigma_x("0 1; 1 0");
	const arma::cx_mat sigma_y(arma::mat("0 0; 0 0"), arma::mat("0 -1; 1 0"));
	const arma::mat sigma_z("1 0; 0 -1");


	OpSum ops;
	for (int i = 0; i < L; ++i) {
		ops += Op("ISING", "Jzz", {i, (i + 1) % L});
		
		ops += Op("Sx", "hx", i);
	}
	ops["hx"] = atof(argv[2])* Coupling(sigma_x);

	ops["Jzz"] = 4*(
			(argc >= 4) ? atof(argv[3]) : 1
			);

	set_verbosity(2);// set verbosity for monitoring progress
					 //
	
	auto block = Spinhalf(L);
	auto lanczos_res = eigs_lanczos(ops, block, n_states,
			/* precision */ 1e-13,
			/* max iter */ 1000,
			/* force complex */ true
			);
	cerr << "iter criterion -> "<< lanczos_res.criterion << endl;

	auto gs = lanczos_res.eigenvectors.col(0);
	gs.make_complex();

	std::cout<<"Energy values:\n"<<lanczos_res.eigenvalues<<std::endl;

	std::cout<<"SZ expectation values:\n";
	std::cout<<"J\t<Sx>\t<Sy>\t<Sz>\n"; 

	for (int J=0; J<L; J++){
		arma::cx_double Sx = innerC(Op("__x",sigma_x, J), gs);
		arma::cx_double Sy = innerC(Op("__y",sigma_y, J), gs);
		arma::cx_double Sz = innerC(Op("__z",sigma_z, J), gs);
		printf("%d\t%16.7e\t%16.7e\t%16.7e\n", 
				J, Sx.real(), Sy.real(), Sz.real());
	}
	

} catch (Error& e) {
	error_trace(e);
}
