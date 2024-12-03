#include <cmath>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <utility>
#include <vector>
#include <xdiag/all.hpp>
#include <armadillo>
#include <nlohmann/json.hpp>
#include <sstream>

using namespace xdiag;
using json = nlohmann::json;

std::string getISOCurrentTimestamp()
{
	time_t now;
    time(&now);
    char buf[sizeof "2011-10-08T07:07:09Z"];
    strftime(buf, sizeof buf, "%FT%TZ", gmtime(&now));
	return std::string(buf);
}

// the Paulis
static const arma::mat sigma_x("0 1; 1 0");
static const arma::cx_mat sigma_y(arma::mat("0 0; 0 0"), arma::mat("0 -1; 1 0"));
static const arma::mat sigma_z("1 0; 0 -1");


using namespace std;
using ivec3=arma::ivec3;


const char* field_proj_label[4] = {"h0", "h1", "h2", "h3"}; 


const static vector<ivec3> pyro_pos = {
	{1,1,1},
	{1,-1,-1},
	{-1,1,-1},
	{-1,-1,1}
};

int main(int argc, char** argv) try {
	if (argc < 6){
		cout <<"USAGE: "<<argv[0]<<" <Jpm/Jzz> <hx> <hy> <hz> (n_eigvals = 10)\n";
		return 1;
	}
    int n_eigvals = 10;
    if (argc == 6) n_eigvals = atoi(argv[5]);

	set_verbosity(1);// set verbosity for monitoring progress
					 //


	OpSum ops;

	// BUILDING THE HAMILTONIAN
	//////////////////////////////////////////////////////////////////////
	///
	// defined types: HB, S+, S-, Sz, EXCHANGE
    //

    for (int J1=0; J1<6; J1++){
        ops += Op("ISING", "Jzz", {J1, (J1+1)%6});
        ops += Op("EXCHANGE", "Jpm", {J1, (J1+1)%6});
    }

    std::vector<int> spin_sl = {1,2,3,1,2,3};


    for (int J1=0; J1<6; J1++){
        int mu = spin_sl[J1];

		ops += Op("S+", field_proj_label[mu], J1);
		ops += Op("S-", field_proj_label[mu], J1);
        
    }
	
	ops["Jzz"] = 1.;
	ops["Jpm"] = atof(argv[1])/2;
		
	arma::vec3 B = {atof(argv[2]), atof(argv[3]), atof(argv[4])};
	for (int mu=0; mu<4; mu++){
		ops[field_proj_label[mu]] = (double) arma::dot(B,pyro_pos[mu])/sqrt(3);
		cout << "B.e_mu "<<mu<<" "<<ops[field_proj_label[mu]]<<"\n";
	}
	

    std::stringstream label;
    label << "%jpm=" << ops["Jpm"] << "%B=" << B[0]<<","<<B[1]<<","<<B[2];
	/////////////////////////////////////////////
	//// PERFORMING THE DIAGONALISATION (full)
	///
	auto block = Spinhalf(6);
    auto H = matrixC(ops, block);


    arma::vec eigval;
    arma::cx_mat U;
    arma::eig_sym(eigval,U, H);

    // evaluating ringflip expect value
    arma::cx_mat ringflip(64,64);
    ringflip(0b101010, 0b010101) = 1;
//    ringflip(0b010101, 0b101010) = 1;
    
    std::cout << "Eigvals: \n";
    eigval.rows(0,n_eigvals).print();
    std::cout << "Ringflip in energy basis: \n";
/*
    arma::cx_mat U_reduced = U.cols(0,n_eigvals);

    auto ringflip_energy_basis = U_reduced.t() * ringflip * U_reduced;
    ringflip_energy_basis.print();
*/

    return EXIT_SUCCESS;

} catch (Error e) {
	error_trace(e);
}
