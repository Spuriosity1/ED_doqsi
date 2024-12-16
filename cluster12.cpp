#include <cmath>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <nlohmann/json_fwd.hpp>
#include <ostream>
#include <stdexcept>
#include <utility>
#include <vector>
#include <xdiag/algebra/algebra.hpp>
#include <xdiag/algebra/matrix.hpp>
#include <xdiag/algorithms/lanczos/eigvals_lanczos.hpp>
#include <xdiag/all.hpp>
#include <armadillo>
#include <xdiag/blocks/spinhalf.hpp>
#include <xdiag/common.hpp>
#include <xdiag/operators/opsum.hpp>
#include <xdiag/utils/say_hello.hpp>
#include <nlohmann/json.hpp>
#include <sstream>
#include "pyrochlore_geometry.hpp"

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
using namespace pyro;


const char* field_proj_label[4] = {"h0", "h1", "h2", "h3"}; 
	// NOTE MINUS SIGN RELATIVE TO PAPER! (actually irrelevant)

static const std::vector<int> spin_sl = {
    // 0  1  2  3  4  5  6  7  8  9  10 11
       1, 3, 2, 1, 3, 2, 0, 0, 0, 2, 3, 1
};

static const std::vector<std::vector<int64_t>> ring_members = {
    {0,5,4,3,2,1},
    {0,6,9,11,8,5},
    {1,2,7,10,9,6},
    {3,4,8,11,10,7}
};

static const std::vector<std::pair<long, long>> bond_list = {
    {0,1}, {1,2}, {2,3}, {3,4}, {4,5}, {5,0}, 
    {0,6}, {1,6}, {6,9},
    {2,7}, {3,7}, {7,10},
    {4,8}, {5,8}, {8,11},
    {9,10}, {10,11}, {11,9}
};


arma::mat evaluate_gs_matrix(const OpSum& O, const std::vector<State>& gs_set){
	arma::mat out(gs_set.size(), gs_set.size());	
	for (int i=0; i<(int)gs_set.size(); i++){
		// the diagonal
		out(i,i) = inner(O, gs_set[i]);
		for (int j=0; j< i; j++){
			auto w = gs_set[j];
			apply(O, gs_set[j], w);
			double val = dot(gs_set[i], w);
			out(i,j) = val;
			out(j,i) = val;
		}
	}
	return out;
}


// returns a 64x64 matrix representing S+S-S+S-S+S-
inline arma::mat ring_flip(){
	// This can be done efficiently
	// This implementation is not that, it's pure horror
	arma::mat O = arma::zeros(64, 64);
    O(0b101010, 0b010101) = 1;
    //O(0b010101, 0b101010) = 1;
	return O;


}


int main(int argc, char** argv) try {
	if (argc < 7){
		cout <<"USAGE: "<<argv[0]<<" <outdir> <Jpm/Jzz> <hx> <hy> <hz> <lanczos_dim> [num_kept_states = 4]\n";
		return 1;
	}

	set_verbosity(1);// set verbosity for monitoring progress
					 //
	int lanczos_dim = atoi(argv[6]);
    std::string out_dir(argv[1]);


	int num_kept_states = 4;
	if (argc >= 8) num_kept_states = atoi(argv[7]);


	OpSum ops;

	// BUILDING THE HAMILTONIAN
	//////////////////////////////////////////////////////////////////////
	///
	// defined types: HB, S+, S-, Sz, EXCHANGE
    //
    for (const auto& pair : bond_list){
        ops += Op("ISING", "Jzz", {pair.first, pair.second});
        ops += Op("EXCHANGE", "Jpm", {pair.first, pair.second});
    }

    for (int J=0; J<12; J++){
        auto mu = spin_sl[J];
        ops += Op("S+", field_proj_label[mu], J);
        ops += Op("S-", field_proj_label[mu], J);
    }
	
	ops["Jzz"] = 1.;
	ops["Jpm"] = atof(argv[2]);
		
	arma::vec3 B = {atof(argv[3]), atof(argv[4]), atof(argv[5])};
	for (int mu=0; mu<4; mu++){
		ops[field_proj_label[mu]] = (double) arma::dot(B,pyro_pos[mu])/sqrt(3);
		cout << "B.e_mu "<<mu<<" "<<ops[field_proj_label[mu]]<<"\n";
	}
	

    std::stringstream label;
    label << "%jpm=" << ops["Jpm"] << "%B=" << B[0]<<","<<B[1]<<","<<B[2];
	/////////////////////////////////////////////
	//// PERFORMING THE DIAGONALISATION (Lanczos)
	///
	auto block = Spinhalf(12);
//	std::vector<string> statev = {"Up","Dn","Up","Dn","Up","Dn","Up","Dn","Up","Dn","Up","Dn","Up","Dn","Up","Dn"};
//	auto init_state = product(block, statev);
	auto lanczos_res = eigs_lanczos(ops, block, 
//			init_state, 
			lanczos_dim,
			/* precision */ 1e-14,
			/* max iter */ 10000,
			/* force complex */ false
			);
	cerr << "iter criterion -> "<< lanczos_res.criterion << endl;


	//////////////////////////////////////////
	//// OUTPUT
	///
	///
	json out;
	out["lattice"] = {};

	out["energies"] = lanczos_res.eigenvalues;

	std::cout<<"Energy values:\n"<<lanczos_res.eigenvalues<<std::endl;



	std::vector<State> gs_set;
	for (int i=0; i<num_kept_states; i++){
		gs_set.push_back(lanczos_res.eigenvectors.col(i));
	}

//	evaluate_exp_Sz(gs_set, out);
//	evaluate_ring_flip(gs_set, out);
//
    out["re_ringflip"] = {};
    for (int mu=0; mu<4; mu++){
        OpSum O({Op("hexa", ring_flip(), ring_members[mu])});
        auto res = evaluate_gs_matrix(O, gs_set);  

        out["re_ringflip"][mu] = res;
        std::cout <<"SL " <<mu <<"ringflip\n" << res;
    }

	out["Jpm"] = atof(argv[2]);
	out["B"] = B;


	// save to file
    //
	//
    //
	std::ofstream file(out_dir + "/out_cluster12_"+label.str()+".json");
    file << out;
} catch (Error& e) {
	error_trace(e);
}
