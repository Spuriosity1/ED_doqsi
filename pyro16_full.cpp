#include <cmath>
#include <cstdint>
#include <iostream>
#include <nlohmann/json_fwd.hpp>
#include <ostream>
#include <stdexcept>
#include <utility>
#include <vector>
#include <xdiag/algebra/matrix.hpp>
#include <xdiag/all.hpp>
#include <armadillo>
#include <nlohmann/json.hpp>
#include <sstream>
#include <xdiag/operators/symmetrize.hpp>
#include <xdiag/symmetries/generated_group.hpp>
#include "pyrochlore_geometry.hpp"

using namespace xdiag;
using json = nlohmann::json;
using namespace pyro;

std::string getISOCurrentTimestamp()
{
	time_t now;
    time(&now);
    char buf[sizeof "2011-10-08T07:07:09Z"];
    strftime(buf, sizeof buf, "%FT%TZ", gmtime(&now));
	return std::string(buf);
}

using namespace std;
using ivec3=arma::ivec3;

ivec3 wrap(const ivec3& x){
	ivec3 y(x);
	y[0] = (y[0]%8 + 8)%8;
	y[1] = (y[1]%8 + 8)%8;
	y[2] = (y[2]%8 + 8)%8;
	return y;
}

int spin_idx(const ivec3& R_){
	auto R = wrap(R_);
	for (int i=0; i<pyro16_sites.size(); i++){
		if (arma::all(R == pyro16_sites[i])) {return i;}
	}
	throw logic_error("Indexed illegal site");
}


const char* field_proj_label[4] = {"h0", "h1", "h2", "h3"}; 
	// NOTE MINUS SIGN RELATIVE TO PAPER! (actually irrelevant)

const ivec3 B_diamond={2,2,2};

void add_tetras(OpSum& ops, int sl){
	assert(sl*sl == 1);

	for (const auto& R_fcc : FCC_pos){
		for (int mu=0; mu<4; mu++){
			auto R1 = R_fcc + (1-sl)/2 * B_diamond + sl*pyro_pos[mu];
			auto J1 = spin_idx(R1);

			for (int nu = (mu+1); nu<4; nu++){
				auto R2 = R_fcc + (1-sl)/2 * B_diamond + sl*pyro_pos[nu];

				auto J2 = spin_idx(R2);
				/*
				printf("ISING %+1lld %+1lld %+1lld (idx %d) --  %+1lld %+1lld %+1lld (idx %d)\n", 
						R1[0],R1[1],R1[2], J1, 
						R2[0],R2[1],R2[2], J2);

						*/

				ops += Op("EXCHANGE", "Jpm", {J1, J2});
				ops += Op("ISING", "Jzz", {J1, J2});
			}
		}
	}
}


std::vector<int> get_translation_symetries(const ivec3& dx) {
	std::vector<int> res(16);
	
	for (const auto& R_fcc : FCC_pos){
		for (int mu=0; mu<4; mu++){
			auto R1 = R_fcc + pyro_pos[mu];
			auto J1 = spin_idx(R1);
			res[J1] = spin_idx(R1 + dx);
		}
	}
	return res;
}



void add_magnetic_field(OpSum& ops){
	for (const auto& R_fcc : FCC_pos){
		for (int mu=0; mu<4; mu++){
			auto R1 = R_fcc + pyro_pos[mu];
			auto J1 = spin_idx(R1);
		  	ops += Op("S+", field_proj_label[mu], J1);
		  	ops += Op("S-", field_proj_label[mu], J1);
		}
	}
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

vector<std::pair<ivec3, Op>> get_hexa_list(){
	auto ring = ring_flip();
	vector<std::pair<ivec3, Op>> retval;
	for (int psl=0;psl<4;psl++){
		for (const auto& R_dfcc : dual_FCC_pos){

			auto R = R_dfcc + pyro_pos[psl];
			vector<int64_t> hex_ind;
			for (int j=0; j<6; j++){
				hex_ind.push_back(spin_idx(R+plaqt[psl][j]));
			}
			auto plaq = Op("hexa", ring, hex_ind);

			retval.push_back(std::make_pair(R, plaq));
		}
	}

	return retval;
}


int main(int argc, char** argv) try {
	if (argc < 5){
		cout <<"USAGE: "<<argv[0]<<" <Jpm/Jzz> <hx> <hy> <hz> [num_kept_states = 4]\n";
		return 1;
	}

	set_verbosity(1);// set verbosity for monitoring progress
					 //

	int num_kept_states = 4;
	if (argc >= 6) num_kept_states = atoi(argv[5]);


	OpSum ops;

	// BUILDING THE HAMILTONIAN
	//////////////////////////////////////////////////////////////////////
	///
	// defined types: HB, S+, S-, Sz, EXCHANGE
	// TODO S+S+ S-S-
	add_tetras(ops, 1);
	add_tetras(ops, -1);
	add_magnetic_field(ops);
	
	ops["Jzz"] = 1.;
	ops["Jpm"] = atof(argv[1]);
		
	arma::vec3 B = {atof(argv[2]), atof(argv[3]), atof(argv[4])};
	for (int mu=0; mu<4; mu++){
		ops[field_proj_label[mu]] = (double) arma::dot(B,pyro_pos[mu])/sqrt(3);
		cout << "B.e_mu "<<mu<<" "<<ops[field_proj_label[mu]]<<"\n";
	}
	

    std::stringstream label;
    label << "%jpm=" << ops["Jpm"] << "%B=" << B[0]<<","<<B[1]<<","<<B[2];
	/////////////////////////////////////////////
	//// PERFORMING THE DIAGONALISATION 
	// k=space symmetries
	//

	std::vector<int> T1 = get_translation_symetries({4,4,0});
	std::vector<int> T2 = get_translation_symetries({0,4,4});
	Permutation p1 (T1);
	Permutation p2 (T2);
	auto g = generated_group({p1, p2});
	// tere be four irreps of this 
	std::vector<Representation> irreps = {
		generated_irrep({p1,p2}, {1,1}),
		generated_irrep({p1,p2}, {1,-1}),
		generated_irrep({p1,p2}, {-1,1}),
		generated_irrep({p1,p2}, {-1,-1})
	};

	std::vector<arma::vec> spec_list; // container for the energies
	std::vector<std::vector<arma::mat>> ringflip_list; // container for the RF evals
	auto hexas = get_hexa_list();

	OpSum symmetrised_hexas_0 = xdiag::symmetrize(hexas[0].second, g);
	OpSum symmetrised_hexas_1 = xdiag::symmetrize(hexas[1].second, g);
	OpSum symmetrised_hexas_2 = xdiag::symmetrize(hexas[2].second, g);
	OpSum symmetrised_hexas_3 = xdiag::symmetrize(hexas[3].second, g);
	
	for (int q=0; q<4; q++){
		 Log("Dynamical Lanczos iterations for q={}", q);

		auto block = Spinhalf(16, g, irreps[q]);

		auto H = matrix(ops, block);
		arma::vec eigval;
		arma::mat eigvec;
		arma::eig_sym(eigval, eigvec, H);

		arma::mat U = eigvec.cols(0,num_kept_states);

		std::cout<<"Energy values:\n"<<eigval<<std::endl;
		std::cout << "Ringflip in energy basis: \n";

		std::vector<arma::mat> ringflip_mats;
		ringflip_mats.push_back(U.t() * matrix(symmetrised_hexas_0, block) * U);
		ringflip_mats.push_back(U.t() * matrix(symmetrised_hexas_1, block) * U);
		ringflip_mats.push_back(U.t() * matrix(symmetrised_hexas_2, block) * U);
		ringflip_mats.push_back(U.t() * matrix(symmetrised_hexas_3, block) * U);


		
		spec_list.push_back(eigval);
		ringflip_list.push_back(ringflip_mats);
	}

	//////////////////////////////////////////
	//// OUTPUT
	///
	///
	json out;
	out["lattice"] = {};

	out["energies"] = {};
	out["energies"]["1,1"] = spec_list[0];
	out["energies"]["1,-1"] = spec_list[1];
	out["energies"]["-1,1"] = spec_list[2];
	out["energies"]["-1,-1"] = spec_list[3];

	out["ringflip"]["1,1"] =   ringflip_list[0];
	out["ringflip"]["1,-1"] =  ringflip_list[1];
	out["ringflip"]["-1,1"] =  ringflip_list[2];
	out["ringflip"]["-1,-1"] = ringflip_list[3];



	out["Jpm"] = atof(argv[1]);
	out["B"] = B;

	std::ofstream file("output/out_pyro16_full_"+label.str()+".json");
    file << out;
} catch (Error e) {
	error_trace(e);
}
