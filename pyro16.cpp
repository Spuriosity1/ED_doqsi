#include <cmath>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <nlohmann/json_fwd.hpp>
#include <ostream>
#include <stdexcept>
#include <utility>
#include <xdiag/all.hpp>
#include <nlohmann/json.hpp>
#include <sstream>
#include "pyrochlore_geometry.hpp"

using namespace xdiag;
using json = nlohmann::json;
using namespace pyro;
using namespace std;
using ivec3=arma::ivec3;

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



ivec3 wrap(const ivec3& x){
	ivec3 y(x);
	y[0] = (y[0]%8 + 8)%8;
	y[1] = (y[1]%8 + 8)%8;
	y[2] = (y[2]%8 + 8)%8;
	return y;
}

inline vector<ivec3> calc_pyro_sites(){
	vector<ivec3> x;
	for (int fcc=0; fcc<4; fcc++){
		for (int pyro=0; pyro<4; pyro++){
			x.push_back(wrap(FCC_pos[fcc] + pyro_pos[pyro] ));
		}
	}
	return x;
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
	for (const auto& R_dfcc : dual_FCC_pos){
		for (int psl=0;psl<4;psl++){

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


arma::mat evaluate_gs_matrix(const OpSum& O, const std::vector<State>& gs_set){
	arma::mat out(gs_set.size(), gs_set.size());	
	for (int i=0; i<gs_set.size(); i++){
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


arma::cx_mat evaluate_gs_matrixC(const OpSum& O, const std::vector<State>& gs_set){
	arma::cx_mat out(gs_set.size(), gs_set.size());	
	for (int i=0; i<gs_set.size(); i++){
		// the diagonal
		out(i,i) = innerC(O, gs_set[i]);
		for (int j=0; j< i; j++){
			auto w = gs_set[j];
			apply(O, gs_set[j], w);
			xdiag::complex val = dotC(gs_set[i], w);
			out(i,j) = val;
			out(j,i) = val;
		}
	}
	return out;
}


void evaluate_exp_Sz(const std::vector<State>& gs_set, json& out){
	// Stores the gs_Set.size()^2-dim matrix <g1 | Sz | g2> for all sites
	// in format [J] 
	//
	//
	
	std::cout<<"SZ expectation values:\n";
	std::cout<<"sublat\tX\tY\tZ\t<Sz>\n"; 
	
	std::vector<arma::mat> Sz_list;
	for (int J=0; J<16; J++){
		auto R1=pyro16_sites[J];

        OpSum Sz({Op("SZ", 1, J)});

		auto sz_mat = evaluate_gs_matrix(Sz,gs_set);
		Sz_list.push_back(sz_mat);
		std::cout << "Spin "<<J<<"<g|Sz|g> = \n"<<sz_mat<<"\n";
	}
	out["Sz"] = Sz_list;
	out["lattice"]["spin_sites"] = pyro16_sites;


}

void evaluate_ring_flip(const std::vector<State>& gs_set, json& out){
	std::cout<<"Plaq expectation values:\n";
	auto hexa_list = get_hexa_list();
	std::cout<<"sublat\tX\tY\tZ\tRe<O>\tIm<O>\n"; 

	int sl=0;

	std::vector<arma::mat> re_ringflip;
	std::vector<arma::mat> im_ringflip;
	std::vector<ivec3> plaq_sites;

	for (auto& [R1, op] : hexa_list){
		arma::cx_mat exp_O = evaluate_gs_matrixC(OpSum({op}), gs_set);
	
		printf("%d\t%+1lld\t%+1lld\t%+1lld\n", 
				sl, R1[0],R1[1],R1[2]);
		std::cout << exp_O << "\n";
		re_ringflip.push_back(real(exp_O));
		im_ringflip.push_back(imag(exp_O));
		plaq_sites.push_back(R1);
		sl = (sl + 1)%4;
	}

	out["lattice"]["plaq_sites"] = plaq_sites;
	out["re_ringflip"] = re_ringflip;
	out["im_ringflip"] = im_ringflip;
}


int main(int argc, char** argv) try {
	if (argc < 6){
		cout <<"USAGE: "<<argv[0]<<" <Jpm/Jzz> <hx> <hy> <hz> <lanczos_dim> [num_kept_states = 4]\n";
		return 1;
	}

	set_verbosity(1);// set verbosity for monitoring progress
					 //
	int lanczos_dim = atoi(argv[5]);


	int num_kept_states = 4;
	if (argc >= 7) num_kept_states = atoi(argv[6]);


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
	//// PERFORMING THE DIAGONALISATION (Lanczos)
	///
	auto block = Spinhalf(16);
	std::vector<string> statev = {"Up","Dn","Up","Dn","Up","Dn","Up","Dn","Up","Dn","Up","Dn","Up","Dn","Up","Dn"};
	auto init_state = product(block, statev);
	auto lanczos_res = eigs_lanczos(ops, block, 
			init_state, 
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

	evaluate_exp_Sz(gs_set, out);
	evaluate_ring_flip(gs_set, out);

	out["Jpm"] = atof(argv[1]);
	out["B"] = B;


	// save to file
    //
	//
    //
	std::ofstream file("output/out_pyro16_"+label.str()+".json");
    file << out;
} catch (Error e) {
	error_trace(e);
}
