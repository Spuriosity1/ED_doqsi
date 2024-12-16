#include <cmath>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <nlohmann/json_fwd.hpp>
#include <ostream>
#include <stdexcept>
#include <utility>
#include <xdiag/all.hpp>
#include <nlohmann/json.hpp>
#include <sstream>
#include <xdiag/symmetries/generated_group.hpp>
#include "pyrochlore_geometry.hpp"
#include "rationalmath.hpp"

using namespace xdiag;
using json = nlohmann::json;
using namespace pyro;
using namespace std;
using ivec3=arma::ivec3;


arma::ivec3 as_integer(const rvec3& v){
	arma::ivec3 u;
	for (int i=0; i<3; i++){
		Rational r(v[i]);
		r.simplify();
		switch (r.denom){
			case 1:
				u[i] = r.num;
				break;
			case -1:
				u[i] = -r.num;
				break;
			default:
				throw std::logic_error("cannot safely cast fraction to int");
		}
	}
	return u;
}

std::string getISOCurrentTimestamp()
{
	time_t now;
    time(&now);
    char buf[sizeof "2011-10-08T07:07:09Z"];
    strftime(buf, sizeof buf, "%FT%TZ", gmtime(&now));
	return std::string(buf);
}

static const rmat33 A = rmat33::from_cols({0,8,8},{8,0,8},{8,8,0});

ivec3 wrap(const ivec3& x){
	rvec3 rational_x = {x[0], x[1], x[2]};
	rvec3 n;
	rlinsolve(n, A, rational_x);
	// A * n = rational_x
	return as_integer(A*mod1(n));
}

int spin_idx(const ivec3& R_){
	auto R = wrap(R_);
	for (int i=0; i<pyro32_sites.size(); i++){
		if (arma::all(R == pyro32_sites[i])) {return i;}
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

std::vector<std::vector<int64_t>> plaq_indices = {
	{10, 26, 20, 12, 25, 17},
	{4, 20, 26, 2, 16, 24},
	{24, 8, 1, 25, 12, 4},
	{17, 1, 8, 16, 2, 10},
	{14, 30, 16, 8, 29, 21},
	{0, 16, 30, 6, 20, 28},
	{28, 12, 5, 29, 8, 0},
	{21, 5, 12, 20, 6, 14},
	{8, 24, 22, 14, 27, 19},
	{6, 22, 24, 0, 18, 26},
	{26, 10, 3, 27, 14, 6},
	{19, 3, 10, 18, 0, 8},
	{12, 28, 18, 10, 31, 23},
	{2, 18, 28, 4, 22, 30},
	{30, 14, 7, 31, 10, 2},
	{23, 7, 14, 22, 4, 12},
	{11, 27, 21, 13, 24, 16},
	{5, 21, 27, 3, 17, 25},
	{25, 9, 0, 24, 13, 5},
	{16, 0, 9, 17, 3, 11},
	{15, 31, 17, 9, 28, 20},
	{1, 17, 31, 7, 21, 29},
	{29, 13, 4, 28, 9, 1},
	{20, 4, 13, 21, 7, 15},
	{9, 25, 23, 15, 26, 18},
	{7, 23, 25, 1, 19, 27},
	{27, 11, 2, 26, 15, 7},
	{18, 2, 11, 19, 1, 9},
	{13, 29, 19, 11, 30, 22},
	{3, 19, 29, 5, 23, 31},
	{31, 15, 6, 30, 11, 3},
	{22, 6, 15, 23, 5, 13}};

vector<std::pair<ivec3, Op>> get_hexa_list(){
	auto ring = ring_flip();
	vector<std::pair<ivec3, Op>> retval;
	for (const auto& hex_ind : plaq_indices){

		ivec3 R = {0,0,0};
		for (int mu=0; mu<6; mu++){
			R += pyro32_sites[hex_ind[mu]];
		}
		R /= 6;
		auto plaq = Op("hexa", ring, hex_ind);

		retval.push_back(std::make_pair(R, plaq));
	}

	return retval;
}


arma::mat evaluate_gs_matrix(const OpSum& O, const std::vector<State>& gs_set){
	arma::mat out(gs_set.size(), gs_set.size());	
	for (int i=0; i<static_cast<int>(gs_set.size()); i++){
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
	for (int i=0; i<static_cast<int>(gs_set.size()); i++){
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

void parse_irrep(std::vector<xdiag::complex> characters, const char* choice){
	assert(strlen(choice) == 3);
	characters.resize(3);
	for (int i=0; i<3; i++){
		switch (choice[i]){
			case 'p': // short for "pi"
				characters[i] = -1;
				break;
			case '0':
				characters[i] = 1;
				break;
			default:
				throw new std::runtime_error("Bad character choice - must be exactly three characters, either 0 (k=0, character=1) or pi (k=pi, character=-1)");
		}
	}
}

int main(int argc, char** argv) try {
	if (argc < 7){
		cout <<"USAGE: "<<argv[0]<<" <Jpm/Jzz> <hx> <hy> <hz> <lanczos_dim> <irrep> [num_kept_states = 4]\n";
		return 1;
	}

	set_verbosity(1);// set verbosity for monitoring progress
					 //
	int lanczos_dim = atoi(argv[5]);

	std::vector<xdiag::complex> characters;
	parse_irrep(characters, argv[6]);

	int num_kept_states = 4;
	if (argc >= 8) num_kept_states = atoi(argv[7]);


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
	///  APPLYING SYMMETRIES
	


	std::vector<int> T1 = get_translation_symetries({4,4,0});
	std::vector<int> T2 = get_translation_symetries({0,4,4});
	std::vector<int> T3 = get_translation_symetries({4,0,4});
	Permutation p1 (T1);
	Permutation p2 (T2);
	Permutation p3 (T3);
	auto g = generated_group({p1, p2, p3});

	/////////////////////////////////////////////
	//// PERFORMING THE DIAGONALISATION (Lanczos)
	///
	auto irrep = generated_irrep({p1,p2,p3}, characters);
	auto block = Spinhalf(32, g, irrep);
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
