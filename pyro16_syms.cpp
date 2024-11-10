#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>
#include <xdiag/algebra/algebra.hpp>
#include <xdiag/algebra/matrix.hpp>
#include <xdiag/algorithms/lanczos/eigs_lanczos.hpp>
#include <xdiag/algorithms/lanczos/eigvals_lanczos.hpp>
#include <xdiag/all.hpp>
#include <armadillo>
#include <xdiag/blocks/spinhalf.hpp>
#include <xdiag/common.hpp>
#include <xdiag/operators/opsum.hpp>
#include <xdiag/symmetries/permutation.hpp>
#include <xdiag/utils/say_hello.hpp>

using namespace xdiag;


// the Paulis
static const arma::mat sigma_x("0 1; 1 0");
static const arma::cx_mat sigma_y(arma::mat("0 0; 0 0"), arma::mat("0 -1; 1 0"));
static const arma::mat sigma_z("1 0; 0 -1");


using namespace std;
using ivec3=arma::ivec3;

const static vector<ivec3> FCC_pos = {
	{0,0,0},
	{0,4,4},
	{4,0,4},
	{4,4,0}
};

const static vector<ivec3> dual_FCC_pos = {
	{4,4,4},
	{4,0,0},
	{0,4,0},
	{0,0,4}
};

const static vector<ivec3> pyro_pos = {
	{1,1,1},
	{1,-1,-1},
	{-1,1,-1},
	{-1,-1,1}
};

const ivec3 plaqt[4][6] = {
    {
	{ 0,-2, 2},
	{ 2,-2, 0},
	{ 2, 0,-2},
	{ 0, 2,-2},
	{-2, 2, 0},
	{-2, 0, 2}},
    {
	{ 0, 2,-2},
	{ 2, 2, 0},
	{ 2, 0, 2},
	{ 0,-2, 2},
	{-2,-2, 0},
	{-2, 0,-2}},
    {
	{ 0,-2,-2},
	{-2,-2, 0},
	{-2, 0, 2},
	{ 0, 2, 2},
	{ 2, 2, 0},
	{ 2, 0,-2}},
    {
	{ 0, 2, 2},
	{-2, 2, 0},
	{-2, 0,-2},
	{ 0,-2,-2},
	{ 2,-2, 0},
	{ 2, 0, 2}}
};


const static vector<ivec3> pyro_sites = {
{1,1,1},
{1,7,7},
{7,1,7},
{7,7,1},
{1,5,5},
{1,3,3},
{7,5,3},
{7,3,5},
{5,1,5},
{5,7,3},
{3,1,3},
{3,7,5},
{5,5,1},
{5,3,7},
{3,5,7},
{3,3,1}
};


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
	for (int i=0; i<pyro_sites.size(); i++){
		if (arma::all(R == pyro_sites[i])) {return i;}
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
	for (int psi=0; psi<64; psi++){
		int res = psi;
		double coeff=1; // this is 0 or 1
						
		for (int i=0; i<6; i++){
			bool spin_i_up = ((res&(1<<i)) != 0);

			if ( i%2 == 0 ){
				// S+	
				if (spin_i_up){
					coeff=0;
					break;
				}
				// flip they spin
				res ^= (1<<i);
			} else {
				// S-
				if (!spin_i_up) {
					//spin is down, can't lower a spin
					coeff=0;
					break;
				}
				// flip they spin
				res ^= (1<<i);
			}
		}
		if (fabs(coeff) > 1e-4){
			printf("%d | %2x -> %2x\n", psi, psi, res);
			O(res, psi) = coeff;
		}
	}
	// equivalent code: M(0b101010,0b010101)=1
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

std::vector<int> calc_perm_of_tl(ivec3 tl){
	std::vector<int> l;
	l.resize(16);
	std::set<int> used_idx; // keep track to make sure it's actually a perm
	for (int J=0; J<16; J++){

		int shift_J = spin_idx(pyro_sites[J]+tl);
		auto [it, success] = used_idx.insert(shift_J);
		if (!success){
			stringstream ss;
			ss << "Permutation is not a permutation:";
			ss << "Site "<<J<<" @ "<< 
				pyro_sites[J][0]<<" "<<pyro_sites[J][1]<<" "<<pyro_sites[J][2]
				<<" is mapped to idx "<<shift_J;
			throw std::logic_error(ss.str());
		}
		l[J] = shift_J;
	}
	return l;
}

std::vector<Permutation> calc_symmetries(){	
	std::vector<Permutation> syms;
	for (int i=1; i<4; i++){
		syms.push_back(
				Permutation(calc_perm_of_tl(FCC_pos[i]))
				);
	}
	return syms;
}


int main(int argc, char** argv) try {
	if (argc < 6){
		cout <<"USAGE: "<<argv[0]<<" <Jpm/Jzz> <hx> <hy> <hz> <eig_number>\n";
		return 1;
	}

	set_verbosity(1);// set verbosity for monitoring progress
					 //
	int NSTATES = atoi(argv[5]);


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
	ops["Jpm"] = atof(argv[1])/2;
		
	arma::vec3 B = {atof(argv[2]), atof(argv[3]), atof(argv[4])};
	for (int mu=0; mu<4; mu++){
		ops[field_proj_label[mu]] = (double) arma::dot(B,pyro_pos[mu])/sqrt(3);
		cout << "B.e_mu "<<mu<<" "<<ops[field_proj_label[mu]]<<"\n";
	}

	// CALCULATING SYMMETRIES
	auto syms = calc_symmetries();
	auto permgroup = PermutationGroup(syms);
	

	/////////////////////////////////////////////
	//// PERFORMING THE DIAGONALISATION (Lanczos)
	///
	auto block = Spinhalf(16);

	auto lanczos_res = eigs_lanczos(ops, block, 
			NSTATES,
			/* precision */ 1e-14,
			/* max iter */ 10000,
			/* force complex */ true
			);
	cerr << "iter criterion -> "<< lanczos_res.criterion << endl;

	auto gs = lanczos_res.eigenvectors.col(0);
	gs.make_complex();


	std::vector<eigs_lanczos_result_t> block_results;

	for (const auto& [k, irrep] : irreps){
		auto block = Spinhalf(16, permgroup, irrep);
		auto lanczos_res = eigvals_lanczos(ops, block, 4); // compute ground state energy
		cerr << "k = " << k << ", iter criterion -> " <<  lanczos_res.criterion << endl;
		eigvals.push_back(lanczos_res);
	}
	
	//////////////////////////////////////////
	//// OUTPUT
	///

	std::cout<<"Energy values:\n"<<lanczos_res.eigenvalues<<std::endl;

	std::cout<<"SZ expectation values:\n";
	std::cout<<"sublat\tX\tY\tZ\t<Sz>\n"; 

	for (int J=0; J<16; J++){
		auto R1=pyro_sites[J];

		arma::cx_double SZ = innerC(Op("SZ",1, J), gs);
		printf("%d\t%+1lld\t%+1lld\t%+1lld\t%16.7e\n", 
				J%4, R1[0],R1[1],R1[2], SZ.real());
	}
	std::cout<<"Plaq expectation values:\n";
	auto hexa_list = get_hexa_list();
	std::cout<<"sublat\tX\tY\tZ\tRe<O>\tIm<O>\n"; 

	int sl=0;
	for (auto& [R1, op] : hexa_list){
		arma::cx_double exp_O = innerC(op, gs);
	
		printf("%d\t%+1lld\t%+1lld\t%+1lld\t%16.7e\t%16.7e\n", 
				sl, R1[0],R1[1],R1[2],
				exp_O.real(), exp_O.imag());

		sl = (++sl)%4;
	}


	

} catch (Error e) {
	error_trace(e);
}
