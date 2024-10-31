#include "../rationalmath.hpp"
#include <cassert>
#include <sstream>

int main (int argc, char *argv[]) {
	std::array<Rational, 3> a0{ Rational(0,2), Rational(1,2), Rational(0,4) };
	std::array<Rational, 3> a1{ Rational(1,3), Rational(0,1), Rational(1,4) };
	std::array<Rational, 3> a2{ Rational(0,1), Rational(1,2), Rational(-3,7) };

	Rational r00(0,0);

	rvec3 b(r00, r00, r00);

	assert(argc == 4);
	for (int i=0; i<3; i++){
		std::stringstream iss(argv[i+1]);
		iss >> b[i];
	}
	rmat33 M = rmat33::from_cols(a0, a1, a2);

	rvec3 x;
	rlinsolve(x, M, b);
	std::cout << "Solved Mx=b with M="<<M<<", b="<<b<<"\n";
	std::cout << "Result: x="<<x << "\n";
	rvec3 y = M * x;
	for (int i=0; i<3; i++){ y[i].simplify(); }
	std::cout << "Mx = "<<y<<", b= "<<b<<"\n";
	assert(M * x == b);
	
	return 0;
}



