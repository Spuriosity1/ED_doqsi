#pragma once
#include "vec3.hpp"
#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>

struct Rational {
	int64_t num;
	int64_t denom;

	Rational(int64_t _num) : num(_num), denom(1){};

	Rational(int64_t _num, int64_t _denom) : num(_num), denom(_denom) {};

	Rational& operator+=(const Rational& other){
		int64_t tmp = static_cast<int64_t>(num)*other.denom + static_cast<int64_t>(other.num)*denom;
		denom = denom * other.denom;
		num = tmp;
		return *this;
	}

	Rational& operator-=(const Rational& other){
		int64_t tmp = static_cast<int64_t>(num)*other.denom - static_cast<int64_t>(other.num)*denom;
		denom = denom * other.denom;
		num = tmp;
		return *this;
	}

	void simplify(){
		int64_t this_gcd = std::gcd(num, denom);
		num /= this_gcd;
		denom /= this_gcd;
	}

	Rational operator/=(int64_t x) {
		denom *= x;
		return *this;
	}


	Rational operator/=(const Rational& x) {
		denom *= x.num;
		num *= x.denom;
		return *this;
	}

	Rational operator*=(int64_t x) {
		num *= x;
		return *this;
	}

	Rational operator*=(const Rational& x) {
		num *= x.num;
		denom *= x.denom;
		return *this;
	}

	Rational operator/(int64_t x) const {
		return Rational(num, denom*x);
	}

	bool operator==(int64_t x) const {
		return num == x * denom;
	}

	bool operator==(const Rational& r) const {
		return static_cast<int64_t>(num) * r.denom == static_cast<int64_t>(r.num)*denom;
	}
};

inline Rational operator*(int64_t x, const Rational& f){
	return Rational(x*f.num, f.denom);
}


inline Rational operator*(const Rational& x, const Rational& f){
	return Rational(x.num*f.num, x.denom*f.denom);
}

inline Rational operator+(const Rational& x, const Rational& y){
	Rational retval(x);
	retval += y;
	return retval;
}


typedef vector3::vec3<Rational> rvec3;
typedef vector3::mat33<Rational> rmat33;

//TODO template this BS
inline void rswap(rmat33& A, int row_i, int row_j, rvec3& b){
	for (int j=0; j<3; j++){
		std::swap(A(row_i, j), A(row_j, j));
	}
	std::swap(b(row_i), b(row_j));
}

// Performs A[row_i] <- a*A[row_j] - A[row_i]
inline void rsub(rmat33& A, Rational a_j, int row_j, int row_i, rvec3& b){
	for (int j=0; j<3; j++){
		A(row_i, j) -= a_j*A(row_j, j);
	}
	b(row_i) -= a_j*b(row_j);
}

inline void rmult(rmat33& A, int row, Rational a, rvec3& b){
	for (int j=0; j<3; j++){
		A(row, j) *= a;
	}
	b(row) *= a;
}

inline Rational inv(const Rational& a){
	return Rational(a.denom, a.num);
}

// Solves Ax = b using Gaussian elimination
inline void rlinsolve(rvec3& x, const rmat33& A, const rvec3& b){
	rmat33 B(A);
	x = b;
	
	for (int col = 0; col<3; col++){
		// pemute rows until A(col, col) != 0
		int row = col + 1;
		while ((B(col,col) == 0)) {
			if (row > 3){
				throw std::invalid_argument("Matrix is singular");
			}

			rswap(B, col, row, x);
			row += 1;
		}

		// rmult to make B(col, col) == 1
		rmult(B, col, inv(B(col,col)), x);
		// simplify the row
		for (int ci=col; ci<3; ci++){
			B(col, ci).simplify();
		}

		// delete the lower part
		for (row = col + 1; row < 3; row++){
			rsub(B, B(row, col), col, row, x);
		}

		// Simplify
		for (int i=0; i<9; i++){
			B[i].simplify();
		}
	}


	// matrix shold now be upper triangular with 1s on diagonal
	// Remove top part and simplify answer
	for (int col=2; col>=0; col--){
		for (int row=0; row<col; row++){
			rsub(B, B(row,col), col, row, x);
		}
		x[col].simplify();
	}

	assert(A * x == b);

}


inline std::ostream& operator<<(std::ostream& os, const Rational& r){
	os << r.num << "/"<<r.denom;
	return os;
}

inline std::istream& operator>>(std::istream& is, Rational& r){
	is >> r.num;
	if (is.get() != '/'){
		r.denom=1;
	} else {
		is >> r.denom;
	}
	return is;
}

