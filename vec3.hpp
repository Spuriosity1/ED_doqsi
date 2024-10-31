#ifndef VEC3_CUSTOM_HPP
#define VEC3_CUSTOM_HPP


#include <cstddef>
#include <array>
#include <iostream>

namespace vector3 {


template <typename T>
struct mat33;
    
template <typename T>
struct vec3 {
	constexpr vec3() = default;
	constexpr vec3(const vec3&) = default;
	constexpr vec3(T x,T y,T z){
		m_x[0] = x;
		m_x[1] = y;
		m_x[2] = z;
	}
	constexpr vec3& operator=(const vec3&) = default;

	constexpr inline T& operator[](size_t idx){ return m_x[idx]; }
	constexpr inline T operator[](size_t idx) const { return m_x[idx];	}

	constexpr inline T& operator()(size_t idx){ return m_x[idx]; }
	constexpr inline T operator()(size_t idx) const { return m_x[idx];	}

	static constexpr size_t size(){ return 3; }

/*
    constexpr vec3& operator=(const vec3& v){
		m_x[0] = v.m_x[0];
		m_x[1] = v.m_x[1];
		m_x[2] = v.m_x[2];
        return *this;
    }
*/
    constexpr vec3& operator+=(const vec3& v){
        m_x[0] += v.m_x[0];
        m_x[1] += v.m_x[1];
        m_x[2] += v.m_x[2];
        return *this;
    }

    constexpr vec3& operator-=(const vec3& v){
        m_x[0] -= v.m_x[0];
        m_x[1] -= v.m_x[1];
        m_x[2] -= v.m_x[2];
        return *this;
    }

    template<typename S>
    constexpr vec3& operator*=(S alpha){
		T tmp = static_cast<T>(alpha);
        m_x[0] *= tmp;
        m_x[1] *= tmp;
        m_x[2] *= tmp;
        return *this;
    }

    template<typename S>
    vec3& operator/=(S alpha){
		T tmp = static_cast<T>(alpha);
        m_x[0] /= tmp;
        m_x[1] /= tmp;
        m_x[2] /= tmp;
        return *this;
    }

    template<typename S>
    vec3& operator%=(S alpha){
		T tmp = static_cast<T>(alpha);
        m_x[0] %= tmp;
        m_x[1] %= tmp;
        m_x[2] %= tmp;
        return *this;
    }

    vec3 operator+(const vec3& v) const {
        vec3 res(*this);
        return res += v;
    }

    vec3 operator-(const vec3& v) const {
        vec3 res(*this);
        return res -= v;
    }

	vec3 operator-() const {	
        vec3 res(*this);
		res.m_x[0] = -res.m_x[0];
		res.m_x[1] = -res.m_x[1];
		res.m_x[2] = -res.m_x[2];
		return res;
	}

	bool operator==(const vec3& other) const {
		return (other.m_x[0] == m_x[0]) && 
			(other.m_x[1] == m_x[1]) &&
			(other.m_x[2] == m_x[2]);
	}

	friend vec3<T> mat33<T>::operator*(const vec3<T>&) const;

protected:

    T m_x[3]={0,0,0};
};


// Free vector functions
template <typename T, typename S>
vec3<S> operator*(T alpha, const vec3<S>& v){
    vec3<S> copy(v);
	copy *= static_cast<S>(alpha);
	return copy;
}

template <typename T>
T dot(const vec3<T>& u, const vec3<T>& v){
	return u[0]*v[0] + u[1]*v[1] + u[2]*v[2];	
}

template<typename T>
vec3<T> operator+(const vec3<T>& v1, const vec3<T>& v2){
	vec3<T> w(v1);
	w += v2;
	return w;
}

template<typename T>
vec3<T> operator-(const vec3<T>& v1, const vec3<T>& v2){
	vec3<T> w(v1);
	w -= v2;
	return w;
}
	

template <typename T>
struct mat33 {
	static constexpr size_t size(){ return 9; }
	static mat33 from_cols(std::array<T,3> a0,
			std::array<T,3> a1,
			std::array<T,3> a2
			){
		mat33 out;
		for (int row=0; row<3; row++){
			out(row, 0) =a0[row];
			out(row, 1) =a1[row];
			out(row, 2) =a2[row];
		}
		return out;
	}

	vec3<T> operator*(const vec3<T>& v) const {
		vec3<T> res;
		res[0] = __dot(m_x, v.m_x);
		res[1] = __dot(m_x+3, v.m_x);
		res[2] = __dot(m_x+6, v.m_x);	
		return res;
	}

	inline T& operator()(int i, int j){return m_x[3*i + j];	}
	inline T  operator()(int i, int j) const {return m_x[3*i + j];	}

	inline T& operator[](int i)		  {return m_x[i];}
	inline T  operator[](int i) const {return m_x[i];}

	mat33<T> operator*(const mat33<T>& x) const {
		mat33<T> res;
		// ah hell the compiler can optimise this
		for (int i=0; i<3; i++){
			for (int k=0; k<3; k++){
				for (int j=0; j<3; j++){
					res(i,j) += (*this)(i, k) * x(k, j);
				}
			}
		}
		return res;
	}

	vec3<T> diagonal(){
		vec3<T> out;
		for (int i=0; i<3; i++)
			out[i] = (*this)(i,i);

		return out;
	}
		
		
protected:
	T m_x[9]={0,0,0, 0,0,0, 0,0,0}; 
	// convention:
	// [ 0 1 2 ]
	// [ 3 4 5 ]
	// [ 6 7 8 ]
	//
	T __dot(const T* x1, const T* x2) const {
		return x1[0] * x2[0] + x1[1] * x2[1] + x1[2] * x2[2];
	}
};

/// CONVENIENT TYPEDEFS
typedef vec3<int> vec3i;
typedef vec3<double> vec3d;

template<typename T>
std::ostream& operator<<(std::ostream& os, vec3<T> v){
	os << "["<<v[0]<<" "<<v[1]<<" "<<v[2]<<"]";
	return os;
}


template<typename T>
std::ostream& operator<<(std::ostream& os, mat33<T> m){
	os << "["<<m(0,0)<<"\t"<<m(0,1)<<"\t"<<m(0,2)<<"]\n";
	os << "["<<m(1,0)<<"\t"<<m(1,1)<<"\t"<<m(1,2)<<"]\n";
	os << "["<<m(2,0)<<"\t"<<m(2,1)<<"\t"<<m(2,2)<<"]\n";
	return os;
}


// Explicit functions preserving intness
template<typename V>
// requires std::signed_integral<V>
inline V det(mat33<V> a){
    return (a(0,0) * (a(1,1) * a(2,2) - a(2,1) * a(1,2))
           -a(1,0) * (a(0,1) * a(2,2) - a(2,1) * a(0,2))
           +a(2,0) * (a(0,1) * a(1,2) - a(1,1) * a(0,2)));
}

};


// HASHABILITY
//
template <typename T>
struct std::hash<vector3::vec3<T>>
{
  std::size_t operator()(const vector3::vec3<T>& k) const
  {
    using std::size_t;
    using std::hash;

    // Compute individual hash values for first,
    // second and third and combine them using XOR
    // and bit shifting:

    return hash<T>()(k[0]) ^ (hash<T>()(k[1]) << 1) ^ (hash<T>()(k[2]) << 2);
  }
};


#endif // !VEC3_CUSTOM_HPP

