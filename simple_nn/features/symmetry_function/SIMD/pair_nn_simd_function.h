#include <immintrin.h>
#include <algorithm>
#ifdef _WIN32
#include <cmath>
inline void sincos(double x, double* sinx, double* cosx){
    *sinx = sin(x);
    *cosx = cos(x);   
}
#else
#include <math.h>
#endif
#include "mkl.h"
#include <assert.h>

namespace NN_SIMD_NS {
  const int IMPLEMENTED_TYPE[] = {2, 4, 5}; // compatibility for original code

  // ex ) number of double value in mm256 (256bit = 32byte = 8byte(double's size) * 4)
#define ALIGN_NUM 64
#define DATASIZE 8 //double precision

#define MM256_DOUBLE_LEN 4
#define MM512_DOUBLE_LEN 8
#define MM256_FLOAT_LEN 8

  //static const double* VecZero;
  static double* VecZero; //this should be const double see ReLU function
  static double* VecOne;
  static double* VecTwo;

  static void init_simd() {
    double* tmpzero = (double*)_mm_malloc(400 * DATASIZE, ALIGN_NUM);
    double* tmpone = (double*)_mm_malloc(400 * DATASIZE, ALIGN_NUM);
    double* tmptwo = (double*)_mm_malloc(400 * DATASIZE, ALIGN_NUM);
    for(int i=0; i<400; i++) {
      tmpzero[i] = 0.0;
      tmpone[i] = 1.0;
      tmptwo[i] = 2.0;
    }
    VecZero = tmpzero;
    VecOne = tmpone;
    VecTwo = tmptwo;
  }

  static void fin_simd() {
    _mm_free(VecZero);
    _mm_free(VecOne);
    _mm_free(VecTwo);
  }

  struct AlignedMultiArr {
    int *idx_addr=nullptr;
    double *v=nullptr;
    int max_idx;
    int true_size;
    AlignedMultiArr() {}
    AlignedMultiArr(const int* size, const int max_idx) : max_idx(max_idx) {
      int totalsize = 0;
      idx_addr = new int[max_idx];
      for (int i = 0; i<max_idx; i++) {
        idx_addr[i] = totalsize;
        totalsize += size[i];
      }
      true_size = totalsize;
      v = (double*)_mm_malloc(totalsize*DATASIZE, ALIGN_NUM);
      //v = (double*)mkl_malloc(totalsize*DATASIZE, ALIGN_NUM);
    }
    //deep copy
    AlignedMultiArr(const AlignedMultiArr& copy) : max_idx(copy.max_idx) {
      idx_addr = new int[max_idx];
      for (int i = 0; i<max_idx; i++) {
        idx_addr[i] = copy.idx_addr[i];
      }
      v = (double*)_mm_malloc(copy.true_size*DATASIZE, ALIGN_NUM);
      //v = (double*)mkl_malloc(copy.true_size*DATASIZE, ALIGN_NUM);
      for (int j=0; j<copy.true_size; j++) {
        v[j] = copy.v[j];
      }
    }
    void init(const int* size, const int max_idx) {
      if(idx_addr != nullptr || v != nullptr) {
        assert("Wrong init on AlignedMultiArr!\n");
      }
      this->max_idx = max_idx;
      int totalsize = 0;
      idx_addr = new int[max_idx];
      for (int i = 0; i<max_idx; i++) {
        idx_addr[i] = totalsize;
        totalsize += size[i];
      }
      true_size = totalsize;
      v = (double*)_mm_malloc(totalsize*DATASIZE, ALIGN_NUM);
      //v = (double*)mkl_malloc(totalsize*DATASIZE, ALIGN_NUM);
    }

    ~AlignedMultiArr() {
      _mm_free(v);
      v = nullptr;
      delete [] idx_addr;
      idx_addr = nullptr;
    }

    inline double* operator[](const int idx) const {
      return v+idx_addr[idx];
    }
  };

  static inline void actifunc_linear_vectorized(double* nodes, double* deriv, const int size) {
    std::fill(deriv, deriv + size, 1.0);
  }

  static inline void actifunc_sigmoid_vectorized(double* nodes, double* deriv, const int size) {
    double* tmp = (double*)_mm_malloc(size*DATASIZE, ALIGN_NUM);

    vdExp(size, nodes, nodes); //exp(nodes)
    vdAdd(size, nodes, VecOne, nodes); // 1+exp(nodes)
    vdInv(size, nodes, nodes); // nodes = 1/(1+exp(nodes))
    vdSub(size, VecOne, nodes, nodes); // sigmoid(nodes) = 1- 1/(1+exp(nodes))

    //here nodes became sigmoid(nodes)
    vdSub(size, VecOne, nodes, tmp); // 1-nodes
    vdMul(size, tmp, nodes, deriv); //deriv = nodes*(1-nodes)

    _mm_free(tmp);
  }

  static inline void actifunc_tanh_vectorized(double* nodes, double* deriv, const int size) {
    vdMul(size, nodes, VecTwo, nodes); // nodes *= 2
    vdExp(size, nodes, nodes); //exp(2*nodes)
    vdAdd(size, nodes, VecOne, nodes); // 1+exp(2*nodes)
    vdInv(size, nodes, nodes); // nodes = 1/(1+exp(2*nodes))
    vdSub(size, VecOne, nodes, nodes); // sigmoid(2*nodes) = 1- 1/(1+exp(nodes))
    vdMul(size, nodes, VecTwo, nodes); // nodes = 2sigmoid(2*nodes) 
    vdSub(size, nodes, VecOne, nodes); // nodes = 2sigmoid(2*nodes) -1

    //deriv
    vdMul(size, nodes, nodes, deriv);
    vdSub(size, VecOne, deriv, deriv);
  }

  static inline void actifunc_relu_vectorized(double* nodes, double* deriv, const int size) {
    double* tmp = (double*)_mm_malloc(size*DATASIZE, ALIGN_NUM);
    cblas_dcopy(size, nodes, 1, tmp, 1); //tmp = nodes;

    //in intel API vdFmax(n, a, b, y) -> a, b are const double* but compiler says argument type is not match
    //I think this is bug. or I'm looking wrong version of mkl docs
    vdFmax(size, nodes, VecZero, nodes); //nodes = ReLU(nodes)
    vdDiv(size, nodes, tmp, deriv);

    _mm_free(tmp);
  }

  //TODO: vectorized it
  static inline void actifunc_selu_vectorized(double* nodes, double* deriv, const int size) {
    static const double alpha = 1.6732632423543772848170429916717;
    static const double scale = 1.0507009873554804934193349852946;
    for (int i=0; i<size; i++) {
      if (nodes[i] > 0) {
        deriv[i] = scale;
        nodes[i]*=scale;
      } else {
        deriv[i] = scale*alpha*exp(nodes[i]);
        nodes[i] = deriv[i] - scale*alpha;
      }
    }
  }

  //need debug
  static inline void actifunc_swish_vectorized(double* nodes, double* deriv, const int size) {
    /*
     * origianl swish
     double expl = 1./(1.+exp(-x));
     double dexpl = expl*(1-expl);
     deriv =  expl + x*dexpl;
     return x*expl;
     */

    double* tmp = (double*)_mm_malloc(size*DATASIZE, ALIGN_NUM);
    double* tmp2 = (double*)_mm_malloc(size*DATASIZE, ALIGN_NUM);

    vdExp(size, nodes, tmp); //exp(nodes)
    vdAdd(size, tmp, VecOne, tmp); // 1+exp(nodes)
    vdInv(size, tmp, tmp); // nodes = 1/(1+exp(nodes))
    vdSub(size, VecOne, tmp, tmp); // tmp = 1- 1/(1+exp(nodes)) = sigmoid(nodes) = expl

    vdSub(size, VecOne, tmp, tmp2); // 1-sigmoid(nodes)
    vdMul(size, tmp2, tmp, tmp2);
    vdMul(size, tmp2, nodes, tmp2); //tmp2 = d/dx(sigmoid) = dexpl
    vdAdd(size, tmp, tmp2, deriv); //deriv = x*d/dx(simgoid) + sigmoid

    vdMul(size, tmp, nodes, nodes); //nodes = sigmoid(nodes)*nodes = swish(nodes)

    _mm_free(tmp);
    _mm_free(tmp2);
  }


  //m256 double version
  struct SIMD_double {
    __m256d v;
    SIMD_double() {}
    SIMD_double(const __m256d in) : v(in) {}
    operator __m256d() const { return v;}
    /*
       inline double& operator[](const int idx) {
       return v[idx];
       }
       */
    inline double operator[](const int idx) const {
      return v[idx];
    }
  };

  inline SIMD_double operator+(const SIMD_double &one, const SIMD_double &two) {
    return _mm256_add_pd(one,two);
  }
  inline SIMD_double operator-(const SIMD_double &one, const SIMD_double &two) {
    return _mm256_sub_pd(one,two);
  }
  inline SIMD_double operator*(const SIMD_double &one, const SIMD_double &two) {
    return _mm256_mul_pd(one,two);
  }
  inline SIMD_double operator+(const SIMD_double &one, const double &two) {
    return _mm256_add_pd(one,_mm256_set1_pd(two));
  }
  inline SIMD_double operator*(const SIMD_double &one, const double &two) {
    return _mm256_mul_pd(one,_mm256_set1_pd(two));
  }
  inline SIMD_double operator-(const SIMD_double &one, const double &two) {
    return _mm256_sub_pd(one,_mm256_set1_pd(two));
  }

  inline SIMD_double fmadd(const SIMD_double a, const SIMD_double b, const SIMD_double c) {
    return a*b+c;
    //return _mm256_fmadd_pd(a,b,c);
  }

  inline SIMD_double SIMD_pow_ori(const SIMD_double &base, const SIMD_double expo) {
    return _mm256_pow_pd(base, expo);
  }

  //expo[0, 1, 2, 3] will be used
  //this is SEQUENTIAL. Use it carefully!!! assume double value inside expo is saft to convert to integer after abs!!
  inline SIMD_double SIMD_pow(const SIMD_double &base, const SIMD_double expo) {
    SIMD_double res = _mm256_set1_pd(1.0);
    for (int i=0; i<MM256_DOUBLE_LEN; i++) {
      int nn = abs(expo[i]);
      double tmp = base[i];
      for (; nn !=0; nn >>= 1, tmp *= tmp) {
        if (nn & 1) {
          res.v[i] *= tmp;
        }
      }
    }
    return res;
  }

  inline SIMD_double SIMD_exp(const SIMD_double &one) {
    return _mm256_exp_pd(one);
  }
  //-------------------------------------------------------------------------------------------------------------//
  struct SIMD_512_double {
    __m512d v;
    SIMD_512_double() {}
    SIMD_512_double(const __m512d in) : v(in) {}
    operator __m512d() const { return v;}
    /*
       inline double& operator[](const int idx) {
       return v[idx];
       }
       */
    inline double operator[](const int idx) const {
      return v[idx];
    }
  };

  inline SIMD_512_double operator+(const SIMD_512_double &one, const SIMD_512_double &two) {
    return _mm512_add_pd(one,two);
  }
  inline SIMD_512_double operator-(const SIMD_512_double &one, const SIMD_512_double &two) {
    return _mm512_sub_pd(one,two);
  }
  inline SIMD_512_double operator*(const SIMD_512_double &one, const SIMD_512_double &two) {
    return _mm512_mul_pd(one,two);
  }
  inline SIMD_512_double operator+(const SIMD_512_double &one, const double &two) {
    return _mm512_add_pd(one,_mm512_set1_pd(two));
  }
  inline SIMD_512_double operator*(const SIMD_512_double &one, const double &two) {
    return _mm512_mul_pd(one,_mm512_set1_pd(two));
  }
  inline SIMD_512_double operator-(const SIMD_512_double &one, const double &two) {
    return _mm512_sub_pd(one,_mm512_set1_pd(two));
  }

  inline SIMD_512_double fmadd(const SIMD_512_double a, const SIMD_512_double b, const SIMD_512_double c) {
    return _mm512_fmadd_pd(a,b,c);
  }
  inline SIMD_512_double SIMD_pow_ori(const SIMD_512_double &base, const SIMD_512_double expo) {
    return _mm512_pow_pd(base, expo);
  }

  //expo[0, 1, 2, 3] will be used
  //this is SEQUENTIAL. Use it carefully!!! assume double value inside expo is saft to convert to integer after abs!!
  //this is really bad. should make integer version of expo
  inline SIMD_512_double SIMD_pow(const SIMD_512_double &base, const SIMD_512_double expo) {
    SIMD_512_double res = _mm512_set1_pd(1.0);
    for (int i=0; i<MM512_DOUBLE_LEN; i++) {
      int nn = abs(expo[i]);
      double tmp = base[i];
      for (; nn !=0; nn >>= 1, tmp *= tmp) {
        if (nn & 1) {
          res.v[i] *= tmp;
        }
      }
    }
    return res;
  }

  inline SIMD_512_double SIMD_exp(const SIMD_512_double &one) {
    return _mm512_exp_pd(one);
  }

  //CPU dependent macro
  // sizeof(double) = 8 bytes
#ifdef __AVX512F__ 
#define SIMD_V_LEN MM512_DOUBLE_LEN
  using simd_v = SIMD_512_double;
#else
#define SIMD_V_LEN MM256_DOUBLE_LEN
  using simd_v = SIMD_double;
#endif

  inline simd_v SIMD_load_aligned(const double *p) {
#ifdef __AVX512F__ 
    return _mm512_load_pd(p);
#else
    return _mm256_load_pd(p);
#endif
  }
  inline simd_v SIMD_load(const double *p) {
#ifdef __AVX512F__ 
    return _mm512_loadu_pd(p);
#else
    return _mm256_loadu_pd(p);
#endif
  }
  inline simd_v SIMD_set(const double v) {
#ifdef __AVX512F__ 
    return _mm512_set1_pd(v);
#else
    return _mm256_set1_pd(v);
#endif
  }
  inline simd_v SIMD_set_two(const double v1, const double v2) {
#ifdef __AVX512F__ 
    return _mm512_set_pd(v1, v2, 0, 0, 0, 0, 0, 0);
#else
    return _mm256_set_pd(0, 0, 0, 0);
#endif
  }
  inline simd_v SIMD_gather(double const * base, const int* vindex) {
#ifdef __AVX512F__ 
    return _mm512_i32gather_pd(_mm256_loadu_epi32(vindex),base, 8);
#else
#ifdef __AVX2__
    return _mm256_i32gather_pd(base, _mm_loadu_si128((__m128i*)vindex), 8);
#else
    return; //code should not reach here
#endif
#endif
  }

  

  /*
  inline void SIMD_store(double * to, simd_v from) {
    _mm256_store_pd(to, from);
  }
  */

  inline simd_v SIMD_exp_approximate(const simd_v val, simd_v& deriv) {
    static const simd_v ef = SIMD_set(1.0/257.0);
    simd_v x_ori = (val*ef + 1.0);
    simd_v x = x_ori;
    x = x * x;
    x = x * x;
    x = x * x;
    x = x * x;
    x = x * x;
    x = x * x;
    x = x * x;
    x = x * x;
    deriv = x;
    return x*x_ori;
  }
  static inline void cutf2_noslot(const double dist, const double cutd, double& f, double& df) {
    double cos, sin;
    static const double H_PI = -M_PI*0.5;
    sincos(M_PI*dist/cutd, &sin, &cos);
    f = 0.5 * (1 + cos);
    df = H_PI * sin / cutd;
  }
}
