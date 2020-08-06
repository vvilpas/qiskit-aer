/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _aer_framework_linalg_almost_equal_hpp_
#define _aer_framework_linalg_almost_equal_hpp_

#include <complex>
#include <limits>
#include <type_traits>

#include "framework/matrix.hpp"
#include "framework/linalg/enable_if_numeric.hpp"

#define MAXDIFF 5.0*epsilon<T>::value
#define MAXRELATIVEDIFF 5.0*epsilon<T>::value

namespace AER {
namespace Linalg {

template<typename T>
struct epsilon {
    static constexpr auto value = std::numeric_limits<T>::epsilon();
};

/**
 * The epsilon of a complex is the epsilon from the value type of the complex.
 * eg: std::complex<double> = double
 *     std::complex<float> = float
 */
template<typename T>
struct epsilon<std::complex<T>> {
    static constexpr auto value = epsilon<T>::value;
};

/*
 * We use function overloads to dispatch to the correct library functions 
 *   wherever necessary - mainly for container types (vector, matrix) and/or
 *   structured data types (std::complex).
 * read is_tt as is-double-template (I think)
 * See here:
 *   https://stackoverflow.com/questions/41438493/how-to-identifying-whether-a-template-argument-is-stdcomplex
 *
 */
template <template <class...> class TT, class... Args>
std::true_type is_tt_impl(TT<Args...>);
template <template <class...> class TT>
std::false_type is_tt_impl(...);

template <template <class...> class TT, class T>
using is_tt = decltype(is_tt_impl<TT>(std::declval<typename std::decay<T>::type>()));

template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type >
bool almost_equal(T f1, T f2,
                  T max_diff = MAXDIFF,
                  T max_relative_diff = MAXRELATIVEDIFF);

template <typename T>
bool almost_equal(const std::complex<T>& f1, const std::complex<T>& f2,
                  T max_diff = MAXDIFF,
                  T max_relative_diff = MAXRELATIVEDIFF);
 
template <typename T>
bool almost_equal(const matrix<T>& mat1, const matrix<T>& mat2,
                  double max_diff = MAXDIFF,
                  double max_relative_diff = MAXRELATIVEDIFF);

template <typename T>
bool almost_equal(const std::vector<T>& mat1, const std::vector<T>& mat2);
 
template <typename T, typename >
bool almost_equal(T f1, T f2,
                  T max_diff,
                  T max_relative_diff) {
  T diff = std::abs(f1 - f2);
  if (diff <= max_diff) return true;
  std::cout << diff << std::endl;
  return diff <=
         max_relative_diff * std::max(std::abs(f1), std::abs(f2));
}

template <typename T>
bool almost_equal(const std::complex<T>& f1, const std::complex<T>& f2,
                  T max_diff,
                  T max_relative_diff) {
    return almost_equal<T>(f1.real(), f2.real(), max_diff, max_relative_diff)
         && almost_equal<T>(f1.imag(), f2.imag(), max_diff, max_relative_diff);
}

template <typename T>
bool almost_equal(const matrix<T>& mat1, const matrix<T>& mat2,
                  double max_diff,
                  double max_relative_diff) {
  if(mat1.size() != mat2.size()) {
       std::cout << "Matrix sizes not equal : " << mat1.size() << " != " << mat2.size() << std::endl;
       return false;
  }
  bool equal{true};
  matrix<T> diff;
  diff.initialize(mat1.GetRows(), mat1.GetColumns());
  auto average_diff = std::abs(diff(0,0));
  auto max_inner_diff = std::abs(diff(0, 0));
  size_t diff_cnt{0};
  for(auto i = 0; i < mat1.size(); ++i){
    if( ! almost_equal(mat1[i], mat2[i], max_diff, max_relative_diff)) {
      auto d = std::abs(mat1[i] - mat2[i]);
      max_inner_diff = std::max(max_inner_diff, d);
      average_diff += d;
      diff[i] = T(d);
      diff_cnt++;
      equal = false;
    }
  }
  if ( equal ) {
    return true;
  } else { 
    std::cout << "matrices not exactly equal : " << std::endl << mat1 << std::endl << mat2 << std::endl;
    average_diff = average_diff / diff_cnt;
    std::cout << "max inner diff: " << max_inner_diff << std::endl;
  }
 
  return equal;
}

template <typename T>
bool almost_equal(const std::vector<T>& vec1, const std::vector<T>& vec2) {
  if (vec1.size() != vec2.size()) {
    return false;
  }

  for(auto i = 0; i < vec1.size(); ++i){
    if( ! almost_equal(vec1[i], vec2[i])) {
      return false;
    }
  }
  return true;
}
 
//------------------------------------------------------------------------------
}  // namespace Linalg
//------------------------------------------------------------------------------
}  // end namespace AER
//------------------------------------------------------------------------------
#endif
