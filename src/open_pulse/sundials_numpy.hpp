#ifndef QASM_SIMULATOR_SUNDIALS_NUMPY_HPP
#define QASM_SIMULATOR_SUNDIALS_NUMPY_HPP

#include "ode/sundials_wrapper/sundials_complex_vector.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>

namespace AER {
  using n_array = py::array_t<complex_t>;
  using numpy_wrapper = pybind11::detail::unchecked_mutable_reference<complex_t, 1>;

  template<>
  struct SundialsComplexContent<n_array> {
    n_array data;
    complex_t *data_raw;

    static N_Vector new_vector(int vec_length) {
      return SundialsOps<SundialsComplexContent>::SundialsComplexContent_New(vec_length);
    };

    static SundialsComplexContent *get_content(N_Vector v) {
      return static_cast<SundialsComplexContent *>(v->content);
    }

    static complex_t *&get_raw_data(N_Vector v) {
      return get_content(v)->data_raw;
    }

    static n_array &get_data(N_Vector v) {
      return get_content(v)->data;
    }

    static sunindextype get_size(N_Vector v) {
      return get_data(v).size();
    }

    static N_Vector new_vector(const py::array_t <complex_t> &container) {
      N_Vector y = SundialsOps<SundialsComplexContent>::SundialsComplexContent_New(container.size());
      auto &data = get_data(y);
      data = n_array({static_cast<int>(container.size()), 1});
//      data = container;
      auto data_raw = get_raw_data(y);
  //    data_raw = static_cast<complex_t *>(data.request().ptr);
      auto con_raw = static_cast<complex_t *>(container.request().ptr);
      for(int i = 0; i<container.size(); i++){
        data_raw[i] = con_raw[i];
      }
      return y;
    }

    static void prepare_data(N_Vector v, int length) {
      auto content = static_cast<SundialsComplexContent *>(v->content);
      content->data = n_array({length, 1});
      content->data_raw = static_cast<complex_t *>(content->data.request().ptr);
    }

    static void set_data(N_Vector v, const n_array& y0) {
      auto &raw_y = SundialsComplexContent::get_data(v);
      raw_y = y0;
    }
  };
}

#endif //QASM_SIMULATOR_SUNDIALS_NUMPY_HPP
