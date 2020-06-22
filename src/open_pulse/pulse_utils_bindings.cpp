/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019, 2020.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#include "numeric_integrator.hpp"
#include "pulse_utils.hpp"
#include "ode/sundials_wrappers/sundials_cvode_wrapper.hpp"
#include "ode/odeint_wrappers/abm_wrapper.hpp"
#include "ode/ode_factory.hpp"
#include <pybind11/functional.h>

namespace AER{
  template<>
  void transform(std::vector<complex_t>& container_to, const py::array_t<complex_t>& container_from){
    auto container_from_raw = container_from.unchecked<1>();
    for (size_t i = 0; i < container_from.size(); ++i) {
      container_to[i] = container_from_raw[i];
    }
  }
}

namespace{
  using Ode_Wrapper_t = AER::Ode<std::vector<complex_t>>;
  using Cvode_Wrapper_t = AER::CvodeWrapper<std::vector<complex_t>>;
  using ABM_Wrapper_t = AER::ABMWrapper<std::vector<complex_t>>;

  using rhsFuncType = std::function<std::vector<complex_t>(double, const py::array_t<complex_t> &)>;
  using pertFuncType = std::function<void(const py::array_t<double> &)>;

  std::unique_ptr<Ode_Wrapper_t> create_sundials_integrator(double t0,
                                                       py::array_t<complex_t> y0,
                                                            rhsFuncType rhs){
    auto func = [rhs](double t, const std::vector<complex_t>& y, std::vector<complex_t>& y_dot){
      auto capsule = py::capsule(&y, [](void *y) {});
      auto y_np = py::array_t<complex_t>(y.size(), y.data(), capsule);
      y_dot = rhs(t, y_np);
    };
    //return std::unique_ptr<Ode_Wrapper_t>(new Cvode_Wrapper_t(AER::OdeMethod::ADAMS, func, y0, t0));
    return AER::ODE::create_ode<std::vector<complex_t>>("sundials_cvode", AER::OdeMethod::ADAMS, func, y0, t0);
  }

  std::unique_ptr<Ode_Wrapper_t> create_sundials_sens_integrator(double t0,
                                                                 py::array_t<complex_t> y0,
                                                                 rhsFuncType rhs, pertFuncType fp,
                                                                 std::vector<double> p){
    auto func = [rhs](double t, const std::vector<complex_t> &y, std::vector<complex_t> &y_dot) {
      auto capsule = py::capsule(&y, [](void *y) {});
      auto y_np = py::array_t<complex_t>(y.size(), y.data(), capsule);
      y_dot = rhs(t, y_np);
    };

    auto pert_func = [fp](const std::vector<double> &p) {
      auto capsule = py::capsule(&p, [](void *p) {});
      auto p_np = py::array_t<double>(p.size(), p.data(), capsule);
      fp(p_np);
    };

//    return std::unique_ptr<Ode_Wrapper_t>(
//        new Cvode_Wrapper_t(AER::OdeMethod::ADAMS, func, y0, pert_func, p, t0));
    return AER::ODE::create_ode<std::vector<complex_t>>("sundials_cvode", AER::OdeMethod::ADAMS, func, y0, pert_func, p, t0);
  }

  //  ABM_Wrapper_t create_abm_integrator(double t0,
//                                      std::vector<complex_t> y0,
//                                      std::function<std::vector<complex_t>(double, const py::array_t<complex_t>&)> rhs){
//    auto func = [rhs](double t, const std::vector<complex_t>& y, std::vector<complex_t>& y_dot){
//      auto capsule = py::capsule(&y, [](void *y) {});
//      auto y_np = py::array_t<complex_t>(y.size(), y.data(), capsule);
//      y_dot = rhs(t, y_np);
//    };
//    return ABM_Wrapper_t(func, y0, t0);
//  }

}


#include "misc/warnings.hpp"
DISABLE_WARNING_PUSH
#include <pybind11/functional.h>
DISABLE_WARNING_POP

RhsFunctor get_ode_rhs_functor(py::object the_global_data, py::object the_exp,
                               py::object the_system, py::object the_channels, py::object the_reg) {
  return RhsFunctor(the_global_data, the_exp, the_system, the_channels, the_reg);
}

PYBIND11_MODULE(pulse_utils, m) {
    m.doc() = "Utility functions for pulse simulator"; // optional module docstring

    m.def("td_ode_rhs_static", &td_ode_rhs, "Compute rhs for ODE");
    m.def("cy_expect_psi_csr", &expect_psi_csr, "Expected value for a operator");
    m.def("occ_probabilities", &occ_probabilities, "Computes the occupation probabilities of the specifed qubits for the given state");
    m.def("write_shots_memory", &write_shots_memory, "Converts probabilities back into shots");
    m.def("oplist_to_array", &oplist_to_array, "Insert list of complex numbers into numpy complex array");
    m.def("spmv_csr", &spmv_csr, "Sparse matrix, dense vector multiplication.");

    py::class_<Ode_Wrapper_t>(m, "OdeCPPWrapper")
      .def("integrate", [](Ode_Wrapper_t &cvode, double time, py::kwargs kwargs){
        bool step = false;
        if(kwargs && kwargs.contains("step")){
          step = kwargs["step"].cast<bool>();
        }
        return cvode.integrate(time, step);
      })
      .def("successful", &Ode_Wrapper_t::succesful)
      .def_property("t", &Ode_Wrapper_t::get_t, &Ode_Wrapper_t::set_t)
      .def_property("_y", [](const Ode_Wrapper_t &cvode){return py::array(py::cast(cvode.get_solution()));},
          &Ode_Wrapper_t::set_solution)
      .def_property_readonly("y", [](const Ode_Wrapper_t &cvode){return py::array(py::cast(cvode.get_solution()));})
      .def("get_sens", [](Ode_Wrapper_t &cvode, int i){return py::array(py::cast(cvode.get_sens_solution(i)));})
      .def("set_intial_value", &Ode_Wrapper_t::set_intial_value)
      .def("set_tolerances", &Ode_Wrapper_t::set_tolerances)
      .def("set_step_limits", &Ode_Wrapper_t::set_step_limits)
      .def("set_maximum_order", &Ode_Wrapper_t::set_maximum_order)
      .def("set_max_nsteps", &Ode_Wrapper_t::set_max_nsteps);

    py::class_<RhsFunctor>(m, "OdeRhsFunctor")
        .def("__call__", &RhsFunctor::operator());

    m.def("get_ode_rhs_functor", &get_ode_rhs_functor, "Get ode_rhs functor to allow caching of parameters");

    m.def("create_sundials_sens_integrator", &create_sundials_sens_integrator, "");
    m.def("create_sundials_integrator", &create_sundials_integrator, "");
}
