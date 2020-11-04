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

#ifndef _aer_controller_execute_hpp_
#define _aer_controller_execute_hpp_

#include <string>
#include "framework/json.hpp"
#include "misc/hacks.hpp"
#include "framework/results/result.hpp"

//=========================================================================
// Controller Execute interface
//=========================================================================

namespace AER {


void load_openmp_if_needed(const json_t &qobj_js);

template <class controller_t>
Result controller_execute(const json_t &qobj_js) {
  controller_t controller;

  load_openmp_if_needed(qobj_js);

  return controller.execute(qobj_js);
}

void load_openmp_if_needed(const json_t &qobj_js){
    static bool loaded = false;
    // Fix for MacOS and OpenMP library double initialization crash.
    // Issue: https://github.com/Qiskit/qiskit-aer/issues/1
    if (!loaded && JSON::check_key("config", qobj_js)) {
        std::string path;
        JSON::get_value(path, "library_dir", qobj_js["config"]);
        Hacks::maybe_load_openmp(path);
        loaded = true;
    }
}

} // end namespace AER
#endif
