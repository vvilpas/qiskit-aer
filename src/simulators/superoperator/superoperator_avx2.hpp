/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2020.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */



#ifndef _qv_superoperator_avx2_hpp_
#define _qv_superoperator_avx2_hpp_

#include "simulators/statevector/base_avx2.hpp"
#include "superoperator.hpp"

namespace AER {
    namespace QV {

        template <typename data_t = double>
        class SuperoperatorAvx2 : public BaseAvx2<Superoperator<data_t, BaseAvx2<data_t>>, data_t>{
        using BaseAvx2<Superoperator<data_t, BaseAvx2<data_t>>, data_t>::BaseAvx2;
    };

}
}
//------------------------------------------------------------------------------
#endif // end module
