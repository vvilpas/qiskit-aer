# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
from typing import Union, List, Optional

from qiskit.quantum_info.operators import Operator
from qiskit.providers.aer.pulse_new.models.operator_models import BaseOperatorModel
from qiskit.providers.aer.pulse_new.de.type_utils import StateTypeConverter

class BMDE_Problem:
    """Class for representing Bilinear Matrix Differential Equations.

    Generator is in the form of a BaseOperatorModel
    """

    def __init__(self,
                 generator: BaseOperatorModel,
                 y0: Optional[np.ndarray] = None,
                 t0: Optional[float] = None,
                 interval: Optional[List[float]] = None,
                 frame_operator: Optional[Union[str, Operator, np.ndarray]] = 'auto',
                 cutoff_freq: Optional[float] = None,
                 state_type_converter: Optional[StateTypeConverter] = None):
        """fill in
        """

        # set state and time parameters
        self.y0 = y0

        if (interval is not None) and (t0 is not None):
            raise Exception('Specify only one of t0 or interval.')

        self.interval = interval
        if interval is not None:
            self.t0 = self.interval[0]
        else:
            self.t0 = t0

        self._state_type_converter = state_type_converter

        # copy the generator
        self._generator = generator.copy()

        # set up frame
        if self._generator.frame_operator is not None:
            # if the generator has a frame specified, leave it as
            self._user_in_frame = True
        else:
            # if auto, go into the drift part of the generator, otherwise
            # set it to whatever as passed
            if frame_operator == 'auto':
                self._generator.frame_operator = anti_herm_part(generator.drift)
            else:
                self._generator.frame_operator = frame_operator

            self._user_in_frame = False

        # set up cutoff frequency
        if self._generator.cutoff_freq is not None and cutoff_freq is not None:
            raise Exception("""Cutoff frequency specified in generator and in
                                solver settings.""")

        if cutoff_freq is not None:
            self._generator.cutoff_freq = cutoff_freq


def anti_herm_part(A: Union[np.ndarray, Operator]):
    """Get the anti-hermitian part of an operator.
    """
    return 0.5 * (A - A.conj().transpose())
