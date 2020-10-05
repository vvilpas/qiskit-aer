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

from abc import ABC, abstractmethod
from typing import Callable, Union, List, Optional, Tuple
import numpy as np

from qiskit.quantum_info.operators import Operator

class BaseFrame(ABC):
    """Abstract interface for functionality for entering a constant rotating
    frame specified by an anti-Hermitian matrix F.

    Frames have relevance within the context of differential equations of the
    form :math:`\dot{y}(t) = G(t)y(t)`. "Entering a frame" given by :math:`F`
    corresponds to a change of solution variable :math:`z(t) = e^{-tF}y(t)`.
    Using the definition, we may write down a differential equation for
    :math:`z(t)`:

    .. math:
        \dot{z}(t) = -F z(t) + e^{-tF}G(t)y(t) = (e^{-tF}G(t)e^{tF} - F)z(t)

    In some cases it is computationally easier to solve for :math:`z(t)`
    than it is to solve for :math:`y(t)`.

    While entering a frame is mathematically well-defined for arbitrary
    matrices :math:`F`, we assume in this class that :math:`F` is
    anti-Hermitian, ensuring beneficial properties:
        - :math:`F` is unitarily diagonalizable.
        - :math:`e^{\pm tF}` is easily inverted by taking the adjoint.
        - The frame transformation is norm preserving.
    That :math:`F` is diagonalizable is especially important, as :math:`e^{tF}`
    will need repeated evaluation for different :math:`t` (e.g. at every RHS
    sample point when solving a DE), so it is useful to work in a basis in which
    which :math:`F` is diagonal to minimize the cost of this.

    Given an anti-Hermitian matrix :math:`F`, this class offers functions
    for:
        - Bringing a "state" into/out of the frame:
          :math:`t, y \mapsto e^{\mp tF}y`
        - Bringing an "operator" into/out of the frame:
          :math:`t, A \mapsto e^{\mp tF}Ae^{\pm tF}`
        - Bringing a generator for a BMDE into/out of the frame:
          :math:`t, G \mapsto e^{\mp tF}Ge^{\pm tF} - F`

    It also contains functions for bringing states/operators into/out of
    the basis in which :math:`F` is diagonalized, which we refer to as the
    "frame basis". All previously mentioned functions also include optional
    arguments specifying whether the input/output are meant to be in the
    frame basis. This is to facilitate use in solvers in which working
    completely in the frame basis is beneficial to minimize costs associated
    with evaluation of :math:`e^{tF}`.

    Finally, this class offers support for evaluating linear combinations of
    operators with coefficients with carrier frequencies, along with frequency
    cutoffs for implementing the Rotating Wave Approximation (RWA). Frame
    information and carrier frequency information are intrinsically tied
    together in this context.

    ***This needs to be filled out/explained***
    """

    @property
    @abstractmethod
    def frame_operator(self) -> Union[Operator, np.array]:
        """The original frame operator."""

    @property
    @abstractmethod
    def frame_diag(self) -> np.array:
        """Diagonal of the frame operator as a 1d array."""

    @property
    @abstractmethod
    def frame_basis(self) -> np.array:
        """Array containing diagonalizing unitary."""

    @property
    @abstractmethod
    def frame_basis_adjoint(self) -> np.array:
        """Adjoint of the diagonalizing unitary."""

    def state_into_frame_basis(self, y: np.array) -> np.array:
        """Transform y into the frame basis.

        Args:
            y: the state
        Returns:
            np.array: the state in the frame basis
        """
        return self.frame_basis_adjoint @ y

    def state_out_of_frame_basis(self, y: np.array) -> np.array:
        """Transform y out of the frame basis.

        Args:
            y: the state
        Returns:
            np.array: the state in the frame basis
        """
        return self.frame_basis @ y

    def operator_into_frame_basis(self,
                                  op: Union[Operator, np.array]) -> np.array:
        """Transform operator into frame basis.

        Args:
            op: the operator.
        Returns:
            np.array: the operator in the frame basis
        """

        return self.frame_basis_adjoint @ to_array(op) @ self.frame_basis

    def operator_out_of_frame_basis(self,
                                    op: Union[Operator, np.array]) -> np.array:
        """Transform operator into frame basis.

        Args:
            op: the operator.
        Returns:
            np.array: the operator in the frame basis
        """

        return self.frame_basis @ to_array(op) @ self.frame_basis_adjoint

    def operators_into_frame_basis(self,
                                   operators: Union[List[Operator], np.array]) -> np.array:
        """Given a list of operators, perform a change of basis on all into
        the frame basis, and return as a 3d array.

        Args:
            operators: list of operators
        """

        return np.array([self.operator_into_frame_basis(o) for o in operators])

    def state_into_frame(self,
                         t: float,
                         y: np.array,
                         y_in_frame_basis: Optional[bool] = False,
                         return_in_frame_basis: Optional[bool] = False):
        """Take a state into the frame, i.e. return exp(-tF) @ y.

        Args:
            t: time
            y: state (array of appropriate size)
            y_in_frame_basis: whether or not the array y is already in
                              the basis in which the frame is diagonal
            return_in_frame_basis: whether or not to return the result
                                   in the frame basis
        """

        out = y

        # if not in frame basis convert it
        if not y_in_frame_basis:
            out = self.state_into_frame_basis(out)

        # go into the frame
        out = np.diag(np.exp(- t * self.frame_diag)) @ out

        # if output is requested to not be in the frame basis, convert it
        if not return_in_frame_basis:
            out = self.state_out_of_frame_basis(out)

        return out

    def state_out_of_frame(self,
                           t: float,
                           y: np.array,
                           y_in_frame_basis: Optional[bool] = False,
                           return_in_frame_basis: Optional[bool] = False):
        """Take a state out of the frame, i.e. return exp(tF) @ y.

        Args:
            t: time
            y: state (array of appropriate size)
            y_in_frame_basis: whether or not the array y is already in
                              the basis in which the frame is diagonal
            return_in_frame_basis: whether or not to return the result
                                   in the frame basis
        """
        return self.state_into_frame(-t, y,
                                     y_in_frame_basis,
                                     return_in_frame_basis)

    def operator_into_frame(self,
                            t: float,
                            operator: Union[Operator, np.array],
                            operator_in_frame_basis: Optional[bool] = False,
                            return_in_frame_basis: Optional[bool] = False):
        """Return exp(-Ft) @ operator @ exp(Ft)

        Args:
            t: time
            y: state (array of appropriate size)
            y_in_frame_basis: whether or not the array y is already in
                              the basis in which the frame is diagonal
            return_in_frame_basis: whether or not to return the result
                                   in the frame basis
        """
        return self._conjugate_and_add(t,
                                       operator,
                                       operator_in_frame_basis=operator_in_frame_basis,
                                       return_in_frame_basis=return_in_frame_basis)

    def operator_out_of_frame(self,
                              t: float,
                              operator: Union[Operator, np.array],
                              operator_in_frame_basis: Optional[bool] = False,
                              return_in_frame_basis: Optional[bool] = False):
        """Return exp(-Ft) @ operator @ exp(Ft)

        Args:
            t: time
            y: state (array of appropriate size)
            y_in_frame_basis: whether or not the array y is already in
                              the basis in which the frame is diagonal
            return_in_frame_basis: whether or not to return the result
                                   in the frame basis
        """
        return self.operator_into_frame(-t,
                                        operator,
                                        operator_in_frame_basis=operator_in_frame_basis,
                                        return_in_frame_basis=return_in_frame_basis)


    def generator_into_frame(self,
                             t: float,
                             operator: Union[Operator, np.array],
                             operator_in_frame_basis: Optional[bool] = False,
                             return_in_frame_basis: Optional[bool] = False):
        """Take an generator into the frame, i.e. return
        exp(-tF) @ operator @ exp(tF) - F.

        Args:
            t: time
            y: state (array of appropriate size)
            y_in_frame_basis: whether or not the array y is already in
                              the basis in which the frame is diagonal
            return_in_frame_basis: whether or not to return the result
                                   in the frame basis
        """

        # conjugate and subtract the frame diagonal
        return self._conjugate_and_add(t,
                                       operator,
                                       op_to_add_in_fb=-np.diag(self.frame_diag),
                                       operator_in_frame_basis=operator_in_frame_basis,
                                       return_in_frame_basis=return_in_frame_basis)

    def generator_out_of_frame(self,
                               t: float,
                               operator: Union[Operator, np.array],
                               operator_in_frame_basis: Optional[bool] = False,
                               return_in_frame_basis: Optional[bool] = False):
        """Take an operator out of the frame, i.e. return
        exp(tF) @ operator @ exp(-tF) + F.

        Args:
            t: time
            y: state (array of appropriate size)
            y_in_frame_basis: whether or not the array y is already in
                              the basis in which the frame is diagonal
            return_in_frame_basis: whether or not to return the result
                                   in the frame basis
        """

        # conjugate and add the frame diagonal
        return self._conjugate_and_add(-t,
                                       operator,
                                       op_to_add_in_fb=np.diag(self.frame_diag),
                                       operator_in_frame_basis=operator_in_frame_basis,
                                       return_in_frame_basis=return_in_frame_basis)

    def _conjugate_and_add(self,
                           t: float,
                           operator: np.array,
                           op_to_add_in_fb: Optional[np.array] = None,
                           operator_in_frame_basis: Optional[bool] = False,
                           return_in_frame_basis: Optional[bool] = False):
        """General helper function for computing
            exp(-tF)Gexp(tF) + B

        Note: B is added in the frame basis before any potential final change
        out of the frame basis.
        """

        out = to_array(operator)

        # if not in frame basis convert it
        if not operator_in_frame_basis:
            out = self.operator_into_frame_basis(out)

        # go into the frame
        trans_diag = np.exp(t * self.frame_diag)
        # assumption that F is anti-Hermitian implies conjugation of
        # diagonal gives inversion
        out = np.diag(trans_diag) @ out @ np.diag(trans_diag.conj())

        if op_to_add_in_fb is not None:
            out = out + to_array(op_to_add_in_fb)

        # if output is requested to not be in the frame basis, convert it
        if not return_in_frame_basis:
            out = self.operator_out_of_frame_basis(out)

        return out


    def get_operators_in_frame_basis_with_cutoffs(self,
                                            operators: Union[np.array, List[Operator]],
                                            carrier_freqs: Optional[np.array] = None,
                                            cutoff_freq: Optional[float] = None):
        """First attempt at cleaning up frequency and cutoff Handling
        """

        ops_in_frame_basis = self.operators_into_frame_basis(operators)

        # Handle optional arguments
        if carrier_freqs is None:
            carrier_freqs = np.zeros(len(operators))

        # create difference matrix for diagonal elements
        dim = len(self.frame_diag)
        D_diff = np.ones((dim, dim)) * self.frame_diag
        D_diff = D_diff - D_diff.transpose()

        # if no cutoff freq is specified, there isn't a difference between
        # the two operator returns
        if cutoff_freq is None:
            return D_diff, ops_in_frame_basis, ops_in_frame_basis

        # set up matrix encoding frequencies
        im_angular_freqs = 1j * 2 * np.pi * carrier_freqs
        freq_array = np.array([w + D_diff for w in im_angular_freqs])

        cutoff_array = ((np.abs(freq_array.imag) / (2 * np.pi))
                                < cutoff_freq).astype(int)

        return (D_diff,
                cutoff_array * ops_in_frame_basis,
                cutoff_array.transpose(0, 2, 1) * ops_in_frame_basis)

    def _evaluate_stupid_thing(self,
                               t,
                               coeffs,
                               D_diff,
                               ops_1,
                               ops_2,
                               return_in_frame_basis=False):

        frame_mat = np.exp(t * D_diff)

        ops_1_in_frame = frame_mat * ops_1
        ops_2_in_frame = frame_mat * ops_2

        full_op = (0.5 * (np.tensordot(coeffs, ops_1_in_frame, axes=1)
                         + np.tensordot(coeffs.conj(), ops_2_in_frame, axes=1))
                         - np.diag(self.frame_diag))

        if return_in_frame_basis:
            return full_op
        else:
            return self.operator_out_of_frame_basis(full_op)


    def _get_canonical_freq_arrays(self,
                                   carrier_freqs: np.array,
                                   cutoff_freq: Optional[float] = None) -> Tuple[np.array, Union[None, np.array]]:
        """Get frequency and cutoff arrays in basis in which F is diagonal.
        """
        # create difference matrix for diagonal elements
        dim = len(self.frame_diag)
        D_diff = np.ones((dim, dim)) * self.frame_diag
        D_diff = D_diff - D_diff.transpose()

        # set up matrix encoding frequencies
        im_angular_freqs = 1j * 2 * np.pi * carrier_freqs
        freq_array = np.array([w + D_diff for w in im_angular_freqs])

        # set up cutoff frequency matrix - i.e. same shape as freq_array - with
        # each entry a 1 if the corresponding entry of freq_array has a
        # frequency below the cutoff, and 0 otherwise
        cutoff_array = None
        if cutoff_freq is not None:
            cutoff_array = ((np.abs(freq_array.imag) / (2 * np.pi))
                                    < cutoff_freq).astype(int)

        return freq_array, cutoff_array

    def _evaluate_canonical_operator_combo(self,
                                           t: float,
                                           coefficients: np.array,
                                           operators_in_frame_basis: np.array,
                                           freq_array: np.array,
                                           cutoff_array: Optional[np.array] = None,
                                           return_in_frame_basis: Optional[bool] = False):
        """Evaluate the "canonical" decomposition.

        Explain this!!
        """
        # first evaluate the unconjugated coefficients for each matrix element,
        # given by the coefficient for the full matrix multiplied by the
        # exponentiated frequency term for each entry
        Q = (coefficients[:, np.newaxis, np.newaxis] * np.exp(freq_array * t))

        # apply cutoff if present
        if cutoff_array is not None:
           Q = cutoff_array * Q

        # multiplying the operators by the average of the "unconjugated" and
        # "conjugated" coefficients
        op_list = (0.5 * (Q + Q.conj().transpose(0, 2, 1)) *
                   operators_in_frame_basis)

        # sum the operators and subtract the frame operator
        op_in_frame_basis = np.sum(op_list, axis=0) - np.diag(self.frame_diag)

        if return_in_frame_basis:
            return op_in_frame_basis
        else:
            return self.operator_out_of_frame_basis(op_in_frame_basis)

class Frame(BaseFrame):

    def __init__(self, frame_operator: Union[Operator, np.array]):

        # if None, set to a 1d array of zeros
        if frame_operator is None:
            raise Exception("""frame_operator cannot be None.""")

        self._frame_operator = frame_operator

        # if frame_operator is a 1d array, assume already diagonalized
        if isinstance(frame_operator, np.ndarray) and frame_operator.ndim == 1:

            # verify that it is anti-hermitian (i.e. purely imaginary)
            if np.linalg.norm(frame_operator + frame_operator.conj()) > 1e-10:
                raise Exception("""frame_operator must be an
                                   anti-Hermitian matrix.""")

            self._frame_diag = frame_operator
            self._frame_basis = np.eye(len(frame_operator))
            self._frame_basis_adjoint = self.frame_basis
        # if not, diagonalize it
        else:
            # Ensure that it is an Operator object
            frame_operator = Operator(frame_operator)

            # verify anti-hermitian
            herm_part = frame_operator + frame_operator.adjoint()
            if herm_part != Operator(np.zeros(frame_operator.dim)):
                raise Exception("""frame_operator must be an
                                   anti-Hermitian matrix.""")

            # diagonalize with eigh, utilizing assumption of anti-hermiticity
            frame_diag, frame_basis = np.linalg.eigh(1j * frame_operator.data)

            self._frame_diag = -1j * frame_diag
            self._frame_basis = frame_basis
            self._frame_basis_adjoint = frame_basis.conj().transpose()

    @property
    def frame_operator(self) -> np.array:
        """The original frame operator."""
        return self._frame_operator

    @property
    def frame_diag(self) -> np.array:
        """Diagonal of the frame operator."""
        return self._frame_diag

    @property
    def frame_basis(self) -> np.array:
        """Array containing diagonalizing unitary."""
        return self._frame_basis

    @property
    def frame_basis_adjoint(self) -> np.array:
        """Adjoint of the diagonalizing unitary."""
        return self._frame_basis_adjoint


# type handling
def to_array(op):
    if isinstance(op, Operator):
        return op.data
    return op
