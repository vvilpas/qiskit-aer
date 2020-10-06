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
"""tests for frame.py"""

import unittest
import numpy as np
from scipy.linalg import expm
from qiskit.providers.aer.pulse_new.models.frame import Frame
from qiskit.quantum_info.operators import Operator

class TestFrame(unittest.TestCase):
    """Tests for Frame."""

    def setUp(self):
        self.X = Operator.from_label('X')
        self.Y = Operator.from_label('Y')
        self.Z = Operator.from_label('Z')

    def test_instantiation_errors(self):
        """Check different modes of error raising for frame setting."""

        # 1d array
        try:
            frame = Frame(np.array([1., 1.]))
        except Exception as e:
            self.assertTrue('anti-Hermitian' in str(e))

        # 2d array
        try:
            frame = Frame(np.array([[1., 0.], [0., 1.]]))
        except Exception as e:
            self.assertTrue('anti-Hermitian' in str(e))

        # Operator
        try:
            frame = Frame(self.Z)
        except Exception as e:
            self.assertTrue('anti-Hermitian' in str(e))

    def test_state_transformations_no_frame(self):
        """Test frame transformations with no frame."""

        frame = Frame(np.zeros(2))

        t = 0.123
        y = np.array([1., 1j])
        out = frame.state_into_frame(t, y)
        self.assertAlmostEqual(out, y)
        out = frame.state_out_of_frame(t, y)
        self.assertAlmostEqual(out, y)

        t = 100.12498
        y = np.eye(2)
        out = frame.state_into_frame(t, y)
        self.assertAlmostEqual(out, y)
        out = frame.state_out_of_frame(t, y)
        self.assertAlmostEqual(out, y)

    def test_state_into_frame(self):
        """Test state_into_frame with a non-trival frame."""
        frame_op = -1j * np.pi * (self.X + 0.1 * self.Y + 12. * self.Z).data
        frame = Frame(frame_op)

        evals, U = np.linalg.eigh(1j * frame_op)
        evals = -1j * evals
        Uadj = U.conj().transpose()

        t = 1312.132
        y0 = np.array([[1., 2.], [3., 4.]])

        # compute frame rotation to enter frame
        emFt = expm(-frame_op * t)

        # test without optional parameters
        value = frame.state_into_frame(t, y0)
        expected = emFt @ y0
        # these tests need reduced absolute tolerance due to the time value
        self.assertAlmostEqual(value, expected, tol=1e-10)

        # test with y0 assumed in frame basis
        value = frame.state_into_frame(t, y0, y_in_frame_basis=True)
        expected = emFt @ U @ y0
        self.assertAlmostEqual(value, expected, tol=1e-10)

        # test with output request in frame basis
        value = frame.state_into_frame(t, y0, return_in_frame_basis=True)
        expected = Uadj @ emFt @ y0
        self.assertAlmostEqual(value, expected, tol=1e-10)

        # test with both input and output in frame basis
        value = frame.state_into_frame(t, y0,
                                          y_in_frame_basis=True,
                                          return_in_frame_basis=True)
        expected = Uadj @ emFt @ U @ y0
        self.assertAlmostEqual(value, expected, tol=1e-10)

    def test_state_out_of_frame(self):
        """Test state_out_of_frame with a non-trival frame."""
        frame_op = -1j * np.pi * (3.1 * self.X + 1.1 * self.Y +
                                  12. * self.Z).data
        frame = Frame(frame_op)

        evals, U = np.linalg.eigh(1j * frame_op)
        evals = -1j * evals
        Uadj = U.conj().transpose()

        t = 122.132
        y0 = np.array([[1., 2.], [3., 4.]])

        # compute frame rotation to exit frame
        epFt = expm(frame_op * t)

        # test without optional parameters
        value = frame.state_out_of_frame(t, y0)
        expected = epFt @ y0
        # these tests need reduced absolute tolerance due to the time value
        self.assertAlmostEqual(value, expected, tol=1e-10)

        # test with y0 assumed in frame basis
        value = frame.state_out_of_frame(t, y0, y_in_frame_basis=True)
        expected = epFt @ U @ y0
        self.assertAlmostEqual(value, expected, tol=1e-10)

        # test with output request in frame basis
        value = frame.state_out_of_frame(t, y0, return_in_frame_basis=True)
        expected = Uadj @ epFt @ y0
        self.assertAlmostEqual(value, expected, tol=1e-10)

        # test with both input and output in frame basis
        value = frame.state_out_of_frame(t,
                                         y0,
                                         y_in_frame_basis=True,
                                         return_in_frame_basis=True)
        expected = Uadj @ epFt @ U @ y0
        self.assertAlmostEqual(value, expected, tol=1e-10)

    def test_operators_with_cutoff_no_cutoff(self):
        """Test operators_into_frame_basis_with_cutoff and
        evaluate_generator_with_cutoff without a cutoff.
        """

        frame_op = -1j * np.pi * self.X
        operators = [Operator(-1j * np.pi * self.Z), Operator(-1j * self.X / 2)]
        carrier_freqs = np.array([0., 1.])

        frame = Frame(frame_op)
        ops_w_cutoff, ops_w_conj_cutoff = frame.operators_into_frame_basis_with_cutoff(operators,
                                                                                       carrier_freqs=carrier_freqs)

        t = np.pi * 0.02
        coeffs = np.array([1., 1.]) * np.exp(1j* 2 * np.pi * carrier_freqs * t)
        val = frame.evaluate_generator_with_cutoff(t,
                                                   coeffs,
                                                   ops_w_cutoff,
                                                   ops_w_conj_cutoff)
        U = expm(frame_op.data * t)
        U_adj = U.conj().transpose()
        expected = (U_adj @ (-1j * np.pi * self.Z.data +
                             1j * np.pi * self.X.data +
                            -1j * np.cos(2 * np.pi * t) * self.X.data / 2) @ U)

        self.assertAlmostEqual(val, expected)

        # with complex envelope
        t = np.pi * 0.02
        coeffs = np.array([1., 1. + 2 * 1j]) * np.exp(1j * 2 * np.pi * carrier_freqs * t)
        val = frame.evaluate_generator_with_cutoff(t,
                                                   coeffs,
                                                   ops_w_cutoff,
                                                   ops_w_conj_cutoff)
        U = expm(frame_op.data * t)
        U_adj = U.conj().transpose()
        expected = (U_adj @ (-1j * np.pi * self.Z.data +
                             1j * np.pi * self.X.data +
                            -1j * np.cos(2 * np.pi * t) * self.X.data / 2 +
                            1j * 2 * np.sin(2 * np.pi * t) * self.X.data / 2) @ U)

        self.assertAlmostEqual(val, expected)

    def test_operators_with_cutoff_diag_frame_no_cutoff(self):
        """Test operators_into_frame_basis_with_cutoff and
        evaluate_generator_with_cutoff without a cutoff, with a frame specified
        as a 1d array.
        """

        frame_op = -1j * np.pi * np.array([1., -1.])
        operators = [Operator(-1j * np.pi * self.Z), Operator(-1j * self.X / 2)]
        carrier_freqs = np.array([0., 1.])

        frame = Frame(frame_op)
        ops_w_cutoff, ops_w_conj_cutoff = frame.operators_into_frame_basis_with_cutoff(operators,
                                                                                       carrier_freqs=carrier_freqs)

        t = np.pi * 0.02
        coeffs = np.array([1., 1.]) * np.exp(1j* 2 * np.pi * carrier_freqs * t)
        val = frame.evaluate_generator_with_cutoff(t,
                                                   coeffs,
                                                   ops_w_cutoff,
                                                   ops_w_conj_cutoff)
        U = np.diag(np.exp(frame_op * t))
        U_adj = U.conj().transpose()
        expected = -1j * np.cos(2 * np.pi * t) * U_adj @ self.X.data @ U / 2

        self.assertAlmostEqual(val, expected)

        # with complex envelope
        t = np.pi * 0.02
        coeffs = np.array([1., 1. + 2 * 1j]) * np.exp(1j* 2 * np.pi * carrier_freqs * t)
        val = frame.evaluate_generator_with_cutoff(t,
                                                   coeffs,
                                                   ops_w_cutoff,
                                                   ops_w_conj_cutoff)
        U = np.diag(np.exp(frame_op * t))
        U_adj = U.conj().transpose()
        expected = -1j * (np.cos(2 * np.pi * t) * U_adj @ self.X.data @ U -
                    2 * np.sin(2 * np.pi * t) * U_adj @ self.X.data @ U ) / 2

        self.assertAlmostEqual(val, expected)

    def test_operators_with_cutoff_no_frame_no_cutoff(self):
        """Test operators_into_frame_basis_with_cutoff and
        evaluate_generator_with_cutoff without a cutoff, and without zero frame
        """

        operators = [self.X, self.Y, self.Z]
        carrier_freqs = np.array([1., 2., 3.])

        frame = Frame(np.zeros(2))
        # set up parameters
        ops_w_cutoff, ops_w_conj_cutoff = frame.operators_into_frame_basis_with_cutoff(operators,
                                                                                       carrier_freqs=carrier_freqs)


        t = 0.123
        coeffs = np.array([1., 1j, 1 + 1j]) * np.exp(1j* 2 * np.pi * carrier_freqs * t)
        val = frame.evaluate_generator_with_cutoff(t,
                                                   coeffs,
                                                   ops_w_cutoff,
                                                   ops_w_conj_cutoff)
        ops_as_arrays = np.array([op.data for op in operators])
        expected_out = np.tensordot(np.real(coeffs), ops_as_arrays, axes=1)
        self.assertAlmostEqual(val, expected_out)

        t = 0.123 * np.pi
        coeffs = np.array([4.131, 3.23, 2.1 + 3.1j]) * np.exp(1j* 2 * np.pi * carrier_freqs * t)
        val = frame.evaluate_generator_with_cutoff(t,
                                                   coeffs,
                                                   ops_w_cutoff,
                                                   ops_w_conj_cutoff)
        ops_as_arrays = np.array([op.data for op in operators])
        expected_out = np.tensordot(np.real(coeffs), ops_as_arrays, axes=1)

        self.assertAlmostEqual(val, expected_out)


    def test_operators_into_frame_basis_with_cutoff_no_cutoff(self):
        """Test function for construction of operators with cutoff,
        without a cutoff.
        """

        # no cutoff with already diagonal frame
        frame_op = -1j * np.pi * np.array([1., -1.])
        operators = np.array([self.X.data, self.Y.data, self.Z.data])
        carrier_freqs = np.array([1., 2., 3.])

        frame = Frame(frame_op)

        ops_w_cutoff, ops_w_conj_cutoff = frame.operators_into_frame_basis_with_cutoff(operators,
                                                                                       carrier_freqs=carrier_freqs)

        self.assertAlmostEqual(ops_w_cutoff, operators)
        self.assertAlmostEqual(ops_w_conj_cutoff, operators)

        # same test but with frame given as a 2d array
        # in this case diagonalization will occur, causing eigenvalues to
        # be sorted in ascending order. This will flip Y and Z
        frame_op = -1j * np.pi * np.array([[1., 0], [0, -1.]])
        carrier_freqs = np.array([1., 2., 3.])

        frame = Frame(frame_op)

        ops_w_cutoff, ops_w_conj_cutoff = frame.operators_into_frame_basis_with_cutoff(operators,
                                                                                       carrier_freqs=carrier_freqs)


        expected_ops = np.array([self.X.data, -self.Y.data, -self.Z.data])

        self.assertAlmostEqual(ops_w_cutoff, expected_ops)
        self.assertAlmostEqual(ops_w_conj_cutoff, expected_ops)

    def test_operators_into_frame_basis_with_cutoff(self):
        """Test function for construction of operators with cutoff."""

        # cutoff test
        frame_op = -1j * np.pi * np.array([1., -1.])
        operators = np.array([self.X.data, self.Y.data, self.Z.data])
        carrier_freqs = np.array([1., 2., 3.])
        cutoff_freq = 3.

        frame = Frame(frame_op)

        cutoff_mat = np.array([[[1, 1],
                                [1, 1]],
                               [[1, 0],
                                [1, 1]],
                               [[0, 0],
                                [1, 0]]
                               ])

        ops_w_cutoff, ops_w_conj_cutoff = frame.operators_into_frame_basis_with_cutoff(operators,
                                                                                       cutoff_freq=cutoff_freq,
                                                                                       carrier_freqs=carrier_freqs)

        ops_w_cutoff_expect = cutoff_mat * operators
        ops_w_conj_cutoff_expect = cutoff_mat.transpose([0, 2, 1]) * operators

        self.assertAlmostEqual(ops_w_cutoff, ops_w_cutoff_expect)
        self.assertAlmostEqual(ops_w_conj_cutoff, ops_w_conj_cutoff_expect)

        # same test with lower cutoff
        cutoff_freq = 2.

        cutoff_mat = np.array([[[1, 0],
                                [1, 1]],
                               [[0, 0],
                                [1, 0]],
                               [[0, 0],
                                [0, 0]]
                               ])

        ops_w_cutoff, ops_w_conj_cutoff = frame.operators_into_frame_basis_with_cutoff(operators,
                                                                                       cutoff_freq=cutoff_freq,
                                                                                       carrier_freqs=carrier_freqs)

        ops_w_cutoff_expect = cutoff_mat * operators
        ops_w_conj_cutoff_expect = cutoff_mat.transpose([0, 2, 1]) * operators

        self.assertAlmostEqual(ops_w_cutoff, ops_w_cutoff_expect)
        self.assertAlmostEqual(ops_w_conj_cutoff, ops_w_conj_cutoff_expect)
        

    def assertAlmostEqual(self, A, B, tol=1e-14):
        diff = np.abs(A - B).max()
        self.assertTrue(diff < tol)
