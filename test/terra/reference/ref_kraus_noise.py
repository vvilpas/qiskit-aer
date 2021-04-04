# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
QasmSimulator kraus error NoiseModel integration tests
"""

from test.terra.utils.utils import list2dict

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors.standard_errors import amplitude_damping_error

# Backwards compatibility for Terra <= 0.13
if not hasattr(QuantumCircuit, 'i'):
    QuantumCircuit.i = QuantumCircuit.iden


# ==========================================================================
# Amplitude damping error
# ==========================================================================

def kraus_gate_error_circuits():
    """Kraus gate error noise model circuits"""
    circuits = []

    # Repeated amplitude damping to diagonal state
    qr = QuantumRegister(1, 'qr')
    cr = ClassicalRegister(1, 'cr')
    circuit = QuantumCircuit(qr, cr)
    circuit.x(qr)  # prepare + state
    for _ in range(30):
        # Add noisy identities
        circuit.barrier(qr)
        circuit.i(qr)
    circuit.barrier(qr)
    circuit.measure(qr, cr)
    circuits.append(circuit)

    return circuits

def kraus_gate_error_noise_models():
    """Kraus gate error noise models"""
    noise_models = []

    # Amplitude damping error on "id"
    error = amplitude_damping_error(0.75, 0.25)
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error, 'id')
    noise_models.append(noise_model)

    return noise_models

def kraus_gate_error_noise_models_full():
    """Kraus gate error noise models on many gate types"""

    # Amplitude damping error on "u1", "u2", "u3", "cx"
    error = amplitude_damping_error(0.2)
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error, ['h'])
    noise_model.add_all_qubit_quantum_error(error.tensor(error), ['cp', 'swap'])
    return noise_model

def kraus_gate_error_counts(shots, hex_counts=True):
    """Kraus gate error circuits reference counts"""
    counts_lists = []

    # 100% all-qubit Pauli error on "id" gates
    counts = [3 * shots / 4, shots / 4, 0, 0]
    counts_lists.append(counts)

    # Convert to counts dict
    return [list2dict(i, hex_counts) for i in counts_lists]

def kraus_gate_error_counts_on_QFT(shots, hex_counts=False):
    """Kraus gate error QFT circuits reference counts"""

    # Kraus error on all kinds of gates - the results are highly dependent on the
    # specific circuit and therefore are hardcoded
    counts = {'0x0':370*shots/1000, '0x1':175*shots/1000, '0x2':170*shots/1000,
              '0x3':75*shots/1000,  '0x4':100*shots/1000,  '0x5':45*shots/1000,
              '0x6':45*shots/1000,  '0x7':20*shots/1000}
    return counts
