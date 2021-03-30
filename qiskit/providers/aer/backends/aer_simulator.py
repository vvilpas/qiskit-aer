# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019, 2021
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Qiskit Aer qasm simulator backend.
"""

import copy
import logging
from qiskit.providers.options import Options
from qiskit.providers.models import QasmBackendConfiguration

from ..version import __version__
from .aerbackend import AerBackend, AerError
from .backend_utils import (cpp_execute, available_methods,
                            available_devices,
                            MAX_QUBITS_STATEVECTOR)
# pylint: disable=import-error, no-name-in-module
from .controller_wrappers import aer_controller_execute

logger = logging.getLogger(__name__)


class AerSimulator(AerBackend):
    """
    Noisy quantum circuit simulator backend.

    **Configurable Options**

    The `AerSimulator` supports multiple simulation methods and
    configurable options for each simulation method. These may be set using the
    appropriate kwargs during initialization. They can also be set of updated
    using the :meth:`set_options` method.

    Run-time options may also be specified as kwargs using the :meth:`run` method.
    These will not be stored in the backend and will only apply to that execution.
    They will also override any previously set options.

    For example, to configure a density matrix simulator with a custom noise
    model to use for every execution

    .. code-block:: python

        noise_model = NoiseModel.from_backend(backend)
        backend = AerSimulator(method='density_matrix',
                                noise_model=noise_model)

    **Simulating an IBMQ Backend**

    The simulator can be automatically configured to mimic an IBMQ backend using
    the :meth:`from_backend` method. This will configure the simulator to use the
    basic device :class:`NoiseModel` for that backend, and the same basis gates
    and coupling map.

    .. code-block:: python

        backend = AerSimulator.from_backend(backend)

    **Returning the Final State**

    The final state of the simulator can be saved to the returned
    ``Result`` object by appending the
    :func:`~qiskit.providers.aer.library.save_state` instruction to a
    quantum circuit. The format of the final state will depend on the
    simulation method used. Additional simulation data may also be saved
    using the other save instructions in :mod:`qiskit.provider.aer.library`.

    **Simulation Method Option**

    The simulation method is set using the ``method`` kwarg. A list supported
    simulation methods can be returned using :meth:`available_methods`, these
    are

    * ``"automatic"``: Default simulation method. Either the "statevector",
      "density_matrix", or "stabilizer" simulation method is selected
      automatically at runtime for each circuit based on the circuit
      instructions, number of qubits, and noise model.

    * ``"statevector"``: A dense statevector simulation that can sample
      measurement outcomes from *ideal* circuits with all measurements at
      end of the circuit. For noisy simulations each shot samples a
      randomly sampled noisy circuit from the noise model. Supports CPU and
      GPU devices.

    * ``"density_matrix"``: A dense density matrix simulation that may
      sample measurement outcomes from *noisy* circuits with all
      measurements at end of the circuit. Supports CPU and GPU devices.

    * ``"stabilizer"``: An efficient Clifford stabilizer state simulator
      that can simulate noisy Clifford circuits if all errors in the noise
      model are also Clifford errors. Supports CPU device only.

    * ``"extended_stabilizer"``: An approximate simulated for Clifford + T
      circuits based on a state decomposition into ranked-stabilizer state.
      The number of terms grows with the number of non-Clifford (T) gates.
      Supports CPU device only.

    * ``"matrix_product_state"``: A tensor-network statevector simulator that
      uses a Matrix Product State (MPS) representation for the state. This
      can be done either with or without truncation of the MPS bond dimensions
      depending on the simulator options. The default behaviour is no
      truncation. Supports CPU device only.

    * ``"unitary"``: A dense unitary matrix simulation of an ideal circuit.
      This simulates the unitary matrix of the circuit itself rather than
      the evolution of an initial quantum state. This method can only
      simulate gates, it does not support measurement, reset, or noise.
      Supports CPU and GPU devices.

    * ``"superop"``: A dense superoperator matrix simulation of an ideal or
      noisy circuit. This simulates the superoperator matrix of the circuit
      itself rather than the evolution of an initial quantum state. This method
      can simulate ideal and noisy gates, and reset, but does not support
      measurement. Supports CPU device only.

    **GPU Simulation**

    By default all simulation methods run on the CPU, however select methods
    also support running on a GPU if qiskit-aer was installed with GPU support
    on a compatible NVidia GPU and CUDA version.

    +--------------------------+---------------+
    | Method                   | GPU Supported |
    +==========================+===============+
    | ``automatic``            | Sometimes     |
    +--------------------------+---------------+
    | ``statevector``          | Yes           |
    +--------------------------+---------------+
    | ``density_matrix``       | Yes           |
    +--------------------------+---------------+
    | ``stabilizer``           | No            |
    +--------------------------+---------------+
    | `"matrix_product_state`` | No            |
    +--------------------------+---------------+
    | ``extended_stabilizer``  | No            |
    +--------------------------+---------------+
    | ``unitary``              | Yes           |
    +--------------------------+---------------+
    | ``superop``              | Yes           |
    +--------------------------+---------------+

    Running a GPU simulation is done using ``device="GPU"`` kwarg during
    initialization or with :meth:`set_options`. The list of supported devices
    for the current system can be returned using :meth:`available_devices`.

    **Additional Backend Options**

    The following simulator specific backend options are supported

    * ``method`` (str): Set the simulation method (Default: ``"automatic"``).

    * ``device`` (str): Set the simulation device (Default: ``"CPU"``).

    * ``precision`` (str): Set the floating point precision for
      certain simulation methods to either ``"single"`` or ``"double"``
      precision (default: ``"double"``).

    * ``zero_threshold`` (double): Sets the threshold for truncating
      small values to zero in the result data (Default: 1e-10).

    * ``validation_threshold`` (double): Sets the threshold for checking
      if initial states are valid (Default: 1e-8).

    * ``max_parallel_threads`` (int): Sets the maximum number of CPU
      cores used by OpenMP for parallelization. If set to 0 the
      maximum will be set to the number of CPU cores (Default: 0).

    * ``max_parallel_experiments`` (int): Sets the maximum number of
      qobj experiments that may be executed in parallel up to the
      max_parallel_threads value. If set to 1 parallel circuit
      execution will be disabled. If set to 0 the maximum will be
      automatically set to max_parallel_threads (Default: 1).

    * ``max_parallel_shots`` (int): Sets the maximum number of
      shots that may be executed in parallel during each experiment
      execution, up to the max_parallel_threads value. If set to 1
      parallel shot execution will be disabled. If set to 0 the
      maximum will be automatically set to max_parallel_threads.
      Note that this cannot be enabled at the same time as parallel
      experiment execution (Default: 0).

    * ``max_memory_mb`` (int): Sets the maximum size of memory
      to store a state vector. If a state vector needs more, an error
      is thrown. In general, a state vector of n-qubits uses 2^n complex
      values (16 Bytes). If set to 0, the maximum will be automatically
      set to the system memory size (Default: 0).

    * ``optimize_ideal_threshold`` (int): Sets the qubit threshold for
      applying circuit optimization passes on ideal circuits.
      Passes include gate fusion and truncation of unused qubits
      (Default: 5).

    * ``optimize_noise_threshold`` (int): Sets the qubit threshold for
      applying circuit optimization passes on ideal circuits.
      Passes include gate fusion and truncation of unused qubits
      (Default: 12).

    These backend options only apply when using the ``"statevector"``
    simulation method:

    * ``statevector_parallel_threshold`` (int): Sets the threshold that
      the number of qubits must be greater than to enable OpenMP
      parallelization for matrix multiplication during execution of
      an experiment. If parallel circuit or shot execution is enabled
      this will only use unallocated CPU cores up to
      max_parallel_threads. Note that setting this too low can reduce
      performance (Default: 14).

    * ``statevector_sample_measure_opt`` (int): Sets the threshold that
      the number of qubits must be greater than to enable a large
      qubit optimized implementation of measurement sampling. Note
      that setting this two low can reduce performance (Default: 10)

    These backend options only apply when using the ``"stabilizer"``
    simulation method:

    * ``stabilizer_max_snapshot_probabilities`` (int): set the maximum
      qubit number for the
      `~qiskit.providers.aer.extensions.SnapshotProbabilities`
      instruction (Default: 32).

    These backend options only apply when using the ``"extended_stabilizer"``
    simulation method:

    * ``extended_stabilizer_sampling_methid`` (string): Choose how to simulate
      measurements on qubits. The performance of the simulator depends
      significantly on this choice. In the following, let n be the number of
      qubits in the circuit, m the number of qubits measured, and S be the
      number of shots. (Default: resampled_metropolis)

      * ``"metropolis"``: Use a Monte-Carlo method to sample many output
        strings from the simulator at once. To be accurate, this method
        requires that all the possible output strings have a non-zero
        probability. It will give inaccurate results on cases where
        the circuit has many zero-probability outcomes.
        This method has an overall runtime that scales as n^{2} + (S-1)n.

      * ``"resampled_metropolis"``: A variant of the metropolis method,
        where the Monte-Carlo method is reinitialised for every shot. This
        gives better results for circuits where some outcomes have zero
        probability, but will still fail if the output distribution
        is sparse. The overall runtime scales as Sn^{2}.

      * ``"norm_estimation"``: An alternative sampling method using
        random state inner products to estimate outcome probabilites. This
        method requires twice as much memory, and significantly longer
        runtimes, but gives accurate results on circuits with sparse
        output distributions. The overall runtime scales as Sn^{3}m^{3}.

    * ``extended_stabilizer_metropolis_mixing_time`` (int): Set how long the
      monte-carlo method runs before performing measurements. If the
      output distribution is strongly peaked, this can be decreased
      alongside setting extended_stabilizer_disable_measurement_opt
      to True (Default: 5000).

    * ``"extended_stabilizer_approximation_error"`` (double): Set the error
      in the approximation for the extended_stabilizer method. A
      smaller error needs more memory and computational time
      (Default: 0.05).

    * ``extended_stabilizer_norm_estimation_samples`` (int): The default number
      of samples for the norm estimation sampler. The method will use the
      default, or 4m^{2} samples where m is the number of qubits to be
      measured, whichever is larger (Default: 100).

    * ``extended_stabilizer_norm_estimation_repetitions`` (int): The number
      of times to repeat the norm estimation. The median of these reptitions
      is used to estimate and sample output strings (Default: 3).

    * ``extended_stabilizer_parallel_threshold`` (int): Set the minimum
      size of the extended stabilizer decomposition before we enable
      OpenMP parallelization. If parallel circuit or shot execution
      is enabled this will only use unallocated CPU cores up to
      max_parallel_threads (Default: 100).

    * ``extended_stabilizer_probabilities_snapshot_samples`` (int): If using
      the metropolis or resampled_metropolis sampling method, set the number of
      samples used to estimate probabilities in a probabilities snapshot
      (Default: 3000).

    These backend options only apply when using the ``"matrix_product_state"``
    simulation method:

    * ``matrix_product_state_max_bond_dimension`` (int): Sets a limit
      on the number of Schmidt coefficients retained at the end of
      the svd algorithm. Coefficients beyond this limit will be discarded.
      (Default: None, i.e., no limit on the bond dimension).

    * ``matrix_product_state_truncation_threshold`` (double):
      Discard the smallest coefficients for which the sum of
      their squares is smaller than this threshold.
      (Default: 1e-16).

    * ``mps_sample_measure_algorithm`` (str):
      Choose which algorithm to use for ``"sample_measure"``. ``"mps_probabilities"``
      means all state probabilities are computed and measurements are based on them.
      It is more efficient for a large number of shots, small number of qubits and low
      entanglement. ``"mps_apply_measure"`` creates a copy of the mps structure and
      makes a measurement on it. It is more effients for a small number of shots, high
      number of qubits, and low entanglement. If the user does not specify the algorithm,
      a heuristic algorithm is used to select between the two algorithms.
      (Default: "mps_heuristic").

    These backend options apply in circuit optimization passes:

    * ``fusion_enable`` (bool): Enable fusion optimization in circuit
      optimization passes [Default: True]
    * ``fusion_verbose`` (bool): Output gates generated in fusion optimization
      into metadata [Default: False]
    * ``fusion_max_qubit`` (int): Maximum number of qubits for a operation generated
      in a fusion optimization [Default: 5]
    * ``fusion_threshold`` (int): Threshold that number of qubits must be greater
      than or equal to enable fusion optimization [Default: 14]
    """
    # Supported basis gates for each simulation method
    _BASIS_GATES = {
        'statevector': sorted([
            'u1', 'u2', 'u3', 'u', 'p', 'r', 'rx', 'ry', 'rz', 'id', 'x',
            'y', 'z', 'h', 's', 'sdg', 'sx', 't', 'tdg', 'swap', 'cx',
            'cy', 'cz', 'csx', 'cp', 'cu1', 'cu2', 'cu3', 'rxx', 'ryy',
            'rzz', 'rzx', 'ccx', 'cswap', 'mcx', 'mcy', 'mcz', 'mcsx',
            'mcphase', 'mcu1', 'mcu2', 'mcu3', 'mcrx', 'mcry', 'mcrz',
            'mcr', 'mcswap', 'unitary', 'diagonal', 'multiplexer',
            'initialize', 'delay', 'pauli', 'mcx_gray'
        ]),
        'density_matrix': sorted([
            'u1', 'u2', 'u3', 'u', 'p', 'r', 'rx', 'ry', 'rz', 'id', 'x',
            'y', 'z', 'h', 's', 'sdg', 'sx', 't', 'tdg', 'swap', 'cx',
            'cy', 'cz', 'cp', 'cu1', 'rxx', 'ryy', 'rzz', 'rzx', 'ccx',
            'unitary', 'diagonal', 'delay', 'pauli',
        ]),
        'matrix_product_state': sorted([
            'u1', 'u2', 'u3', 'u', 'p', 'cp', 'cx', 'cy', 'cz', 'id', 'x', 'y', 'z', 'h', 's',
            'sdg', 'sx', 't', 'tdg', 'swap', 'ccx', 'unitary', 'roerror', 'delay',
            'r', 'rx', 'ry', 'rz', 'rxx', 'ryy', 'rzz', 'rzx', 'csx', 'cswap', 'diagonal',
            'initialize'
        ]),
        'stabilizer': sorted([
            'id', 'x', 'y', 'z', 'h', 's', 'sdg', 'sx', 'cx', 'cy', 'cz',
            'swap', 'delay',
        ]),
        'extended_stabilizer': sorted([
            'cx', 'cz', 'id', 'x', 'y', 'z', 'h', 's', 'sdg', 'sx',
            'swap', 'u0', 't', 'tdg', 'u1', 'p', 'ccx', 'ccz', 'delay'
        ]),
        'unitary': sorted([
            'u1', 'u2', 'u3', 'u', 'p', 'r', 'rx', 'ry', 'rz', 'id', 'x',
            'y', 'z', 'h', 's', 'sdg', 'sx', 't', 'tdg', 'swap', 'cx',
            'cy', 'cz', 'csx', 'cp', 'cu1', 'cu2', 'cu3', 'rxx', 'ryy',
            'rzz', 'rzx', 'ccx', 'cswap', 'mcx', 'mcy', 'mcz', 'mcsx',
            'mcp', 'mcu1', 'mcu2', 'mcu3', 'mcrx', 'mcry', 'mcrz',
            'mcr', 'mcswap', 'unitary', 'diagonal', 'multiplexer', 'delay', 'pauli',
        ]),
        'superop': sorted([
            'u1', 'u2', 'u3', 'u', 'p', 'r', 'rx', 'ry', 'rz', 'id', 'x',
            'y', 'z', 'h', 's', 'sdg', 'sx', 't', 'tdg', 'swap', 'cx',
            'cy', 'cz', 'cp', 'cu1', 'rxx', 'ryy',
            'rzz', 'rzx', 'ccx', 'unitary', 'diagonal', 'delay',
        ])
    }
    # Automatic method basis gates are the union of statevector,
    # density matrix, and stabilizer methods
    _BASIS_GATES[None] = _BASIS_GATES['automatic'] = sorted(
        set(_BASIS_GATES['statevector']).union(
            _BASIS_GATES['stabilizer']).union(
                _BASIS_GATES['density_matrix']))

    _CUSTOM_INSTR = {
        'statevector': sorted([
            'roerror', 'kraus', 'snapshot', 'save_expval', 'save_expval_var',
            'save_probabilities', 'save_probabilities_dict',
            'save_amplitudes', 'save_amplitudes_sq',
            'save_density_matrix', 'save_state', 'save_statevector',
            'save_statevector_dict', 'set_statevector'
        ]),
        'density_matrix': sorted([
            'roerror', 'kraus', 'superop', 'snapshot',
            'save_state', 'save_expval', 'save_expval_var',
            'save_probabilities', 'save_probabilities_dict',
            'save_density_matrix', 'save_amplitudes_sq',
            'set_density_matrix'
        ]),
        'matrix_product_state': sorted([
            'roerror', 'snapshot', 'kraus', 'save_expval', 'save_expval_var',
            'save_probabilities', 'save_probabilities_dict',
            'save_state', 'save_matrix_product_state', 'save_statevector',
            'save_density_matrix', 'save_amplitudes', 'save_amplitudes_sq'
        ]),
        'stabilizer': sorted([
            'roerror', 'snapshot', 'save_expval', 'save_expval_var',
            'save_probabilities', 'save_probabilities_dict',
            'save_amplitudes_sq', 'save_state', 'save_stabilizer',
            'set_stabilizer'
        ]),
        'extended_stabilizer': sorted([
            'roerror', 'snapshot', 'save_statevector',
            'save_expval', 'save_expval_var'
        ]),
        'unitary': sorted([
            'snapshot', 'save_state', 'save_unitary', 'set_unitary'
        ]),
        'superop': sorted([
            'kraus', 'superop', 'save_state', 'save_superop', 'set_superop'
        ])
    }
    # Automatic method custom instructions are the union of statevector,
    # density matrix, and stabilizer methods
    _CUSTOM_INSTR[None] = _CUSTOM_INSTR['automatic'] = sorted(
        set(_CUSTOM_INSTR['statevector']).union(
            _CUSTOM_INSTR['stabilizer']).union(
                _CUSTOM_INSTR['density_matrix']))

    _DEFAULT_CONFIGURATION = {
        'backend_name': 'aer_simulator',
        'backend_version': __version__,
        'n_qubits': MAX_QUBITS_STATEVECTOR,
        'url': 'https://github.com/Qiskit/qiskit-aer',
        'simulator': True,
        'local': True,
        'conditional': True,
        'open_pulse': False,
        'memory': True,
        'max_shots': int(1e6),
        'description': 'A C++ QasmQobj simulator with noise',
        'coupling_map': None,
        'basis_gates': _BASIS_GATES['automatic'],
        'custom_instructions': _CUSTOM_INSTR['automatic'],
        'gates': []
    }

    _SIMULATION_METHODS = [
        'automatic', 'statevector', 'density_matrix',
        'stabilizer', 'matrix_product_state', 'extended_stabilizer',
        'unitary', 'superop'
    ]

    _AVAILABLE_METHODS = None

    _SIMULATION_DEVICES = ['CPU', 'GPU', 'Thrust']

    _AVAILABLE_DEVICES = None

    def __init__(self,
                 configuration=None,
                 properties=None,
                 provider=None,
                 **backend_options):

        self._controller = aer_controller_execute()

        # Update available methods and devices for class
        if AerSimulator._AVAILABLE_METHODS is None:
            AerSimulator._AVAILABLE_METHODS = available_methods(
                self._controller, AerSimulator._SIMULATION_METHODS)
        if AerSimulator._AVAILABLE_DEVICES is None:
            AerSimulator._AVAILABLE_DEVICES = available_devices(
                self._controller, AerSimulator._SIMULATION_DEVICES)

        # Default configuration
        if configuration is None:
            configuration = QasmBackendConfiguration.from_dict(
                AerSimulator._DEFAULT_CONFIGURATION)

        super().__init__(configuration,
                         properties=properties,
                         available_methods=AerSimulator._AVAILABLE_METHODS,
                         provider=provider,
                         backend_options=backend_options)

    @classmethod
    def _default_options(cls):
        return Options(
            # Global options
            shots=1024,
            method=None,
            device=None,
            precision="double",
            zero_threshold=1e-10,
            validation_threshold=None,
            max_parallel_threads=None,
            max_parallel_experiments=None,
            max_parallel_shots=None,
            max_memory_mb=None,
            optimize_ideal_threshold=5,
            optimize_noise_threshold=12,
            fusion_enable=True,
            fusion_verbose=False,
            fusion_max_qubit=5,
            fusion_threshold=14,
            accept_distributed_results=None,
            blocking_qubits=None,
            memory=None,
            noise_model=None,
            # statevector options
            statevector_parallel_threshold=14,
            statevector_sample_measure_opt=10,
            # stabilizer options
            stabilizer_max_snapshot_probabilities=32,
            # extended stabilizer options
            extended_stabilizer_sampling_method='resampled_metropolis',
            extended_stabilizer_metropolis_mixing_time=5000,
            extended_stabilizer_approximation_error=0.05,
            extended_stabilizer_norm_estimation_samples=100,
            extended_stabilizer_norm_estimation_repitions=3,
            extended_stabilizer_parallel_threshold=100,
            extended_stabilizer_probabilities_snapshot_samples=3000,
            # MPS options
            matrix_product_state_truncation_threshold=1e-16,
            matrix_product_state_max_bond_dimension=None,
            mps_sample_measure_algorithm='mps_heuristic',
            chop_threshold=1e-8,
            mps_parallel_threshold=14,
            mps_omp_threads=1)

    def __repr__(self):
        """String representation of an AerSimulator."""
        display = super().__repr__()
        noise_model = getattr(self.options, 'noise_model', None)
        if noise_model is None or noise_model.is_ideal():
            return display
        pad = ' ' * (len(self.__class__.__name__) + 1)
        return '{}\n{}noise_model={})'.format(display[:-1], pad, repr(noise_model))

    def name(self):
        """Format backend name string for simulator"""
        name = self._configuration.backend_name
        method = getattr(self.options, 'method', None)
        if method not in [None, 'automatic']:
            name += f'_{method}'
        device = getattr(self.options, 'device', None)
        if device not in [None, 'CPU']:
            name += f'_{device}'.lower()
        return name

    @classmethod
    def from_backend(cls, backend, **options):
        """Initialize simulator from backend."""
        # pylint: disable=import-outside-toplevel
        # Avoid cyclic import
        from ..noise.noise_model import NoiseModel

        # Get configuration and properties from backend
        configuration = copy.copy(backend.configuration())
        properties = copy.copy(backend.properties())

        # Customize configuration name
        name = configuration.backend_name
        configuration.backend_name = 'aer_simulator({})'.format(name)

        # Use automatic noise model if none is provided
        if 'noise_model' not in options:
            noise_model = NoiseModel.from_backend(backend)
            if not noise_model.is_ideal():
                options['noise_model'] = noise_model

        # Initialize simulator
        sim = cls(configuration=configuration,
                  properties=properties,
                  **options)
        return sim

    def available_devices(self):
        """Return the available simulation methods."""
        return self._AVAILABLE_DEVICES

    def configuration(self):
        """Return the simulator backend configuration.

        Returns:
            BackendConfiguration: the configuration for the backend.
        """
        # Update basis gates based on custom options, config, method,
        # and noise model
        basis_gates = self._basis_gates()
        method = getattr(self.options, 'method', 'automatic')
        custom_inst = self._CUSTOM_INSTR[method]
        config = super().configuration()
        config.custom_instructions = custom_inst
        config.basis_gates = basis_gates + custom_inst
        # Update simulator name
        config.backend_name = self.name()
        return config

    def _execute(self, qobj):
        """Execute a qobj on the backend.

        Args:
            qobj (QasmQobj): simulator input.

        Returns:
            dict: return a dictionary of results.
        """
        return cpp_execute(self._controller, qobj)

    def set_options(self, **fields):
        out_options = {}
        for key, value in fields.items():
            if key == 'method':
                self._set_method_config(value)
                out_options[key] = value
            elif key == 'device':
                if value is not None and value not in self._AVAILABLE_DEVICES:
                    raise AerError(
                        "Invalid simulation device {}. Available devices"
                        " are: {}".format(value, self._AVAILABLE_DEVICES))
                out_options[key] = value
            elif key == 'custom_instructions':
                self._set_configuration_option(key, value)
            else:
                out_options[key] = value
        super().set_options(**out_options)

    def _validate(self, qobj):
        """Semantic validations of the qobj which cannot be done via schemas.

        Warn if no measure or save instructions in run circuits.
        """
        for experiment in qobj.experiments:
            # If circuit does not contain measurement or save
            # instructions raise a warning
            no_data = True
            for op in experiment.instructions:
                if op.name == "measure" or op.name[:5] == "save_":
                    no_data = False
                    break
            if no_data:
                logger.warning(
                    'No measure or save instruction in circuit "%s": '
                    'results will be empty.',
                    experiment.header.name)

    def _basis_gates(self):
        """Return simualtor basis gates.

        This will be the option value of basis gates if it was set,
        otherwise it will be the intersection of the configuration, noise model
        and method supported basis gates.
        """
        # Use option value for basis gates if set
        if 'basis_gates' in self._options_configuration:
            return self._options_configuration['basis_gates']

        # Set basis gates to be the intersection of config, method, and noise model
        # basis gates
        config_gates = self._configuration.basis_gates
        basis_gates = set(config_gates)

        # Compute intersection with method basis gates
        method = getattr(self._options, 'method', 'automatic')
        method_gates = self._BASIS_GATES[method]
        basis_gates = basis_gates.intersection(method_gates)

        # Compute intersection with noise model basis gates
        noise_model = getattr(self.options, 'noise_model', None)
        if noise_model:
            noise_gates = noise_model.basis_gates
            basis_gates = basis_gates.intersection(noise_gates)
        else:
            noise_gates = None

        if not basis_gates:
            logger.warning(
                "The intersection of configuration basis gates (%s), "
                "simulation method basis gates (%s), and "
                "noise model basis gates (%s) is empty",
                config_gates, method_gates, noise_gates)
        return sorted(basis_gates)

    def _set_method_config(self, method=None):
        """Set non-basis gate options when setting method"""
        super().set_options(method=method)
        # Update configuration description and number of qubits
        if method == 'statevector':
            description = 'A C++ statevector simulator with noise'
            n_qubits = MAX_QUBITS_STATEVECTOR
        elif method == 'density_matrix':
            description = 'A C++ density matrix simulator with noise'
            n_qubits = MAX_QUBITS_STATEVECTOR // 2
        elif method == 'unitary':
            description = 'A C++ unitary matrix simulator'
            n_qubits = MAX_QUBITS_STATEVECTOR // 2
        elif method == 'superop':
            description = 'A C++ superop matrix simulator with noise'
            n_qubits = MAX_QUBITS_STATEVECTOR // 4
        elif method == 'matrix_product_state':
            description = 'A C++ matrix product state simulator with noise'
            n_qubits = 63  # TODO: not sure what to put here?
        elif method == 'stabilizer':
            description = 'A C++ Clifford stabilizer simulator with noise'
            n_qubits = 10000  # TODO: estimate from memory
        elif method == 'extended_stabilizer':
            description = 'A C++ Clifford+T extended stabilizer simulator with noise'
            n_qubits = 63  # TODO: estimate from memory
        else:
            # Clear options to default
            description = None
            n_qubits = None
        self._set_configuration_option('description', description)
        self._set_configuration_option('n_qubits', n_qubits)
