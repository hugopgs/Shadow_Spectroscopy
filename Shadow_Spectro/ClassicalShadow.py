from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Pauli
import random
import numpy as np
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.circuit.library import RXXGate, RYYGate, RZZGate

# single shot classical shadow


class ClassicalShadow:
    """single shot classical shadow
    """

    def __init__(self, noise_error=None):
        self.err = noise_error
        if isinstance(self.err, (list, np.ndarray)):
            nm = NoiseModel()
            nm.add_all_qubit_quantum_error(
                depolarizing_error(self.err[0], 1), ["x", "z"])
            nm.add_all_qubit_quantum_error(
                depolarizing_error(self.err[1], 2), ["rzz", "ryy", "rxx"])
            self.sim = AerSimulator(method="statevector", noise_model=nm)
        else:
            self.sim = AerSimulator(method="statevector")
        self.bitstring_matrix0 = np.array([[1, 0], [0, 0]])
        self.bitstring_matrix1 = np.array([[0, 0], [0, 1]])
        self.X = np.array([[0, 1],  [1, 0]])
        self.Y = np.array([[0, -1j], [1j, 0]])
        self.Z = np.array([[1, 0],  [0, -1]])
        self.I = np.array([[1, 0],  [0, 1]])
        self.S = np.array([[1, 0],  [0, 1j]])
        self.H = np.array([[1/np.sqrt(2), 1/np.sqrt(2)],
                          [1/np.sqrt(2), -1/np.sqrt(2)]])
        self.V = self.H@self.S@self.H@self.S
        self.W = self.V@self.V
        self.gate_set = {"X": self.X, "Y": self.Y, "Z": self.Z, "I": self.I, "S": self.S,
                         "H": self.H, "V": self.V, "W": self.W}

    def get_bit_string(self, circ: QuantumCircuit, shots=1) -> str:
        """bit string measurement of a given quantum circuit.

        Args:
            circ (QuantumCircuit): quantum circuit to measure
            shots (int, optional): number of shots. Defaults to 1.

        Returns:
            str: bit string 
        """
        try:
            counts = self.sim.run(circ, shots=1, ).result().get_counts()
        except Exception as e:
            counts = self.sim.run(transpile(circ, self.sim),
                                  shots=1).result().get_counts()
        res = list(counts.keys())[0]  # type: ignore
        del counts
        return res

    def random_clifford_gate(self, idx: int = None) -> UnitaryGate:
        """Get a random clifford gate from the Clifford gate set"""
        if idx is None:
            idx = np.random.randint(0, 23)
        Clifford_Gate_set = [
            "III", "XII", "YII", "ZII", "VII", "VXI", "VYI", "VZI",
            "WXI", "WYI", "WZI", "HXI", "HYI", "HZI", "HII",
            "HVI", "HVX", "HVY", "HVZ", "HWI", "HWX",
            "HWY", "HWZ", "WII"]
        return Clifford_Gate_set[idx]

    def add_random_clifford(self, circuit: QuantumCircuit, copy: bool = False, backend=None) -> tuple[list[UnitaryGate], QuantumCircuit]:
        """Add a random clifford gate for each qubits in a given circuit. add a "measure_all()" instruction after adding the clifford gates.

        Args:
            circuit (QuantumCircuit): circuit to add clifford gates

        Returns:
            tuple[list[UnitaryGate], QuantumCircuit]: the list of clifford gates applied to the circuit and the new circuit with the gate added
        """
        num_qubits = circuit.num_qubits
        clifford_gates = [None]*num_qubits
        if copy:
            circuit_copy = circuit.copy()
            for qubit in range(num_qubits):
                gate = self.random_clifford_gate()
                clifford_gates[qubit] = gate
                circuit_copy.append(UnitaryGate(np.linalg.multi_dot(
                    [self.gate_set[gate[i]]for i in range(len(gate))])), [qubit])
            circuit_copy.measure_all()
            if backend is not None:
                transpiled_circ = transpile(
                    circuit_copy, backend, optimization_level=1)
                return clifford_gates, transpiled_circ
            else:
                return clifford_gates, circuit_copy
        else:
            for qubit in range(num_qubits):
                gate = self.random_clifford_gate()
                clifford_gates[qubit] = gate
                circuit.append(UnitaryGate(np.linalg.multi_dot(
                    [self.gate_set[gate[i]]for i in range(len(gate))])), [qubit])
            circuit.measure_all()
            return clifford_gates

    def classical_shadow(self, circuit: QuantumCircuit, density_matrix: bool = False) -> tuple[list[UnitaryGate], str]:
        """Do one classical shadow shot on a given quantum circuit: 
        ## Step 1:
            Add a random clifford gate to the circuit.
        ## Step 2:
            get the bit string from the circuit measurement 
        ## Step 3:
            return the list of clifford gate added and the bit string 
        Args:
            circuit (QuantumCircuit): _description_
        Returns:
            tuple[list[UnitaryGate], str]: The UnitaryGate added to each Qubit of the circuit, the bit string from measurement 
        """
        if isinstance(circuit, str):
            circuit = self.__deserialize_circuit(circuit)
            clifford_gates = self.add_random_clifford(circuit)
            measurement_result_list = self.get_bit_string(circuit, shots=1)
            del circuit
        else:
            clifford_gates = self.add_random_clifford(circuit)
            measurement_result_list = self.get_bit_string(circuit, shots=1)
            circuit.remove_final_measurements()
            self.remove_last_single_qubit_gates(circuit)

        if density_matrix:
            return self.snapshot_density_matrix(clifford_gates, measurement_result_list)
        else:
            return clifford_gates, measurement_result_list


    def get_expectation_value(self, obs: str, unitary_list: list[str], measurement_result_list: str) -> float:
        """
        Get the expectation value from classical shadow

        Args:
            obs (str): str of Pauli operator, ordered as Pn-1...P1P0
            unitary_list (list[UnitaryGate]): The clifford gate applied on the circuit ordered as C0, C1...Cn-1
            measurement_result_list (str): The bit string from the measurement

        Returns:
            float: snapshot expectation value from classical shadow
        """
        expectation_value = 1
        for n in range(len(obs)):
            P = self.gate_set[obs[-(1+n)]]
            U_mat = np.linalg.multi_dot(
                [self.gate_set[unitary_list[n][i]]for i in range(len(unitary_list[n]))])
            U_mat_dagg = np.conj(U_mat).T
            if int(measurement_result_list[-(1+n)]) == 0:
                expectation_value *= np.trace((3*U_mat_dagg @
                                              self.bitstring_matrix0 @ U_mat-np.identity(2))@P)
            else:
                expectation_value *= np.trace((3*U_mat_dagg @
                                              self.bitstring_matrix1 @ U_mat-np.identity(2))@P)
        return expectation_value.real

    def snapshot_density_matrix(self, unitary_list: list[str], measurement_result: str) -> np.ndarray:
        """
        Reconstruct the density matrix from classical shadow data.
        Args:
            unitary_list (list[UnitaryGate]): Clifford gates applied to the circuit.
            measurement_result (str): Measurement result as a bitstring.

        Returns:
            np.ndarray: The reconstructed density matrix.
        """
        n_qubits = len(unitary_list)
        identity = np.identity(2)
        bit_matrices = {
            '0': 3 * self.bitstring_matrix0 - identity,
            '1': 3 * self.bitstring_matrix1 - identity
        }
        measurements = list(measurement_result[-n_qubits:])[::-1]
        U_first = np.linalg.multi_dot(
            [self.gate_set[unitary_list[0][i]]for i in range(len(unitary_list[0]))])
        U_first_dag = np.conj(U_first).T
        rho = U_first_dag @ bit_matrices[measurements[0]] @ U_first
        for n in range(1, n_qubits):
            U = np.linalg.multi_dot(
                [self.gate_set[unitary_list[n][i]]for i in range(len(unitary_list[n]))])
            U_dag = np.conj(U).T
            partial_rho = U_dag @ bit_matrices[measurements[n]] @ U
            rho = np.kron(rho, partial_rho)
        return rho

    def remove_last_single_qubit_gates(self, circuit):
        # Initialize a list to track the last gate indices for each qubit
        last_single_qubit_gate_indices = [-1] * circuit.num_qubits

        # Traverse the circuit in reverse to find the last single-qubit gate for each qubit
        for index in range(len(circuit.data) - 1, -1, -1):
            instruction = circuit.data[index]
            op = instruction.operation
            qargs = instruction.qubits

            if len(qargs) == 1:  # Check if it's a single-qubit gate
                # Get index of the qubit in the circuit
                qubit_index = circuit.qubits.index(qargs[0])

                # If this is the first last single-qubit gate encountered, record its position
                if last_single_qubit_gate_indices[qubit_index] == -1:
                    last_single_qubit_gate_indices[qubit_index] = index

        # Filter the data to exclude the last single-qubit gate for each qubit
        new_data = [instruction for i, instruction in enumerate(
            circuit.data) if i not in last_single_qubit_gate_indices]
        circuit.data = new_data

    def __deserialize_circuit(self, qasm_str):
        # VERY UNSTABLE, Lot of gates not recognize->prefere clifford circuit without swap gate
        """Deserialize a QuantumCircuit from JSON."""
        from qiskit.qasm2 import loads, CustomInstruction
        # Define custom instructions for rxx, ryy, and rzz
        rxx_custom = CustomInstruction(
            name="rxx", num_params=1, num_qubits=2, builtin=False, constructor=RXXGate)
        ryy_custom = CustomInstruction(
            name="ryy", num_params=1, num_qubits=2, builtin=False,  constructor=RYYGate)
        rzz_custom = CustomInstruction(
            name="rzz", num_params=1, num_qubits=2, builtin=False, constructor=RZZGate)
        return loads(qasm_str, custom_instructions=[rxx_custom, ryy_custom, rzz_custom])
