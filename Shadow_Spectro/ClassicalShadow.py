
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Pauli
import numpy as np
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator

# single shot classical shadow
class ClassicalShadow:
    """Classical Shadow for Quantum State Estimation

    This class implements the concept of classical shadows, which provide an efficient way to estimate properties
    of quantum states using randomized measurements. Instead of working directly with the exponentially large
density matrix, classical shadows rely on random Clifford operations and bitstring measurements.

    Attributes:
        nq (int): Number of qubits in the quantum circuit.

    Methods:
        - get_bit_string(circ, shots=1): Measure a quantum circuit and return a bitstring.
        - random_clifford_gate(idx=None): Generate a specific or random Clifford gate.
        - add_random_clifford(circuit): Add random Clifford gates to a circuit.
        - classical_shadow(circuit): Perform a single classical shadow measurement.
        - get_expectation_value(obs, unitary_list, measurement_result_list): Calculate the expectation value of an observable.
        - snapshot_density_matrix(unitary_list, measurement_result): Reconstruct the density matrix from classical shadow data.
    """

    def __init__(self, nqbits):
        """
        Initialize the ClassicalShadow instance.

        Args:
            nqbits (int): Number of qubits in the quantum system.
        """
        self.nq = nqbits

    def get_bit_string(self, circ: QuantumCircuit, shots=1) -> str:
        """
        Measure a given quantum circuit and return a bitstring.

        Args:
            circ (QuantumCircuit): The quantum circuit to measure.
            shots (int, optional): Number of shots for measurement. Defaults to 1.

        Returns:
            str: Bitstring measurement result.
        """
        sim = AerSimulator(method="statevector")
        try:
            counts = sim.run(circ, shots=shots).result().get_counts()
        except Exception:
            counts = sim.run(transpile(circ, sim), shots=shots).result().get_counts()
        res = list(counts.keys())[0]  # type: ignore
        return res

    def random_clifford_gate(self, idx: int = None) -> UnitaryGate:
        """
        Generate a random or specific Clifford gate.

        Args:
            idx (int, optional): Index specifying which Clifford gate to return. If None, a random gate is returned.

        Returns:
            UnitaryGate: The selected Clifford gate as a UnitaryGate object.
        """
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])
        I = np.array([[1, 0], [0, 1]])
        S = np.array([[1, 0], [0, 1j]])
        H = np.array([[1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), -1 / np.sqrt(2)]])
        V = H @ S @ H @ S
        W = V @ V

        gates = [I, X, Y, Z, V, V @ X, V @ Y, V @ Z, W @ X, W @ Y, W @ Z,
                 H @ X, H @ Y, H @ Z, H, H @ V, H @ V @ X, H @ V @ Y, H @ V @ Z,
                 H @ W, H @ W @ X, H @ W @ Y, H @ W @ Z, W]

        if idx is None:
            idx = np.random.randint(0, len(gates))
        elif idx < 0 or idx >= len(gates):
            raise ValueError("Index out of range. Must be between 0 and 23.")

        return UnitaryGate(gates[idx])

    def add_random_clifford(self, circuit: QuantumCircuit) -> tuple[list[UnitaryGate], QuantumCircuit]:
        """
        Add random Clifford gates to each qubit of a given circuit and append measurement operations.

        Args:
            circuit (QuantumCircuit): Circuit to which Clifford gates are added.

        Returns:
            tuple[list[UnitaryGate], QuantumCircuit]:
                List of Clifford gates applied to the circuit and the updated circuit.
        """
        clifford_gates = []
        for qubit in range(self.nq):
            gate = self.random_clifford_gate()
            clifford_gates.append(gate)
            circuit.append(gate, [qubit])
        circuit.measure_all()
        return clifford_gates, circuit

    def classical_shadow(self, circuit: QuantumCircuit) -> tuple[list[UnitaryGate], str]:
        """
        Perform a single classical shadow measurement on a given quantum circuit.

        Steps:
            1. Add random Clifford gates to the circuit.
            2. Measure the circuit to obtain a bitstring.

        Args:
            circuit (QuantumCircuit): The quantum circuit to measure.

        Returns:
            tuple[list[UnitaryGate], str]:
                List of Clifford gates applied and the measurement bitstring.
        """
        circuit_copy = circuit.copy()  # Avoid modifying the original circuit
        unitary_list, shadow_circuit = self.add_random_clifford(circuit_copy)
        measurement_result = self.get_bit_string(shadow_circuit, shots=1)
        return unitary_list, measurement_result

    def get_expectation_value(self, obs: str, unitary_list: list[UnitaryGate], measurement_result: str) -> float:
        """
        Calculate the expectation value of an observable using classical shadow data.

        Args:
            obs (str): String representation of the Pauli operator, ordered as Pn-1...P1P0.
            unitary_list (list[UnitaryGate]): Clifford gates applied to the circuit, ordered as C0, C1, ..., Cn-1.
            measurement_result (str): Measurement result as a bitstring.

        Returns:
            float: Expectation value of the observable.
        """
        expectation_value = 1
        for n in range(self.nq):
            P = np.array(Pauli(obs[-(1 + n)]).to_matrix())
            U_mat = np.array(unitary_list[n].to_matrix())
            U_mat_dagg = np.conj(U_mat).T

            if int(measurement_result[-(1 + n)]) == 0:
                bitstring_matrix = np.array([[1, 0], [0, 0]])
            else:
                bitstring_matrix = np.array([[0, 0], [0, 1]])

            expectation_value *= np.trace((3 * U_mat_dagg @ bitstring_matrix @ U_mat - np.identity(2)) @ P)

        return expectation_value.real

    def snapshot_density_matrix(self, unitary_list: list[UnitaryGate], measurement_result: str) -> np.ndarray:
        """
        Reconstruct the density matrix from classical shadow data.

        Args:
            unitary_list (list[UnitaryGate]): Clifford gates applied to the circuit.
            measurement_result (str): Measurement result as a bitstring.

        Returns:
            np.ndarray: The reconstructed density matrix.
        """
        rho_m = []
        for n in range(self.nq):
            U_mat = np.array(unitary_list[n].to_matrix())
            U_mat_dagg = np.conj(U_mat).T

            if int(measurement_result[-(1 + n)]) == 0:
                bitstring_matrix = np.array([[1, 0], [0, 0]])
            else:
                bitstring_matrix = np.array([[0, 0], [0, 1]])

            partial_rho = 3 * U_mat_dagg @ bitstring_matrix @ U_mat - np.identity(2)

            if n == 0:
                rho_m.append(partial_rho)
            else:
                rho_m.append(np.kron(rho_m[-1], partial_rho))

        return np.array(rho_m[-1])

        