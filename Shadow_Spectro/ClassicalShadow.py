# Written by: Hugo PAGES 
# Date: 2024-01-05

# Standard library imports
from typing import Union

# Third-party imports
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate, RXXGate, RYYGate, RZZGate
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.qasm2 import loads, CustomInstruction
import secrets
class GateConstructor:
    def __init__(self, matrix):
        self.matrix = matrix

    def __call__(self):
        return UnitaryGate(self.matrix)


class ClassicalShadow:
    """
    ClassicalShadow represents a single-shot classical shadow for quantum circuits. 
    It enables the generation of classical shadows through the application of random Clifford gates, measurement, 
    and post-processing to compute expectation values or density matrices.

    Main functionality:
    - addition of random Clifford gates for each qubit in the circuit.
    - Measurement of quantum circuits to obtain bitstring outcomes.
    - Snapshot of classical shadows, including density matrix reconstruction.
    - Post-processing capabilities for calculating expectation values from classical shadows.

    Parameters:
    - noise_error (optional): Specifies noise levels for depolarizing error in the quantum simulator.

    Methods:
    - `get_bit_string(circ)`: Measures a given quantum circuit and returns the resulting bitstring.
    - `random_clifford_gate(idx)`: Returns a random Clifford gate from a predefined set.
    - `add_random_clifford(circuit, copy, backend)`: Adds random Clifford gates to each qubit of a circuit.
    - `snapshot_classical_shadow(circuit, density_matrix)`: Takes a snapshot of the classical shadow, optionally returning the density matrix.
    - `classical_shadow(Quantum_circuit, shadow_size, density_matrix)`: Generates multiple snapshots of the classical shadow, potentially averaging density matrices.
    - `get_expectation_value(obs, unitary_list, measurement_result_list)`: Computes the expectation value of a given Pauli operator from classical shadow data.
    - `snapshot_density_matrix(unitary_list, measurement_result)`: Reconstructs the density matrix from classical shadow data.
    - `deserialize_circuit(qasm_str)`: Deserializes a quantum circuit from a QASM string.
    - `add_clifford_gate_to_qasm(qasm_str)`: Adds random Clifford gates to a QASM string and returns the modified circuit.
    """
    def create_gate_function(self, matrix):
        return GateConstructor(matrix)
    
    def __init__(self, noise_error=None):
        self.err = noise_error
        if isinstance(self.err, (list, np.ndarray)):
            nm = NoiseModel()
            nm.add_all_qubit_quantum_error(depolarizing_error(self.err[0], 1), ["x", "z"])
            nm.add_all_qubit_quantum_error(depolarizing_error(self.err[1], 2), ["rzz", "ryy", "rxx"])
            self.sim = AerSimulator(method="statevector", noise_model=nm)
        else:
            # Choose simulator without noise model if no error is provided
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
        self.Clifford_Gate_set = [
            "III", "XII", "YII", "ZII", "VII", "VXI", "VYI", "VZI",
            "WXI", "WYI", "WZI", "HXI", "HYI", "HZI", "HII",
            "HVI", "HVX", "HVY", "HVZ", "HWI", "HWX",
            "HWY", "HWZ", "WII"]
         # Extracting the gate to apply
        rxx_custom = CustomInstruction(
        name="rxx", num_params=1, num_qubits=2, builtin=False, constructor=RXXGate)
        ryy_custom = CustomInstruction(
            name="ryy", num_params=1, num_qubits=2, builtin=False,  constructor=RYYGate)
        rzz_custom = CustomInstruction(
            name="rzz", num_params=1, num_qubits=2, builtin=False, constructor=RZZGate)
        self.custom_instruction_list=[rxx_custom,ryy_custom,rzz_custom]
        for gate in self.Clifford_Gate_set:
            gate_matrix=self.gate_set[gate[0]]@self.gate_set[gate[1]]@self.gate_set[gate[2]]
            gate_constructor = self.create_gate_function(gate_matrix)
            gate_instruction=CustomInstruction(
                name=str.lower(gate), num_params=0, num_qubits=1, builtin=False, constructor=gate_constructor)
            self.custom_instruction_list.append(gate_instruction)
 
#####################################################################################################################################
                ######################### Classical shadow  #######################
#####################################################################################################################################  
 
            
    def get_bit_string(self, circ: QuantumCircuit) -> str:
        """bit string measurement of a given quantum circuit. shot= 1
        Args:
            circ (QuantumCircuit): quantum circuit to measure
            shots (int, optional): number of shots. Defaults to 1.
        Returns:
            str: bit string 
        """
        try:
            counts = self.sim.run(circ, shots=1).result().get_counts()
        except Exception as e:
            counts = self.sim.run(transpile(circ, self.sim),
                                  shots=1).result().get_counts()
        res = list(counts.keys())[0]  # type: ignore
        del counts
        return res

    def random_clifford_gate(self, idx: int = None) -> UnitaryGate:
        """Get a random clifford gate from the Clifford gate set"""
        if idx is None:
            idx_list = [np.random.randint(0, 23) for _ in range(24)]
            idx=secrets.choice(idx_list)
        return self.Clifford_Gate_set[idx]

        
    def add_random_clifford(self, circuit: QuantumCircuit, copy: bool = False, backend=None) -> tuple[list[UnitaryGate], QuantumCircuit]:
        """Add a random clifford gate for each qubits in a given circuit. add a "measure_all()" instruction after adding the clifford gates.
        Args:
            circuit (QuantumCircuit): circuit to add clifford gates
            copy (bool): If True the gate is added to a copy of the circuit, wich does not modified the given circuit. Default to False
            backend (Qiskit backend): if not None, the return circuit is transpile to the given backend
        ## returns 
        if copy:tuple[list[str], QuantumCircuit]: the list of clifford gates applied to the circuit and the new circuit with the gate added. \n
        if backend not None : tuple[list[str], QuantumCircuit]: the list of clifford gates applied to the circuit and the new circuit transpiled with the gate added.\n
        else : list[str] The clifford gate added to the circuit       
        """
        num_qubits = circuit.num_qubits
        clifford_gates = [None]*num_qubits
        if copy or backend is not None:
            circuit_copy = circuit.copy()
            for qubit in range(num_qubits):
                gate = self.random_clifford_gate()
                clifford_gates[qubit] = gate
                circuit_copy.append(UnitaryGate(np.linalg.multi_dot(
                    [self.gate_set[gate[i]]for i in range(len(gate))])), [qubit])
            circuit_copy.measure_all()
            if backend is not None:
                transpiled_circ = transpile(
                    circuit_copy, backend, optimization_level=2)
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

    def snapshot_classical_shadow(self, circuit: Union[QuantumCircuit, str], density_matrix: bool = False) -> tuple[list[UnitaryGate], str]:
        """ A unique snapshot of the circuit: can handle simple qasm circuit. 
            1. add random clifford to all qubits 
            2. Measure the bitstring
        Args:
            circuit (QuantumCircuit): circuit to add clifford gates
            density matrix (bool): If True return directly a snapshot of the density matrix. Default to False.\n
        Returns : 
            tuple[list[str], str] : The clifford gate added to each qubits and the bit string measurement. \n
            if density matrix True : np.ndarray (2^nq, 2^nq), The density Matrix.
        """
        if isinstance(circuit, str):
            circuit_str,clifford_gates= self.add_clifford_gate_to_qasm(circuit)
            circ_copy=self.deserialize_circuit(circuit_str)
            circ_copy.measure_all()
        else:
            circ_copy=circuit.copy()
            clifford_gates = self.add_random_clifford(circ_copy)
        measurement_result_list = self.get_bit_string(circ_copy)
        if density_matrix:
            return self.snapshot_density_matrix(clifford_gates, measurement_result_list)
        else:
            return clifford_gates, measurement_result_list

    
    def classical_shadow(self,Quantum_circuit: Union[QuantumCircuit, str], shadow_size: int, density_matrix: bool=False):
        """
        Multiple snapshot of classical shadow : 
        Args:
            circuit (QuantumCircuit): circuit to add clifford gates.
            shadow_size (int): the shadow size 
            density matrix (bool): If True return the average density matrix over each snapshot. Default to False 
        """
        if density_matrix :
            density_matrix=sum(self.snapshot_classical_shadow(Quantum_circuit, density_matrix=True) for _ in range(shadow_size))
            return density_matrix/shadow_size
        snapshots_Clifford, snapshots_bits_string = zip(*[self.snapshot_classical_shadow(Quantum_circuit) for _ in range(shadow_size)]                                                        )
        return list(snapshots_Clifford), list(snapshots_bits_string)

    



#####################################################################################################################################
    ######################### Post processing : get_expectation value or get_density_matrix #######################
#####################################################################################################################################  
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

#####################################################################################################################################
            ######################### Qasm Methods : serialize, deserialize  #######################
#####################################################################################################################################  
    def deserialize_circuit(self,qasm_str):
    # VERY UNSTABLE, Lot of gates not recognize->prefere clifford circuit without swap gate
        """Deserialize a QuantumCircuit from JSON."""
        from qiskit.qasm2 import loads
        return loads(qasm_str, custom_instructions=self.custom_instruction_list)

    def add_clifford_gate_to_qasm(self, qasm_str: str) -> str:
        import re
        
        qreg_pattern = r'qreg\s+([a-zA-Z0-9_]+)\[(\d+)\];'
        qreg_matches = re.findall(qreg_pattern, qasm_str)
        total_qubits = sum(int(size) for _, size in qreg_matches)
        clifford_gate_def="""
        // Define v and w
            gate v a { h a; s a; h a; s a; }
            gate w a { v a; v a; }

            // Identity gate
            gate iii a { id a; }

            // Single gates
            gate xii a { x a; }
            gate yii a { y a; }
            gate zii a { z a; }
            gate vii a { v a; }
            gate wii a { w a; }
            gate hii a { h a; }

            // V combinations
            gate vxi a { v a; x a; }
            gate vyi a { v a; y a; }
            gate vzi a { v a; z a; }

            // W combinations
            gate wxi a { w a; x a; }
            gate wyi a { w a; y a; }
            gate wzi a { w a; z a; }

            // H combinations
            gate hxi a { h a; x a; }
            gate hyi a { h a; y a; }
            gate hzi a { h a; z a; }

            // H+V combinations
            gate hvi a { h a; v a; }
            gate hvx a { h a; v a; x a; }
            gate hvy a { h a; v a; y a; }
            gate hvz a { h a; v a; z a; }

            // H+W combinations
            gate hwi a { h a; w a; }
            gate hwx a { h a; w a; x a; }
            gate hwy a { h a; w a; y a; }
            gate hwz a { h a; w a; z a; }"""  
        qasm_str=re.sub(r"(qreg q\[\d+\];\s*)", r"\1" + clifford_gate_def+ "\n", qasm_str)
        Clifford_gate_list=[]
        for n in range(total_qubits):
            gate = self.random_clifford_gate()
            
            # Prepare the gate in QASM format
            qasm_gate = ""
            qasm_gate = f"{str.lower(gate)} q[{n}];"
            
            # Add the new gate to the QASM string
            qasm_str +=qasm_gate
            Clifford_gate_list.append(gate)

        # print(qasm_str)
        return qasm_str, Clifford_gate_list
    
 