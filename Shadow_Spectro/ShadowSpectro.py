import numpy as np
import itertools
from functools import reduce
from operator import concat
from qiskit.circuit.library import UnitaryGate
from tqdm import tqdm
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from typing import Union
from numba import jit

class ShadowSpectro:
    def __init__(self, shadow, spectro, nq: int, k: int, shadow_size: int) -> None:
        """Class constructor for Shadow spectroscopy

        Args:
            shadow (ClassicalShadow): class for classical shadow
            spectro (Spectroscopy): class for spectroscopy
            nq (int): number of qubits
            k (int): set the observable as all the k-Pauli observable
        """

        self.shadow = shadow
        self.nq = nq
        self.k = k
        self.spectro = spectro
        self.q_Pauli = self.q_local_shadow_observable(self.k)
        self.No = len(self.q_Pauli)
        self.C = np.ndarray
        self.shadow_size = shadow_size
        try:
            self.Nt = spectro.Nt
            self.dt = spectro.dt
        except:
            Warning("spectroscopy class have no attributs")
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


    def expectation_value_q_Pauli(self, U_clifford_shadow: list[list[str]], bit_string_shadow: list[str], density_matrix=None) -> np.ndarray:
        """Get the expectation value of all k-Pauli Observable from a list of classical snapshot, CliffordGate and bit_string. 
           The length of bit_string_shadow and U_clifford_shadow is equal  of the number of snapshot for classical shadow.
           The length of list[UnitaryGate] and bit_string (str) is equal to the number of qubits in the system. 
           The function calculate the expectation value average of all snapshot for each k-Pauli Observable.
           Return a 1d array with the expectation value of all k-Pauli Observable.
           The array is standardize as [array-mean(array)]/std(array).

        Args:
            U_clifford_shadow (list[list[str]]): Clifford gate applied to each snapshot for each Qubits. 
            bit_string_shadow (list[str]): bit string measurement for each snapshot.

        Returns:
            np.ndarray: Expectation value of each Observable
        """
        fk = np.zeros(self.No, dtype=np.complex128)  # Preallocate memory
        if isinstance(density_matrix, (np.ndarray, list)):
            for n, obs in enumerate(self.q_Pauli):               
                pauli_matrices =  reduce(np.kron, [self.gate_set[obs[i]] for i in range(self.nq)])
                fk[n] += np.trace(density_matrix @ pauli_matrices)
        else:
            fk[:] = [
                np.mean([
                    self.shadow.get_expectation_value(observable, U, b)
                    for U, b in zip(U_clifford_shadow, bit_string_shadow)
                ])
                for observable in self.q_Pauli
            ]
        return fk.real

    def q_local_shadow_observable(self, K: int) -> list[str]:
        """Generate the sequence of all the observable from 1-Pauli observable to K-Pauli observable

        Args:
            K (int): K-pauli observable to generate
        Returns:
            list[str]: list of all the observable from 1-Pauli observable to K-Pauli observable
        """
        q_local = []
        for k in range(K):
            q_local.append(self.q_local_Pauli(k+1))
        return reduce(concat, q_local)

    def q_local_Pauli(self, k: int) -> list[str]:
        """Generate the sequence of all the k-Pauli observable

        Args:
            k (int):  K-pauli observable to generate

        Returns:
            list[str]:  list of all the k-Pauli observable
        """
        pauli_operators = ["X", "Y", "Z",]
        q_local = []
        all_combinations = list(itertools.product(pauli_operators, repeat=k))
        for positions in itertools.combinations(range(self.nq), k):
            for combination in all_combinations:
                observable = ['I'] * self.nq
                for i, pos in enumerate(positions):
                    observable[pos] = combination[i]
                q_local.append(tuple(observable))
        return q_local

    def get_snapshots_classical_shadow(self, Quantum_circuit: QuantumCircuit) -> tuple[list[list[UnitaryGate]], list[str]]:
        """Get the snapshots of classical shadow on a given quantum circuit: 
        ### Step 1:
            Add a random clifford gate to the circuit.
        ### Step 2:
            get the bit string from the circuit measurement

        Args:
           Quantum_circuit (QuantumCircuit) : Quantum circuit to do the classical shadow 
        Returns:
            tuple[list[list[UnitaryGate]], list[str]] : snapshots_Clifford, snapshots_bits_string : list of all the snapshot from classical shadow. 
        """

        snapshots_Clifford, snapshots_bits_string = zip(*[self.shadow.classical_shadow(Quantum_circuit) for _ in range(self.shadow_size)]
                                                        )
        return list(snapshots_Clifford), list(snapshots_bits_string)

    def correlation_matrix(self, D: np.ndarray) -> np.ndarray:
        """Calculate the normalize correlation matrix of X as C=(X.Xt)/No

        Args:
            X (np.ndarray): matrix to get the correlation matrix

        Returns:
            np.ndarray: correlation matrix
        """
        Dt = np.transpose(D)
        C = (D@Dt)/self.No
        C = np.array(C)
        self.C = C
        return C

    def shadow_spectro(self, hamil, init_state: Union[np.ndarray, list, QuantumCircuit] = None, Trotter_steps: int = 1000, density_matrix: bool = False, verbose: bool = True, Data_Matrix:bool=False) -> tuple[float, float, np.ndarray, np.ndarray]:
        D = np.zeros((self.Nt, self.No))
        T = np.linspace(0.00, self.Nt * self.dt, self.Nt)
        is_unitary = isinstance(
            hamil(1), UnitaryGate) if callable(hamil) else False

        for n, t in tqdm(enumerate(T), desc="Time evolution", disable=not verbose):
            if is_unitary:
                circ = QuantumCircuit(self.nq)
                if isinstance(init_state, (np.ndarray, list)):
                    circ.initialize(init_state, normalize=True)
                circ.append(hamil(t), range(self.nq))
                C = init_state.compose(circ) if isinstance(
                    init_state, QuantumCircuit) else circ
            else:
                C = hamil.gen_quantum_circuit(
                    t, init_state=init_state, trotter_step=Trotter_steps)

            if density_matrix:
                Rho = sum(self.shadow.classical_shadow(C, density_matrix=True)
                          for _ in range(self.shadow_size)
                          ) / self.shadow_size

                fkt = self.expectation_value_q_Pauli(
                    None, None, density_matrix=Rho)
            else:
                snapshots_clifford, snapshots_bit_string = self.get_snapshots_classical_shadow(
                    C)
                fkt = self.expectation_value_q_Pauli(
                    snapshots_clifford, snapshots_bit_string)

            D[n][:] = fkt.tolist()

        if Data_Matrix:
            return D    
        solution, frequencies = self.spectro.Spectroscopy(D)

        return solution, frequencies


##############################################################################################################################
