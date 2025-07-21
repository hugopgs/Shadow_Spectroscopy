# Written by: Hugo PAGES 
# Date: 2024-01-05

# Standard library imports
import time
import itertools
from functools import reduce
from operator import concat
from typing import Union

# Third-party imports
import numpy as np
from tqdm import tqdm
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
import multiprocessing as mp
import random



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
        if len(self.q_Pauli)>15_000:
            self.q_Pauli=random.sample(self.q_Pauli,15000)
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
        self.num_processes = min(40, int(mp.cpu_count()*0.5))
        self.multiprocessing=False

    def expectation_value_q_pauli(self, snapshots_shadow:Union[np.ndarray, tuple[list[list[str]], list[str]]], multiprocessing: bool=False) -> np.ndarray:
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
        if isinstance(snapshots_shadow, np.ndarray):
            for n, obs in enumerate(self.q_Pauli):               
                pauli_matrices =  reduce(np.kron, [self.gate_set[obs[i]] for i in range(self.nq)])
                fk[n] += np.trace(snapshots_shadow @ pauli_matrices)
        elif isinstance(snapshots_shadow, tuple):
            self.U_clifford_shadow, self.bit_string_shadow = snapshots_shadow
            if multiprocessing:
                with mp.Pool(processes=self.num_processes) as pool:
                    fk[:] = pool.map(self._compute_expectation, self.q_Pauli)
            else:
                fk[:] = [
                    np.mean([
                        self.shadow.get_expectation_value(observable, U, b)
                        for U, b in zip(self.U_clifford_shadow, self.bit_string_shadow)
                    ])
                    for observable in self.q_Pauli
                ]
        else:
            raise ValueError("snapshots_shadow must be a tuple or a numpy array")
        return fk.real


    def _compute_expectation(self, observable):
        return np.mean([
            self.shadow.get_expectation_value(observable, U, b)
            for U, b in zip(self.U_clifford_shadow, self.bit_string_shadow)
        ])


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

    def classical_shadow(self, Quantum_circuit: QuantumCircuit, density_matrix:bool=False, multiprocessing: bool=False) -> tuple[list[list[UnitaryGate]], list[str]]:                                        
        self.multiprocessing=multiprocessing
        if self.multiprocessing: 
            res=[]
            self.density_matrix=density_matrix
            with mp.Pool(processes=int(self.num_processes)) as pool:
                res=pool.map(self.get_snapshot_shadow, [Quantum_circuit]*self.shadow_size)
            pool.close()
            if not density_matrix:
                snapshots_Clifford, snapshots_bits_string = zip(*res)
                snapshots_Clifford=list(snapshots_Clifford)
                snapshots_bits_string=list(snapshots_bits_string)
                return snapshots_Clifford, snapshots_bits_string
            else:
                res_array = np.array(res)
                return np.mean(res_array, axis=0)
        else:
            return self.shadow.classical_shadow(Quantum_circuit, self.shadow_size, density_matrix)
            
    
    def get_snapshot_shadow(self,Quantum_circuit) -> tuple[list[list[UnitaryGate]], list[str]]:
        "return a shadow snapshot "
        return self.shadow.snapshot_classical_shadow(Quantum_circuit, self.density_matrix)
        
    

    def correlation_matrix(self, D: np.ndarray) -> np.ndarray:
        """Calculate the normalize correlation matrix of X as C=(X.Xt)/No
        Args:
            D (np.ndarray): matrix to get the correlation matrix

        Returns:
            np.ndarray: correlation matrix
        """
        Dt = np.transpose(D)
        C = (D@Dt)/self.No
        C = np.array(C)
        self.C = C
        return C



    def shadow_spectro(self, hamil, init_state: Union[np.ndarray, list, QuantumCircuit] = None, N_Trotter_steps: int = 1000, density_matrix: bool = False, verbose: bool = True, Data_Matrix:bool=False, serialize=False, multiprocessing:bool =False) -> tuple[float, float, np.ndarray, np.ndarray]:
        """
        Perform complete shadow spectroscopy using Trotterized time evolution.

        This is the main function that executes shadow spectroscopy on a given system.
        The Hamiltonian can be provided directly or through a `UnitaryGate` constructor 
        for time-dependent evolution.

        Parameters
        ----------
        hamil : Hamiltonian or UnitaryGate
            The Hamiltonian governing time evolution. Either an instance of the `Hamiltonian` class 
            (see `Hamiltonian` folder), or a callable that returns a `UnitaryGate` for time evolution.
        
        init_state : np.ndarray or list or QuantumCircuit, optional
            Initial state of the system. Can be a statevector, list of per-qubit states, or a 
            `QuantumCircuit` to prepend to each run. Defaults to `None`.
        
        N_Trotter_steps : int, optional
            Number of discrete Trotter steps for the time evolution. Defaults to `1000`.
        
        density_matrix : bool, optional
            If `True`, computes the full density matrix for each snapshot. 
            **Warning**: introduces exponential memory complexity. Defaults to `False`.
        
        verbose : bool, optional
            If `True`, displays progress bars and other status messages. Defaults to `True`.
        
        Data_Matrix : bool, optional
            If `True`, returns the full data matrix in addition to the spectroscopy results. Defaults to `False`.
        
        serialize : bool, optional
            If `True`, serializes circuits for faster reuse. Circuits will still be converted to 
            `QuantumCircuit` objects during simulation. Defaults to `False`.
        
        multiprocessing : bool, optional
            If `True`, parallelizes time evolution over multiple processes. Defaults to `False`.

        Returns
        -------
        tuple of np.ndarray
            A tuple `(Intensity, Frequencies)`, where:
            - `Intensity` is the spectral intensity at each frequency.
            - `Frequencies` is the array of frequency values.
            Length of both arrays is `N_Trotter_steps // 2`.
        """
        
        
        D = np.zeros((self.Nt, self.No))
        T = np.linspace(0.00, self.Nt * self.dt, self.Nt)
        is_unitary = isinstance(
            hamil(1), UnitaryGate) if callable(hamil) else False
        self.density_matrix = density_matrix
        for n, t in tqdm(enumerate(T), desc="Time evolution", disable=not verbose):
            if is_unitary:
                circ = QuantumCircuit(self.nq)
                if isinstance(init_state, (np.ndarray, list)):
                    circ.initialize(init_state, normalize=True)
                circ.append(hamil(t), range(self.nq))
                C = init_state.compose(circ) if isinstance(
                    init_state, QuantumCircuit) else circ
            else:
                C = hamil.gen_quantum_circuit(t, init_state=init_state, N_Trotter_steps=N_Trotter_steps, serialize=serialize)

            self.multiprocessing=multiprocessing
            snapshots_shadow = self.classical_shadow(C, self.density_matrix, self.multiprocessing)
            fkt = self.expectation_value_q_pauli(snapshots_shadow, multiprocessing=True)
            D[n][:] = fkt

        if Data_Matrix:
            return D    
        frequencies,solution = self.spectro.Spectroscopy(D)

        return  frequencies,solution
##############################################################################################################################

    