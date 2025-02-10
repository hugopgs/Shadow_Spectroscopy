import numpy as np
import itertools
from functools import reduce 
from operator import concat 
from qiskit.circuit.library import UnitaryGate
from tqdm import tqdm
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from typing import Union
class ShadowSpectro:
    def __init__(self, shadow , spectro, nq: int, k : int, shadow_size : int)-> None:
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
        self.spectro=spectro
        self.q_Pauli = self.q_local_shadow_observable(self.k)
        self.No=len(self.q_Pauli)
        self.C= np.ndarray
        self.shadow_size=shadow_size
        try:
            self.Nt=spectro.Nt
            self.dt=spectro.dt
        except : 
            Warning("spectroscopy class have no attributs")
        
    def expectation_value_q_Pauli(self, U_clifford_shadow: list[list[UnitaryGate]], bit_string_shadow: list[str], density_matrix=None)->np.ndarray:
        """Get the expectation value of all k-Pauli Observable from a list of classical snapshot, CliffordGate and bit_string. 
           The length of bit_string_shadow and U_clifford_shadow is equal  of the number of snapshot for classical shadow.
           The length of list[UnitaryGate] and bit_string (str) is equal to the number of qubits in the system. 
           The function calculate the expectation value average of all snapshot for each k-Pauli Observable.
           Return a 1d array with the expectation value of all k-Pauli Observable.
           The array is standardize as [array-mean(array)]/std(array).

        Args:
            U_clifford_shadow (list[list[UnitaryGate]]): Clifford gate applied to each snapshot for each Qubits. 
            bit_string_shadow (list[str]): bit string measurement for each snapshot.

        Returns:
            np.ndarray: Expectation value of each Observable.
        """
        fk = []
        if isinstance(density_matrix, (np.ndarray, list)) :
            for observable in self.q_Pauli:
                obs=np.array(Pauli(''.join(observable)).to_matrix())
                fk.append(np.trace(density_matrix @ obs))
        else:
            for observable in self.q_Pauli:
                shadow_expectation_value=[]
                for U_clifford, bit_string in zip(U_clifford_shadow, bit_string_shadow):
                    shadow_expectation_value.append(self.shadow.get_expectation_value(observable, U_clifford, bit_string))
                fk.append(np.mean(shadow_expectation_value))
        return np.array(fk)
 

    
    def q_local_shadow_observable(self,K: int)->list[str]:
        """Generate the sequence of all the observable from 1-Pauli observable to K-Pauli observable

        Args:
            K (int): K-pauli observable to generate
        Returns:
            list[str]: list of all the observable from 1-Pauli observable to K-Pauli observable
        """
        q_local=[]
        for k in range(K):
            q_local.append(self.q_local_Pauli(k+1))
        return reduce(concat, q_local)
     
    def q_local_Pauli(self, k: int)->list[str]:
        """Generate the sequence of all the k-Pauli observable

        Args:
            k (int):  K-pauli observable to generate

        Returns:
            list[str]:  list of all the k-Pauli observable
        """
            
        pauli_operators=["X", "Y", "Z",]
        q_local=[]

        all_combinations = list(itertools.product(pauli_operators, repeat=k))
        for positions in itertools.combinations(range(self.nq), k):
            for combination in all_combinations:
                observable = ['I'] * self.nq
                
                for i, pos in enumerate(positions):
                    observable[pos] = combination[i]

                q_local.append(tuple(observable))
            
        return q_local

    def get_snapshots_classical_shadow(self, Quantum_circuit: QuantumCircuit)->tuple[list[list[UnitaryGate]], list[str]]:
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
        snapshots_Clifford=[]
        snapshots_bits_string=[]
        for i in range(self.shadow_size):
            Clifford_gates, bits_string=self.shadow.classical_shadow(Quantum_circuit)
            snapshots_Clifford.append(Clifford_gates)
            snapshots_bits_string.append(bits_string)
        return snapshots_Clifford, snapshots_bits_string
    
    
    
    def correlation_matrix(self, D: np.ndarray)->np.ndarray:
        """Calculate the normalize correlation matrix of X as C=(X.Xt)/No

        Args:
            X (np.ndarray): matrix to get the correlation matrix

        Returns:
            np.ndarray: correlation matrix
        """
        Dt=np.transpose(D)
        C =  (D@Dt)/self.No
        C=np.array(C)
        self.C=C
        return C
    
    def shadow_spectro(self, hamil , init_state: Union[np.ndarray, list, QuantumCircuit]=None, density_matrix:bool=False, verbose: bool=True)->tuple[float, float, np.ndarray,np.ndarray]:
        """Do Shadow spectroscopy by doing:
        
        ## Step 1:
            Get the dominant eigenvectors of the correlation matrix.
            For more details: see spectro.get_dominant_eigenvectors
        ## Step 2:
            Do the spectral_cross_correlation of the eigenvectors
            for more details: see spectro.spectral_cross_correlation
        Args:
           Ljung (bool, optional): If True do a Ljung box test. Defaults to True.

        Returns:
             tuple[float, float, np.ndarray,np.ndarray]: (float, float ): frequencies with highest amplitude in the frequency spectrum
            (np.ndarray, np.ndarray): amplitude, frequencies: frequency spectrum
        """
        D=[]
        T = np.linspace(0.00,self.Nt*self.dt,self.Nt)
        try:
            if isinstance(hamil(1), UnitaryGate):
                flag=True     
        except:
            flag=False
        
        for t in tqdm(T, desc="Time evolution", disable= not verbose):
            if flag:
                circ = QuantumCircuit(self.nq)
                if isinstance(init_state,(np.ndarray, list)):
                    circ.initialize(init_state, normalize=True)
                circ.append(hamil(t),[n for n in range(self.nq)])
                C=circ.copy()
                if  isinstance(init_state,QuantumCircuit):
                    C = init_state.compose(circ,[i for i in range(self.nq)])
            else:    
                C=hamil.gen_quantum_circuit(t,init_state=init_state)
            
            if density_matrix:
                Rho=np.zeros((2**self.nq,2**self.nq), dtype='complex128')
                for i in range(self.shadow_size):
                    cliff,bit=self.shadow.classical_shadow(C)
                    Rho+=self.shadow.snapshot_density_matrix(cliff, bit)
                Rho=Rho/self.shadow_size
                fkt=self.expectation_value_q_Pauli(0,0,density_matrix=Rho)
                D.append(fkt.tolist())
            else:
                snapshots_clifford, snapshots_bit_string = self.get_snapshots_classical_shadow(C)
                fkt=self.expectation_value_q_Pauli(snapshots_clifford, snapshots_bit_string)
                D.append(fkt.tolist())
        D=np.array(D).real
        
        solution, frequencies= self.spectro.Spectroscopy(D)

        return solution, frequencies