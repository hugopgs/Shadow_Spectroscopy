import numpy as np
import itertools
from functools import reduce 
from operator import concat 
from qiskit.circuit.library import UnitaryGate

class ShadowSpectro:
    def __init__(self, shadow , spectro, nq: int, k : int ):
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
        self.C=None
                  
    def expectation_value_q_Pauli(self, U_clifford_shadow: list[list[UnitaryGate]], bit_string_shadow: list[str])->np.ndarray:
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
        for observable in self.q_Pauli:
            shadow_expectation_value=[]
            for U_clifford, bit_string in zip(U_clifford_shadow, bit_string_shadow):
                 shadow_expectation_value.append(self.shadow.get_expectation_value(observable, U_clifford, bit_string))
            fk.append(np.mean(shadow_expectation_value))
        return np.array(fk)
          
    def standardisation(self, Matrix:np.ndarray)->np.ndarray:
        """
        Standardise observables according to the classical shadow.
        
        Args:
            vectors: list[float]: vector to standardize
        Returns:
            np.ndarray: standardize vector
        """
        Matrix=np.transpose(Matrix)
        standardize_matrix=[]
        for vector in Matrix:
             standardize_matrix.append((np.array(vector)-np.mean(vector))/np.std(vector).tolist())

        return np.transpose(np.array(standardize_matrix))
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

    
    def reduction(self, D: np.ndarray)->np.ndarray:
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
    
    def shadow_spectro(self)->tuple[ np.ndarray,np.ndarray]:
        """Do Shadow spectroscopy by doing:
        
        ## Step 1:
            Get the dominant eigenvectors of the correlation matrix.
            For more details: see spectro.get_dominant_eigenvectors
        ## Step 2:
            Do the spectral_cross_correlation of the eigenvectors
            for more details: see spectro.spectral_cross_correlation
        Args:
            plot (bool, optional): If True plot the frequency spectrum. Defaults to False.

        Returns:
             tuple[float, float, np.ndarray,np.ndarray]: (float, float ): frequencies with highest amplitude in the frequency spectrum
            (np.ndarray, np.ndarray): amplitude, frequencies: frequency spectrum
        """
        self.list_eigenvector=self.spectro.get_dominant_eigenvectors(self.C )
        solution, frequencies= self.spectro.spectral_cross_correlation(self.list_eigenvector)

        return solution, frequencies