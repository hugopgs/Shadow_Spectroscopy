
from qiskit import QuantumCircuit, transpile
from tqdm import tqdm 
from qiskit.quantum_info import Pauli
import random
import numpy as np
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

# single shot classical shadow
class ClassicalShadow:
    """single shot classical shadow
    """
    def __init__(self, noise_error=None):
        self.err=noise_error
    
    def get_bit_string(self,circ:QuantumCircuit, shots=1)->str:
        """bit string measurement of a given quantum circuit.

        Args:
            circ (QuantumCircuit): quantum circuit to measure
            shots (int, optional): number of shots. Defaults to 1.

        Returns:
            str: bit string 
        """
        if isinstance(self.err, (list, np.ndarray)):
            nm = NoiseModel()
            nm.add_all_qubit_quantum_error(depolarizing_error(self.err[0], 1), ["x", "z"])
            nm.add_all_qubit_quantum_error(
                depolarizing_error(self.err[1], 2), ["rzz", "ryy", "rxx"])
            sim = AerSimulator(method="statevector", noise_model=nm)
        else: 
            sim = AerSimulator(method="statevector")
        
        try:
            counts = sim.run(circ, shots=1, ).result().get_counts()
        except Exception as e:
            counts = sim.run(transpile(circ,sim), shots=1 ).result().get_counts()
        res = list(counts.keys())[0]  # type: ignore
        return res
    
    def random_clifford_gate(self, idx:int=None )-> UnitaryGate:
        """return either a random clifford gate or a choosen clifford gate depending on the value of idx.
            idx=0: return UnitaryGate(I)
            idx=1: return UnitaryGate(X)
            idx=2: return UnitaryGate(Y)
            idx=3: return UnitaryGate(Z)
            idx=4: return UnitaryGate(V)
            idx=5: return UnitaryGate(V@X)
            idx=6: return UnitaryGate(V@Y)
            idx=7: return UnitaryGate(V@Z)
            idx=8: return UnitaryGate(W@X)
            idx=9: return UnitaryGate(W@Y)
            idx=10: return UnitaryGate(W@Z)
            idx=11: return UnitaryGate(H@X)
            idx=12: return UnitaryGate(H@Y)
            idx=13: return UnitaryGate(H@Z)
            idx=14: return UnitaryGate(H)
            idx=15: return UnitaryGate(H@V)
            idx=16: return UnitaryGate(H@V@X)
            idx=17: return UnitaryGate(H@V@Y)
            idx=18: return UnitaryGate(H@V@Z)
            idx=19: return UnitaryGate(H@W)
            idx=20: return UnitaryGate(H@W@X)
            idx=21: return UnitaryGate(H@W@Y)
            idx=22: return UnitaryGate(H@W@Z)
            idx=23: return UnitaryGate(W)
            idx= None: return randomly one of the previous Unitary gate 
        Args:
            idx (int, optional): index of the wanted clifford gate. if None return a random clifford gate. Defaults to None.
    
        Returns:
            UnitaryGate: Clifford Unitary gate object
        """
        X = np.array([[0,1],  [1,0]])
        Y = np.array( [[0,-1j], [1j,0]])
        Z = np.array( [[1,0],  [0,-1]])
        I = np.array( [[1,0],  [0,1]])
        S=np.array([[1,0],  [0,1j]])
        H= np.array([[1/np.sqrt(2),1/np.sqrt(2)],  [1/np.sqrt(2),-1/np.sqrt(2)]])
        V=H@S@H@S
        W=V@V
        if idx==None:
            idx = np.random.randint(0,23)
        if idx>=24 or idx<0:
            print("idx out of range, idx need to be between 0 and 23")
        if idx == 0:
            return UnitaryGate(I)                    
        elif idx == 1:
            return UnitaryGate(X)                      
        elif idx == 2:
            return UnitaryGate(Y)                     
        elif idx == 3:
            return UnitaryGate(Z) 
        elif idx == 4:
            return UnitaryGate(V)                  
        elif idx == 5:
            return UnitaryGate(V@X)
        elif idx == 6:
            return UnitaryGate(V@Y)               
        elif idx == 7:
            return UnitaryGate(V@Z)
        elif idx == 8:
            return UnitaryGate(W@X)             
        elif idx == 9:
            return UnitaryGate(W@Y)             
        elif idx == 10:
            return UnitaryGate(W@Z)
        elif idx == 11:
            return UnitaryGate(H@X)
        elif idx == 12:
            return UnitaryGate(H@Y)
        elif idx == 13:
            return UnitaryGate(H@Z)                     
        elif idx == 14:
            return UnitaryGate(H)
        elif idx == 15:
            return UnitaryGate(H@V)                    
        elif idx == 16:
            return UnitaryGate(H@V@X)
        elif idx == 17:
            return UnitaryGate(H@V@Y)       
        elif idx == 18:
            return UnitaryGate(H@V@Z)            
        elif idx == 19:
            return UnitaryGate(H@W)     
        elif idx == 20:
            return UnitaryGate(H@W@X)             
        elif idx == 21:
            return UnitaryGate(H@W@Y)                     
        elif idx == 22:
            return UnitaryGate(H@W@Z)                
        elif idx == 23:
            return UnitaryGate(W)    

    def add_random_clifford(self, circuit: QuantumCircuit)->tuple[list[UnitaryGate], QuantumCircuit]:
        """Add a random clifford gate for each qubits in a given circuit. add a "measure_all()" instruction after adding the clifford gates.

        Args:
            circuit (QuantumCircuit): circuit to add clifford gates

        Returns:
            tuple[list[UnitaryGate], QuantumCircuit]: the list of clifford gates applied to the circuit and the new circuit with the gate added
        """
        Circuit_copy=circuit.copy()
        clifford_gate=[]
        for qubits in range(Circuit_copy.num_qubits):
            clifford_gate.append(self.random_clifford_gate())
            Circuit_copy.append(clifford_gate[-1],[qubits])
        Circuit_copy.measure_all()
        return clifford_gate, Circuit_copy


    def classical_shadow(self, circuit: QuantumCircuit)->tuple[list[UnitaryGate], str]:
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
        
        circuit_copy = circuit.copy() # create a copy of the circuit to avoid modifying it in-place
        unitary_list, shadow_circuit= self.add_random_clifford(circuit_copy) #add a random clifford gate
        measurement_result_list = self.get_bit_string(shadow_circuit, shots=1) #measurement 
        return unitary_list, measurement_result_list# res is already a real number but in the form: a+0.j, to avoid type problem we get rid of the imaginary part


    def get_expectation_value(self, obs:str, unitary_list:list[UnitaryGate], measurement_result_list:str)->float:
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
            P=np.array(Pauli(obs[-(1+n)]).to_matrix())
            U_mat=np.array(unitary_list[n].to_matrix()) # clifford gate to matrix
            U_mat_dagg=np.conj(U_mat).T 
            if int(measurement_result_list[-(1+n)])==0:
                bite_string_matrix=np.array([[1,0],[0,0]]) 
            else:
                bite_string_matrix=np.array([[0,0],[0,1]])
                
            expectation_value *= np.trace((3*U_mat_dagg@bite_string_matrix@U_mat-np.identity(2))@P)
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
        for n in range(len(unitary_list)):
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

        