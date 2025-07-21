from qiskit.circuit.library import UnitaryGate
from qiskit import QuantumCircuit
from scipy.linalg import expm
import numpy as np

from Shadow_Spectro.ClassicalShadow import ClassicalShadow
from Shadow_Spectro.Spectroscopy import Spectroscopy
from Shadow_Spectro.ShadowSpectro import ShadowSpectro


def Generate_Unitary_Hermitian_Matrix(numQbits,eigenvalues):
    """
    Generate a Hermitian matrix with specified eigenvalues.

    Args:
        eigenvalues (list or np.ndarray): A list of desired eigenvalues.

    Returns:
        np.ndarray: A Hermitian matrix with the given eigenvalues.
    """
    diagonal_matrix = np.identity(2**numQbits)
    k=0
    
    for eigenvalue, multiplicity in eigenvalues:      
        for i in range(multiplicity):
            diagonal_matrix[k+i][k+i]=eigenvalue
        k+=multiplicity
    
    # Generate a random unitary matrix (P)
    random_matrix = np.random.randn(2**numQbits, 2**numQbits) + 1j * np.random.randn(2**numQbits, 2**numQbits)
    Q, _ = np.linalg.qr(random_matrix)  # QR decomposition to get a unitary matrix
    # Construct the Hermitian matrix: H = P \Lambda P^†
    hermitian_matrix = Q @ diagonal_matrix @ Q.conj().T
    return hermitian_matrix


def Generate_Evolution_Matrix(hermitian_matrix:np.ndarray):
    """Frome a given hermitian matrix
    generate an evolution matrix as U(t)= exp(-iHt)

    Args:
        hermitian_matrix (np.ndarray): The Hermitian matrix

    Returns:
        function : A function of the time Unitary Gate evolution matrix
    """
    hamil=(lambda t: UnitaryGate(expm(-1.j*hermitian_matrix*t)))
    return hamil

if __name__=='__main__':
    # parameters
    eigenvalues=[(2,2**5),(8,2**5)]
    Energy_gap=6 #rad/s
    numQbits=6
    Nt=200
    dt=0.21 #s
    T = np.linspace(0,Nt*dt,Nt)#
    shadow_size= 10
    
    # Generate random hermitian matrix and evolution matrix with wanted eigenvalues
    hermitian_matrix=Generate_Unitary_Hermitian_Matrix(numQbits,eigenvalues)
    evolution_matrix=Generate_Evolution_Matrix(hermitian_matrix)
        

    # Initialize the shadow and spectroscopy objects
    shadow=ClassicalShadow()
    spectro= Spectroscopy(Nt,dt)
    shadow_spectro=ShadowSpectro(shadow,spectro, numQbits,3)


######################### Shadow spectroscopy #############################################
    frequencies,solution=shadow_spectro.shadow_spectro(evolution_matrix,density_matrix=True)
    
    
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(frequencies, solution, color='blue', label=f"Spectral cross correlation")
    plt.axvline(x=Energy_gap, color='black', linestyle='--', linewidth=1, label=f"Real Energie Gap: {Energy_gap:.2f} rad/s")
    plt.text(Energy_gap, max(solution)*1.01, f"{Energy_gap:.2f} (rad/s)", color='Black', ha='left', va='center', fontsize=10)
        
            
    # Enhance aesthetics
    plt.grid(True, linestyle='--', color='gray', linewidth=0.5, alpha=0.7)
    plt.title(f"Spectral Cross-Correlation", fontsize=14, fontweight='bold')
    plt.xlabel("Frequency (rad/s)", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)
    plt.legend(fontsize=10, loc="best")


################################### IMPORTANT ###############################################
"""
To calculate The No signals shadow spectroscopy will calculate the evolution matrix of the hamiltonian for differents time t. 

To work the input Hamiltonian need to be either : 

- A function of t that return a qiskit UnitaryOperation.

- Or an Hamiltonian class defined as follow : 
        - Attributs (at least): nqubits: int
                                terms: List[Tuple[str, List[int], Callable[[float], float]]]
        where nqubits is the numbers of qubits and terms represent the differents terms of the Hamiltonian exemple
        terms=[("ZZ",[0,1], J(t)),("ZZ",[1,2], J(t)),("ZZ",[2,3], J(t))]
        which is equivalent to a ZZ operation on qubits [0,1], [1,2],[2,3], the coefficient J(t) is a function of t:
        Exemple of a function that creat such an Hamiltonian : 
        @dataclass
        class Hamiltonian:
            nqubits: int
            terms: List[Tuple[str, List[int], Callable[[float], float]]]

            def __post_init__(self):
                print("The number of qubit:" + str(self.nqubits))
                print("Number of terms in the Hamiltonian:" + str(len(self.terms)))
            @staticmethod
            def Transverse_Ising_hamil(n, J, d):
                terms = [("ZZ", [k, (k + 1)%n], lambda t: -1*J*t)for k in range(n)]
                terms += [("X", [k], lambda t: -1*d*t) for k in range(n)]
        
         - Methods (at least):  gen_quantum_circuit: a method that whatever t return the quantum circuit representation of the Hamiltonian at t ex:
            def gen_quantum_circuit(self, t: float)->QuantumCircuit:
                nq=self.nqubits
                circ = QuantumCircuit(nq)
                for pauli, qubits, coef in self.get_term(t):
                    circ.append(self.__rgate(pauli, coef), qubits)
                return circ
    
            def __rgate(self, pauli, r):
                return {
                    "X": RXGate(r),
                    "Z": RZGate(r),
                    "XX": RXXGate(r),
                    "YY": RYYGate(r),
                    "ZZ": RZZGate(r),
                }[pauli]
                
Exemple of Hamiltonian can be found at :                 
            




"""
    
