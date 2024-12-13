from qiskit.circuit.library import UnitaryGate
from qiskit import QuantumCircuit
from scipy.linalg import expm
import numpy as np

from Shadow_Spectro import ClassicalShadow
from Shadow_Spectro import Spectroscopy
from Shadow_Spectro import ShadowSpectro


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
        print(type(multiplicity))           
        for i in range(multiplicity):
            diagonal_matrix[k+i][k+i]=eigenvalue
        k+=multiplicity
    
    print(diagonal_matrix)
    
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
    eigenvalues=[(2,2**5),(8,2**5)]
    Energy_gap=6 #rad/s
    numQbits=6
    Nt=200
    dt=0.21 #s
    T = np.linspace(0,Nt*dt,Nt)#
    shadow_size= 10
    
    # Generate hermitian matrix and evolution matrix
    hermitian_matrix=Generate_Unitary_Hermitian_Matrix(numQbits,eigenvalues)
    evolution_matrix=Generate_Evolution_Matrix(hermitian_matrix)
        

    # Initialize the shadow and spectroscopy objects
    shadow=ClassicalShadow()
    spectro= Spectroscopy(Nt,dt)
    shadow_spectro=ShadowSpectro(shadow,spectro, numQbits,3)


    D=[]
    for t in T:
        C= QuantumCircuit(numQbits)
        C.append(evolution_matrix(t),[i for i in range(numQbits)])
        
        Clifford_array, bits_array=[],[]
        for i in range(shadow_size):
            Clifford_gates, bits_string=shadow.classical_shadow(C)
            Clifford_array.append(Clifford_gates)
            bits_array.append(bits_string)
    
        fkt=shadow_spectro.expectation_value_q_Pauli(Clifford_array,bits_array)
        D.append(fkt.tolist())
    
    D=np.array(D) 
    D= spectro.Ljung_Box_test(shadow_spectro.standardisation(D))
    C=shadow_spectro.reduction()
    solution, frequencies=shadow_spectro.shadow_spectro()
    
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




    