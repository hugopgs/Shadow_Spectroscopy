# README: Shadow Spectroscopy

This repository contains three Python classes designed to implement shadow spectroscopy. Shadow spectroscopy is a computational technique for efficiently estimating observable properties and spectral properties of quantum systems using a combination of classical shadows and spectral analysis.

## Overview of the Classes

The classes included in this repository are:

1. **ClassicalShadow**
   - Implements the functionality to generate and use classical shadows, a data-efficient representation of quantum states.

2. **Spectroscopy**
   - Provides tools for spectral analysis of quantum systems, including eigenvector extraction and frequency domain analysis.

3. **ShadowSpectro**
   - Combines the features of classical shadows and spectroscopy to perform shadow spectroscopy on quantum systems. It integrates functionalities from the `ClassicalShadow` and `Spectroscopy` classes.

---

### 1. ClassicalShadow

#### Description
The `ClassicalShadow` class provides tools for efficiently storing and extracting properties of quantum states based on measurements and Clifford operations.

#### Key Methods:
- `get_expectation_value(observable: str, U_clifford: list[UnitaryGate], bit_string: str) -> float`:
  Computes the expectation value of a given observable from a quantum snapshot.

---

### 2. Spectroscopy

#### Description
The `Spectroscopy` class includes methods for analyzing the spectral properties of a quantum system. It focuses on extracting key frequencies and their associated amplitudes from a system's correlation matrix.

#### Key Methods:
- `get_dominant_eigenvectors(matrix: np.ndarray) -> list[np.ndarray]`:
  Identifies dominant eigenvectors of a correlation matrix.

- `spectral_cross_correlation(eigenvectors: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]`:
  Analyzes the spectral cross-correlation of dominant eigenvectors and extracts frequencies and amplitudes.

---

### 3. ShadowSpectro

#### Description
The `ShadowSpectro` class integrates classical shadow techniques with spectral analysis to perform shadow spectroscopy. This involves generating classical snapshots, constructing correlation matrices, and performing spectral analysis to extract key frequencies.

#### Key Methods:

1. **Initialization**
   ```python
   __init__(shadow, spectro, nq: int, k: int)
   ```
   - **Parameters:**
     - `shadow (ClassicalShadow)`: Instance of the classical shadow class.
     - `spectro (Spectroscopy)`: Instance of the spectroscopy class.
     - `nq (int)`: Number of qubits in the system.
     - `k (int)`: Maximum order of k-Pauli observables to consider.

2. **Expectation Value Calculation**
   ```python
   expectation_value_q_Pauli(U_clifford_shadow: list[list[UnitaryGate]], bit_string_shadow: list[str]) -> np.ndarray
   ```
   - Calculates the expectation values of all k-Pauli observables based on snapshots.

3. **Standardization**
   ```python
   standardisation(matrix: np.ndarray) -> np.ndarray
   ```
   - Standardizes a matrix of observables.

4. **Observable Generation**
   ```python
   q_local_shadow_observable(K: int) -> list[str]
   ```
   - Generates all observables from 1-Pauli to K-Pauli.

   ```python
   q_local_Pauli(k: int) -> list[str]
   ```
   - Generates all k-Pauli observables.

5. **Correlation Matrix Construction**
   ```python
   reduction(D: np.ndarray) -> np.ndarray
   ```
   - Constructs and normalizes a correlation matrix from the input matrix `D`.

6. **Shadow Spectroscopy**
   ```python
   shadow_spectro() -> tuple[np.ndarray, np.ndarray]
   ```
   - **Steps:**
     1. Extract dominant eigenvectors of the correlation matrix.
     2. Perform spectral cross-correlation to identify key frequencies and their amplitudes.

---

## Workflow
The shadow spectroscopy process consists of the following steps:

### Step 1: Classical Shadows
Generate classical snapshots of the quantum system by measuring the system in random bases defined by Clifford operations.

### Step 2: Spectral Analysis
Using the snapshots:
1. Compute the correlation matrix.
2. Extract the dominant eigenvectors.
3. Perform spectral analysis to identify key frequencies and amplitudes.

### Step 3: Shadow Spectroscopy
Combine the classical shadow data with spectral analysis to extract physical properties and dynamic characteristics of the quantum system.

---

## Usage
1. Instantiate the `ClassicalShadow` and `Spectroscopy` classes.
2. Create an instance of the `ShadowSpectro` class using the previously instantiated objects.
3. Use the `shadow_spectro` method to perform shadow spectroscopy and obtain key spectral properties.

---

## Example
```python
from classical_shadow import ClassicalShadow
from spectroscopy import Spectroscopy
from shadow_spectro import ShadowSpectro

# Initialize components
shadow = ClassicalShadow()
spectro = Spectroscopy()
nq = 5  # Number of qubits
k = 3   # Maximum order of k-Pauli observables

# Create ShadowSpectro instance
shadow_spectro = ShadowSpectro(shadow, spectro, nq, k)

# Perform shadow spectroscopy
solution, frequencies = shadow_spectro.shadow_spectro()

# Output results
print("Dominant frequencies:", solution)
print("Frequency spectrum:", frequencies)
```

---

## Dependencies
- `numpy`
- `itertools`
- `qiskit`

---

## License
This project is licensed under the MIT License.

---

For any issues or questions, feel free to reach out!

