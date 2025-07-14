# README: Shadow Spectroscopy

This repository contains three Python classes designed to implement shadow spectroscopy.  Shadow spectroscopy is a simulator-agnostic quantum algorithm for estimating energy gaps using very few circuit repetitions (shots) and no extra resources (ancilla qubits) beyond performing time evolution and measurement.

Based on the following paper :  https://arxiv.org/abs/2212.11036, from : Hans Hon Sang Chan, Richard Meister, Matthew L. Goh, Bálint Koczor.

## Overview of the Classes

The classes included in this repository are:

1. **ClassicalShadow**
   - Implements the functionality to generate classical shadow snapshot, a data-efficient representation of quantum states.

2. **Spectroscopy**
   - Provides tools for spectral analysis of quantum systems, including eigenvector extraction and frequency domain analysis.

3. **ShadowSpectro**
   - Combines the features of classical shadows and spectroscopy to perform shadow spectroscopy on quantum systems. It integrates functionalities from the `ClassicalShadow` and `Spectroscopy` classes.

---

### 1. ClassicalShadow

#### Description
The `ClassicalShadow` class provides tools for efficiently storing and extracting properties of quantum states based on random measurements and Clifford operations.

---

### 2. Spectroscopy

#### Description
The `Spectroscopy` class includes methods for analyzing the spectral properties of a quantum system. It focuses on extracting key frequencies and their associated amplitudes from a system's correlation matrix.

---

### 3. ShadowSpectro

#### Description
The `ShadowSpectro` class integrates classical shadow techniques with spectral analysis to perform shadow spectroscopy. This involves generating classical snapshots, constructing correlation matrices, and performing spectral analysis to extract key frequencies.

## Workflow
The shadow spectroscopy process consists of the following steps:

### Step 1: Classical Shadows

1. Using classical shadow, calculate for Nt time stepes the expectation value of all the k-pauli observable. The results can be represented in a Matrix with $N_o$ column representing  the expectation value of all the No observables and $N_t$ lines representing the time evolution of the expectation value of the $N_o$ observables. We call D that matrix. 
2. Standardize the columns of the matrix D
 
### Step 2: Spectral Analysis

1. Compute the correlation matrix. $C=DD^T$
2. Extract the dominant eigenvectors.
3. Perform spectral analysis to identify key frequencies and amplitudes.
4. The highest peak represent the energy difference between the energy level of the hamiltonian

## Dependencies
- `numpy`
- `itertools`
- `qiskit`

---

## License
This project is licensed under the MIT License.

---

For any issues or questions, feel free to reach out!

