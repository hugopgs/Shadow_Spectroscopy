import numpy as np
from math import comb
import itertools
import heapq
from statsmodels.stats.diagnostic import acorr_ljungbox
from functools import reduce
from operator import concat


class Spectroscopy:
    def __init__(self, Nt: int, dt: float, cutoff: int = 4):
        """Class constructor for time depandent 2d matrix cross correlation Spectroscopy

        Args:
            Nt (int): number of time steps
            dt (float): time step
            cutoff (int, optional): number of dominant eigenvectors to keep. Defaults to 4.
        """
        self.dt = dt
        self.Nt = Nt
        self.cutoff = cutoff

    def standardisation(self, Matrix: np.ndarray) -> np.ndarray:
        """
        Standardise observables according to the classical shadow.

        Args:
            vectors: list[float]: vector to standardize
        Returns:
            np.ndarray: standardize vector
        """
        Matrix = np.transpose(Matrix)
        standardize_matrix = []
        for vector in Matrix:
            standardize_matrix.append(
                (np.array(vector)-np.mean(vector))/np.std(vector).tolist())
        return np.transpose(np.array(standardize_matrix))

    def correlation_matrix(self, X: np.ndarray) -> np.ndarray:
        """Calculate the correlation matrix of X as C=X.Xt

        Args:
            X (np.ndarray): matrix to get the correlation matrix

        Returns:
            np.ndarray: correlation matrix
        """
        Xt = np.transpose(X)
        C = (X@Xt)
        C = np.array(C)
        return C

    def get_dominant_eigenvectors(self, matrix: np.ndarray) -> list:
        """ Return the dominant eigenvectors of the given matrix. i.e. the vectors with highest eigenvalue.
        The number of eigen vector choose is determined by the cutoff.

        Args:
            matrix (np.ndarray): Matrix to get the dominant eigenvectors

        Returns:
            list of np.ndarray:  dominant_eigenvectors of X
        """
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        max_eigenvalues = heapq.nlargest(self.cutoff, eigenvalues)
        max_index = [eigenvalues.tolist().index(val)
                     for val in max_eigenvalues]
        vectors = []
        for index in max_index:
            vectors.append(eigenvectors[:, index])
        return vectors

    def Ljung_Box_test(self, matrix: np.ndarray, Nmax: int = 100) -> np.ndarray:
        """ Ljung_Box test on the column of the given matrix. Return the best Nmax column as a 2d np.ndarray

        Args:
            matrix (np.ndarray): Matrix to conduct the test
            Nmax (int, optional): Number of column to keep. Default to 100

        Returns:
            np.ndarray: matrix of the best Nmax column """
        p_values = [acorr_ljungbox(column, lags=len(
            column)-1, return_df=False)["lb_pvalue"].array for column in np.transpose(matrix)]
        p_values = np.array([p[0] for p in p_values])
        sorted_indices = np.argsort(p_values)[:Nmax]
        Matrix_sorted = matrix[:, sorted_indices]
        return Matrix_sorted

    def hann_window(self, length: int) -> np.ndarray:
        """Generate a  Hanning window  of a given lenght

        Args:
            length (int): length of the  Hanning window 

        Returns:
            ndarray, shape(length,):  Hanning window 
        """
        return np.hanning(length)

    def xcorr(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """calculate the cross correlation between two vectors

        Args:
            x (np.ndarray): vectors 1
            y (np.ndarray): vectors 2

        Returns:
            np.ndarray: cross correlation between vectors x and y
        """
        xc = [np.mean(x[:-m] * y[m:]) for m in range(1, len(x))]
        return np.array(xc)

    def spectral_cross_correlation(self, list_eigenvector: list) -> tuple[float, float, np.ndarray, np.ndarray]:
        """ Do the spectral cross correlation given a set of vector.
        ## Step 1: 
            Generate a hanning window of length: (lenght(vectors)-1)
        ## Step 2: 
            Fourier transform of the product of the hanning windows with the cross-correlation between each pair of vectors
        ## Step 3: 
            Perform Singular Value Decomposition (SVD) on the Fourier transform data array from Step 2
        ## Step 5: 
            Return the frequency spectrum of the Fourier-transformed cross-correlation between each pair of eigenvectors
        ## Step 6: 
            calculate the frequencies of the two higest amplitude in the frequency spectrum 
        ## Step 7: 
            Return the frequency spectrum and the frequencies with highest amplitude in the frequency spectrum

        Args:
            list_eigenvector (list of np.ndarray): list of vectors to do the spectral cross correlation

        Returns:
            tuple[float, float, np.ndarray,np.ndarray]: (float, float ): frequencies with highest amplitude in the frequency spectrum
            (np.ndarray, np.ndarray): amplitude, frequencies: frequency spectrum
        """        """"""
        # length eigenvector
        Nt_corr = self.Nt - 1  # Adjusted length after correlation calculation
        total_time_sim = Nt_corr * self.dt  # Total time  in arbitrary time units

        # w = self.hann_window(Nt_corr)
        win = 1
        """k: Refers to the eigenvector list_eigenvector[k] (row in the eigenvector interaction matrix).
           l: Refers to the eigenvector list_eigenvectors[l] (column in the eigenvector interaction matrix).
           :: Represents the frequency spectrum of the Fourier-transformed cross-correlation between list_eigenvectors[k] and list_eigenvectors[l]."""
        data = np.array([
            [
                # Fourier transform of the weighted correlation
                np.fft.fft(
                    win * self.xcorr(list_eigenvector[k], list_eigenvector[l]))
                for l in range(len(list_eigenvector))
            ]
            for k in range(len(list_eigenvector))
        ])

        """Singular Value Decomposition (SVD):
           captures the strongest correlated behavior between the eigenvectors at that frequency
        """
        solution = np.array([
            np.max(np.linalg.svd(data[:, :, k], compute_uv=False))
            for k in range(0, Nt_corr)
        ])

        frequencies = np.linspace(
            0, 2*np.pi * Nt_corr / total_time_sim, len(solution))
        results = solution[:int(len(solution)/2)]
        frequencies = frequencies[:int(len(frequencies)/2)]
        return results, frequencies

    def Spectroscopy(self, Data_Matrix: np.ndarray, Ljung: bool = True):
        if Ljung:
            D = self.Ljung_Box_test(self.standardisation(Data_Matrix))
        else:
            D = self.standardisation(Data_Matrix)

        self.C = self.correlation_matrix(D)
        self.list_eigenvector = self.get_dominant_eigenvectors(self.C)
        solution, frequencies = self.spectral_cross_correlation(
            self.list_eigenvector)

        return solution, frequencies
