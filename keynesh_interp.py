# Save scipyillowing to WannierEph.py:
import matplotlib.pyplot as plt
import numpy as np
import sys
import h5py
import pandas as pd
import concurrent.futures
from scipy.optimize import curve_fit
import itertools
from multiprocessing import Value, Lock#np.set_printoptions(precision=8, threshold=1000000, linewidth=1000000, suppress=True)

# ====================================================================
# Here we are reading in data output by JDFTx, no modification needed
# ====================================================================

# Read the MLWF cell map, weights and Hamiltonian:
cellMap = np.loadtxt("wannier.mlwfCellMap")[:,0:3].astype(int)
Wwannier = np.fromfile("wannier.mlwfCellWeights")
nCells = cellMap.shape[0]
nBands = int(np.sqrt(Wwannier.shape[0] / nCells))
Wwannier = Wwannier.reshape((nCells,nBands,nBands)).swapaxes(1,2)
# --- Get cell volume, mu and k-point folding from totalE.out:
for line in open('totalE.out'):
    if line.startswith("unit cell volume"):
        Omega = float(line.split()[-1])
    if line.startswith('\tFillingsUpdate:'):
        mu = float(line.split()[2])
    if line.startswith('kpoint-folding'):
        kfold = np.array([int(tok) for tok in line.split()[1:4]])
kfoldProd = np.prod(kfold)
kStride = np.array([kfold[1]*kfold[2], kfold[2], 1])

# --- Read reduced Wannier Hamiltonian, momenta and expand them:
Hreduced = np.fromfile("wannier.mlwfH").reshape((kfoldProd,nBands,nBands)).swapaxes(1,2)
iReduced = np.dot(np.mod(cellMap, kfold[None,:]), kStride)
Hwannier = Wwannier * Hreduced[iReduced]

# Read phonon dispersion relation:
cellMapPh = np.loadtxt('totalE.phononCellMap', usecols=[0,1,2]).astype(int)
nCellsPh = cellMapPh.shape[0]
omegaSqR = np.fromfile('totalE.phononOmegaSq')  # just a list of numbers
nModes = int(np.sqrt(omegaSqR.shape[0] // nCellsPh))
omegaSqR = omegaSqR.reshape((nCellsPh, nModes, nModes)).swapaxes(1,2)

# Read e-ph matrix elements
cellMapEph = np.loadtxt('wannier.mlwfCellMapPh', usecols=[0,1,2]).astype(int)
nCellsEph = cellMapEph.shape[0]

# --- Get phonon supercell from phonon.out:
for line in open('phonon.out'):
    tokens = line.split()
    if len(tokens)==5:
        if tokens[0]=='supercell' and tokens[4]=='\\':
            phononSup = np.array([int(token) for token in tokens[1:4]])
prodPhononSup = np.prod(phononSup)
phononSupStride = np.array([phononSup[1]*phononSup[2], phononSup[2], 1])

# --- Read e-ph cell weights:
nAtoms = nModes // 3
cellWeightsEph = np.fromfile("wannier.mlwfCellWeightsPh").reshape((nCellsEph,nBands,nAtoms)).swapaxes(1,2)
cellWeightsEph = np.repeat(cellWeightsEph.reshape((nCellsEph,nAtoms,1,nBands)), 3, axis=2)  # repeat atom weights for 3 directions
cellWeightsEph = cellWeightsEph.reshape((nCellsEph,nModes,nBands))  # coombine nAtoms x 3 into single dimension: nModes

# --- Read, reshape and expand e-ph matrix elements:
iReducedEph = np.dot(np.mod(cellMapEph, phononSup[None,:]), phononSupStride)
HePhReduced = np.fromfile('wannier.mlwfHePh').reshape((prodPhononSup,prodPhononSup,nModes,nBands,nBands)).swapaxes(3,4)
HePhWannier = cellWeightsEph[:,None,:,:,None] * cellWeightsEph[None,:,:,None,:] * HePhReduced[iReducedEph][:,iReducedEph]

# END read in ===============================================================

# Below here is interesting

# ------------------------------------------------------------------
# Electronic Wannier interpolation
# Calculate energies, eigenvectors and velocities for given k
# ------------------------------------------------------------------
def calcE(k):
    # Fourier transform to k:
    phase = np.exp((2j*np.pi)*np.dot(k,cellMap.T))
    H = np.tensordot(phase, Hwannier, axes=1)
    # Diagonalize and switch to eigen-basis:
    E,U = np.linalg.eigh(H)  # Diagonalize
    return E, U

# ------------------------------------------------------------------
# Phonon Fourier interpolation
# Calculate phonon energies and eigenvectors for given q
# ------------------------------------------------------------------
def calcPh(q):
    # construct phases for the Fourier transform
    phase = np.exp((2j*np.pi)*np.tensordot(q,cellMapPh.T, axes=1))
    omegaSq, U = np.linalg.eigh(np.tensordot(phase, omegaSqR, axes=1))
    omegaPh = np.sqrt(np.maximum(omegaSq, 0.))
    return omegaPh, U

# ------------------------------------------------------------------
# Electron-phonon Wannier interpolation
# Calculate e-ph matrix elements, along with ph and e energies, and e velocities
# ------------------------------------------------------------------
def calcEph(k1, k2, HePhWannier):

    # Electrons:
    E1, U1 = calcE(k1)
    E2, U2 = calcE(k2)

    # Phonons for all pairs pf k1 - k2:
    omegaPh, Uph = calcPh(k1[:,None,:] - k2[None,:,:])

    # E-ph matrix elements for all pairs of k1 - k2:
    phase1 = np.exp((2j * np.pi) * np.dot(k1,cellMapEph.T))
    phase2 = np.exp((2j * np.pi) * np.dot(k2,cellMapEph.T))

    arg = np.dot(k1,cellMapEph.T)
    phasePrint = phase1.flatten();

    normFac = np.sqrt(0.5 / np.maximum(omegaPh,1e-6))
    normFac = np.ones(normFac.shape)
    g = np.einsum(
        'kKy, kac, Kbd, kKxy, kr, KR, rRxab -> kKycd',
        normFac, U1.conj(), U2, Uph, phase1.conj(), phase2, HePhWannier,
        optimize='optimal'
    )
    return g, omegaPh, E1, E2


def calcEph_truncated(k1, k2, HePhWannier_truncated):

    # Electrons:
    E1, U1 = calcE(k1)
    E2, U2 = calcE(k2)

    # Phonons for all pairs pf k1 - k2:
    omegaPh, Uph = calcPh(k1[:,None,:] - k2[None,:,:])

    # E-ph matrix elements for all pairs of k1 - k2:
    phase1 = np.exp((2j * np.pi) * np.dot(k1,cellMapEph.T))
    phase2 = np.exp((2j * np.pi) * np.dot(k2,cellMapEph.T))

    # arg = np.dot(k1,cellMapEph.T)
    # phasePrint = phase1.flatten();

    normFac = np.sqrt(0.5 / np.maximum(omegaPh,1e-6))
    normFac = np.ones(normFac.shape)
    g = np.einsum(
        'kKy, kac, Kbd, kKxy, kr, KR, rRxab -> kKycd',
        normFac, U1.conj(), U2, Uph, phase1.conj(), phase2, HePhWannier_truncated,
        optimize='optimal'
    )
    return g, omegaPh, E1, E2

# ------------------------------------------------------------------
# Call the above functions and print the results
# ------------------------------------------------------------------

# set up example kpoints in crystal coordinates
k1 = np.array([[0.0, 0.0, 0.]])
k2 = np.array([[-0.375, 0.375, 0.]])

# Get e-ph properties
g, omegaPh, E1, E2  = calcEph(k2,k1,HePhWannier)

# Shared counter and lock
counter = Value('i', 0)
lock = Lock()


# Transpose the array to bring the last two dimensions to the front
T_transposed = HePhWannier.transpose(2, 3, 4, 0, 1)  # New shape will be (9, 8, 8, 131, 131)

# Function to find max value and terms till 1% of max
def findMaxAndOnePercent(array):
    maxValue = np.max(array)
    threshold = maxValue * 0.001
    termsToOnePercent = np.sum(array >= threshold)
    return maxValue, termsToOnePercent

# Function to truncate matrices
def truncate_matrices(U, S, Vt, r):
    U_trunc = U[:, :r]
    S_trunc = S[:r]
    Vt_trunc = Vt[:r, :]
    return U_trunc, S_trunc, Vt_trunc


# Function to increment and print the counter
def increment_counter():
    with lock:
        counter.value += 1
        print(f"Processed {counter.value} truncated results")

# Function to perform SVD on a specific slice and additional calculations
def compute_svd(params):
    k, l, m, T, g = params

    matrix_2D = T[k, l, m, :, :]  # Extract (131, 131) slice+
    U, S, Vt = np.linalg.svd(matrix_2D, full_matrices=False)
    
    # Calculate max singular value and terms to 1% of max
    maxSingularValue, termsToOnePercent = findMaxAndOnePercent(S)
    
    # Set the desired truncation dimension
    r = int(termsToOnePercent)

    # Reconstruct the truncated matrix
    U_trunc, S_trunc, Vt_trunc = truncate_matrices(U, S, Vt, r)
    truncated_matrix = np.dot(U_trunc, np.dot(np.diag(S_trunc), Vt_trunc))
    
    # Create a 5D array with the correct shape
    truncated_result = np.empty((131, 131, T.shape[0], T.shape[1], T.shape[2]))
    truncated_result[:, :, k, l, m] = truncated_matrix
    # print("Shape of truncted_result is: ", truncated_result.shape)


    sum_result, _, _, _ = calcEph_truncated(k2, k1, truncated_result)
    
    # Extract the scalar value from g
    gOld = g[0, 0, k, l, m]
    gNew = sum_result[0,0,k,l,m]

    
    # Increment and print the counter
    increment_counter()

    return (k, l, m, maxSingularValue, termsToOnePercent, gNew*gNew.conj(), gOld*gOld.conj())

# Prepare the parameters for each task using itertools.product
params = [(k, l, m, T_transposed, g) for k, l, m in itertools.product(range(9), range(8), range(8))]

# Use ThreadPoolExecutor for parallel execution
with concurrent.futures.ProcessPoolExecutor() as executor:
    results = list(executor.map(compute_svd, params))

# Filter out None results
results = [result for result in results if result is not None]

# Create a DataFrame from the results
df = pd.DataFrame(results, columns=['k', 'l', 'm', 'max_singular_value', 'terms_to_one_percent', 'exponential_decay', 'sum_result', 'g_value'])

# Sort the DataFrame by columns 'k', 'l', 'm' in ascending order
df_sorted = df.sort_values(by=['k', 'l', 'm'], ascending=[True, True, True])

# Reset the index if you want a clean index
df_sorted.reset_index(drop=True, inplace=True)

# Save the DataFrame to a CSV file
df_sorted.to_csv('svd_results.csv', index=False)

# Display the DataFrame
print(df_sorted.head())
