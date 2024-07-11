# Save the following to WannierEph.py:
import matplotlib.pyplot as plt
import numpy as np
import sys
import h5py
import pandas as pd
import concurrent.futures
from scipy.optimize import curve_fit
np.set_printoptions(precision=8, threshold=1000000, linewidth=1000000, suppress=True)

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
def calcEph(k1, k2):

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

# ------------------------------------------------------------------
# Call the above functions and print the results
# ------------------------------------------------------------------

# set up example kpoints in crystal coordinates
k1 = np.array([[0.0, 0.0, 0.]])
k2 = np.array([[-0.375, 0.375, 0.]])

# Get e-ph properties
g, omegaPh, E1, E2  = calcEph(k2,k1)


# ===================================================================
# We now run the SVD and get the table that contains the maxSIngularVlaues, 
# expoCoeff, and the termsToOnePercent all in parralel.
# ==================================================================

#twoDimMatrix = HePhWannier[;, ;, m ,n ,v]

# Exponential decay function
def expDecay(x, a, b, c):
    return a * np.exp(-b * x) + c

# Function to find max value and terms till 1% of max
def findMaxAndOnePercent(array):
    #Find the max first 
    maxValue = np.max(array)

    #Calculate the threshold for 1% of maxValue
    threshold = maxValue * 0.01

    # Count the no. of Elements to reach 1% of maxValue
    termsToOnePercent = np.sum(array >= threshold)

    return maxValue, termsToOnePercent

def computeSVD(v, n, m, T):
    try:
        twoDimMatrix = HePhWannier[: ,: ,v ,n ,m]
        UMatrix, SingValMatrix, VTransposeMatrix = np.linalg.svd(twoDimMatrix, full_matrices=False)

        # Exponential decay fit 
        x = np.arange(len(SingValMatrix))
        popt, WeDontCareAboutThisTerm = curve_fit(expDecay, x, SingValMatrix, p0=(1, 1, 1))
        a, expoCoeff, c = popt

        # Calculate max singular value and terms to 1% of max
        maxSingularValue, termsToOnePercent = findMaxAndOnePercent(SingValMatrix)

        return (v, n, m, maxSingularValue, termsToOnePercent, expoCoeff)
    
    except Exception as e:
        # print(f"Error processing slice k={k}, l={l}, m={m}: {e}")
        return None 

# ThreadPoolExecutor will run parallel execution for computeSVD on each 2D slice.

results = []

with concurrent.futures.ThreadPoolExecutor() as executor:
    # Prepare the list of tasks
    futures = [executor.submit(computeSVD, v, n, m, HePhWannier) for v in range(9) for n in range(8) for m in range(8)]

    # Process the rest of the results as they complete
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        if result is not None:
            results.append(result)

# Create a DataFrame from the results
df = pd.DataFrame(results, columns=['v', 'n', 'm', 'maxSingularValue', 'termsToOnePercent', 'expoCoeff'])

# Display the DataFrame
#print(df.head())
#df.to_csv("Readmev2")

# Create a DataFrame from the results
# df = pd.DataFrame(results, columns=['k', 'l', 'm', 'max_singular_value', 'terms_to_one_percent', 'exponential_decay'])

# Sort the DataFrame by columns 'k', 'l', and 'm' in ascending order
df_sorted = df.sort_values(by=['v', 'n', 'm'], ascending=[True, True, True])

# Reset the index if you want a clean index
df_sorted.reset_index(drop=True, inplace=True)

# Display the sorted DataFrame
print(df_sorted.head())
print(f"Total number of results: {len(df_sorted)}")    

# Save the DataFrame to a Pickle file
df_sorted.to_pickle('svd_results_redone.pkl')




# print the result if we want to
#np.set_printoptions(precision=3, threshold=1000000, linewidth=1000000, suppress=False)
#g = (g.real**2 + g.imag**2) * 27.2114**2
#g[g < 1e-14] = 0 # remove super small matrix elements
#print(g[0,0,:,:,:])
