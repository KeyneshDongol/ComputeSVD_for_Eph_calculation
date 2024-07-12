# Save the following to WannierEph.py:
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2
# from scipy.linalg import svd
import sys
import h5py
import inspect
# np.set_printoptions(precision=8, threshold=1000000, linewidth=1000000, suppress=True)

debug = False  
def Print(someVariable):
    if debug:
        # Get the name of the variable
        frame = inspect.currentframe().f_back
        var_name = None
        for name, val in frame.f_locals.items():
            if val is someVariable:
                var_name = name
                break
        
        if var_name is None:
            var_name = 'unknown variable'

        if hasattr(someVariable, 'shape'):
            print(f"The shape of {var_name} is: {someVariable.shape}")
        else:
            print(f"{var_name} does not have a shape attribute")

# ====================================================================
# Here we are reading in data output by JDFTx, no modification needed
# ====================================================================

# Read the MLWF cell map, weights and Hamiltonian:
cellMap = np.loadtxt("wannier.mlwfCellMap")[:,0:3].astype(int)
Print(cellMap)


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
Print(Hwannier)


# Read phonon dispersion relation:
cellMapPh = np.loadtxt('totalE.phononCellMap', usecols=[0,1,2]).astype(int)
Print(cellMapPh)

nCellsPh = cellMapPh.shape[0]
omegaSqR = np.fromfile('totalE.phononOmegaSq')  # just a list of numbers
nModes = int(np.sqrt(omegaSqR.shape[0] // nCellsPh))
omegaSqR = omegaSqR.reshape((nCellsPh, nModes, nModes)).swapaxes(1,2)
Print(omegaSqR)

# Read e-ph matrix elements
cellMapEph = np.loadtxt('wannier.mlwfCellMapPh', usecols=[0,1,2]).astype(int)
Print(cellMapEph)
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

    Print(U1)
    Print(U2)

    # Phonons for all pairs pf k1 - k2:
    omegaPh, Uph = calcPh(k1[:,None,:] - k2[None,:,:])
    Print(Uph)

    # E-ph matrix elements for all pairs of k1 - k2:
    phase1 = np.exp((2j * np.pi) * np.dot(k1,cellMapEph.T))
    phase2 = np.exp((2j * np.pi) * np.dot(k2,cellMapEph.T))
    Print(cellMapEph.T)

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
g, omegaPh, E1, E2  = calcEph(k2,k1,HePhWannier)



# print(HePhWannier.shape)
# print(g[:,:,0,0,0])





# Step 1: Transpose the array to move dim1 and dim2 to the last two positions
data = np.transpose(HePhWannier, (2, 3, 4, 0, 1))  # Shape: (v, m, n, dim1, dim2)
U,S,Vt = np.linalg.svd(data, full_matrices=False)




# sum_of_singular = np.sum(S[0, 0, 0, :]) #0.00018920037589969
# print("Sum of singular values: ",sum_of_singular)

# #FInd Singular max
# lastCol = S[:, :, :, -1]
# maxSingular = np.max(lastCol)
# print(lastCol)

# exit()

# how to find 1% 
#maxKeptValue = np.max(S)
#indices = np.where(S>maxKeptValue*0.01)

# Define the truncation value
target = 50 # Truncate to the top 3 singular values



#print("U is:\n\n", U)

# Utest = U[1,1,1, :, :]

# print("U shape is:  ", U.shape) #U shape is 
# print("S shape is:  ", S.shape)
# print("Vt shape is:  ", Vt.shape)


truncated_U = U[ :, :, :, : , :target]
truncated_S = S[ :, :, :, :target]
truncated_Vt = Vt[ :, :, :, :target, :]

# print("Shape of U truncated area", truncated_U.shape)
# print("Shape of S truncated area", truncated_S.shape)
# print("Shape of Vt truncated area", truncated_Vt.shape)


fft_U = fft2(truncated_U, axes=(-2,-1))
fft_Vt = fft2(truncated_Vt, axes=(-2,-1))

# print(fft_U.shape)
# print(fft_Vt.shape)


truncatedHePhWannier_transposed = np.einsum('...ij,...j,...jk->...ik', truncated_U, truncated_S, truncated_Vt)
Print(truncatedHePhWannier_transposed)

# print("Truncated recontructed HePhWannier shape: ", truncatedHePhWannier_transposed.shape)

truncatedHePhWannier = np.transpose(truncatedHePhWannier_transposed, (3,4,0,1,2))
Print(truncatedHePhWannier_transposed)
# SVD the elph matrix elements, get S_gamma, U_gamma(R1), V_gamma(R2)
# Cut down to only certain number of gamma, S_NC, U_NC(R1), V_NC(R2) ... U(x, a, b, R, gamma=Nc)
# now Fourier transform U and V -> Uk, Vk
# reconstruct the gKk matrix from transposed Uk and Vk 
# use unitary rotation matrices by einsum function
# compare output to old function 

def calcEph_truncated(k1, k2):

    # Electrons:
    E1, U1 = calcE(k1)
    E2, U2 = calcE(k2)

    # Phonons for all pairs pf k1 - k2:
    omegaPh, Uph = calcPh(k1[:,None,:] - k2[None,:,:])

    # E-ph matrix elements for all pairs of k1 - k2:
    phase1 = np.exp((2j * np.pi) * np.dot(k1,cellMapEph.T))
    phase2 = np.exp((2j * np.pi) * np.dot(k2,cellMapEph.T))

    arg = np.dot(k1,cellMapEph.T)
    phasePrint = phase1.flatten()

    normFac = np.sqrt(0.5 / np.maximum(omegaPh,1e-6))
    normFac = np.ones(normFac.shape)
    g = np.einsum(
        'kKy, kac, Kbd, kKxy, kr, KR, rRxab -> kKycd',
        normFac, U1.conj(), U2, Uph, phase1.conj(), phase2, truncatedHePhWannier,
        optimize='optimal'
    )
    #    truncated_U_k = np.einsum(            # fourier transform U(x, a, b, R, gamma=Nc)
    #    'KR, xabg, xabRg -> xabKg',
    #    phase2, truncated_S, truncated_U_R,
    #    optimize='optimal'
    #)
    #     gFinal = np.einsum(       # return this 
    #    'kKy, kac, Kbd, kKxy, kKxab -> kKycd',
    #    normFac, U1.conj(), U2, Uph, gK, # gK --> this already transformed to k space
    #    optimize='optimal'
    #)

    #    transpose = np.einsum(            # fourier transform U(x, a, b, R, gamma=Nc)
    #    'ba -> ab',
    #    phase2,
    #    optimize='optimal'
    #)

    return g, omegaPh, E1, E2

gNew, _, _, _ = calcEph_truncated(k2,k1)




print("\n Value of new G is: ", np.absolute(gNew[0,0,:,1,1]),"\n The new value of G is: ",(np.absolute(gNew[0,0,:,1,1]) -np.absolute((g[0,0,:,1,1]))))
# print("The old value of G is: ",   (g[0,0,:,1,1])*(g[0,0,:,1,1]).conj())
