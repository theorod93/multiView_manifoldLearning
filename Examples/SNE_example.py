

# Imports
from Functions.SNE import *

# Load data
Xa = np.loadtxt("caltech7/centrist_caltech7.txt")
Xb = np.loadtxt("caltech7/gabor_caltech7.txt")
Xc = np.loadtxt("caltech7/gist_caltech7.txt")
Xd = np.loadtxt("caltech7/hog_caltech7.txt")
Xe = np.loadtxt("caltech7/lbp_caltech7.txt")
Xf = np.loadtxt("caltech7/wavelet_moments_caltech7.txt")

# Data initialisation
Xi = np.array([[(0,0),(0,0), (0,0)],[(0,0),(0,0), (0,0)], [(0,0),(0,0)],
               [(0,0),(0,0), (0,0), (0,0), (0,0)],
               #[(0,0),(0,0), (0,0), (0,0)], 
               [(0,0),(0,0)]], 
              dtype="object")

## Multi-SNE

# Data input
Xinput = Xi
Xinput[0] = Xa
Xinput[1] = Xb
Xinput[2] = Xc
Xinput[3] = Xd
Xinput[4] = Xe
Xinput[5] = Xf

start_time_multiSNE = time.time()
# Run algorithm
Y_multiSNE = multi_SNE(Xinput, 2, 50, 20.0, 1000)
end_time_multiSNE = time.time()
# Collect computational time
running_time_multiSNE = end_time_multiSNE - start_time_multiSNE

# Data input
Xinput = Xi
Xinput[0] = Xa
Xinput[1] = Xb
Xinput[2] = Xc
Xinput[3] = Xd
Xinput[4] = Xe
Xinput[5] = Xf

start_time_mSNE = time.time()
# Run algorithm
Y_mSNE = mSNE(Xinput, 2, 50, 10.0, 1000)
end_time_mSNE = time.time()
# Collect computational time
running_time_mSNE = end_time_mSNE - start_time_mSNE
