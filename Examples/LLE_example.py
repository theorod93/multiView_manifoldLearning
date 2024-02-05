# Imports
from Functions.LLE import *

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

## Multi-LLE

# Data input
Xinput = Xi
Xinput[0] = Xa
Xinput[1] = Xb
Xinput[2] = Xc
Xinput[3] = Xd
Xinput[4] = Xe
Xinput[5] = Xf

start_time_multiLLE = time.time()
# Run algorithm
Y_multiLLE = multiLLE(Xinput, 2, 50, 20.0, 1000)
end_time_multiLLE = time.time()
# Collect computational time
running_time_multiLLE = end_time_multiLLE - start_time_multiLLE

# Data input
Xinput = Xi
Xinput[0] = Xa
Xinput[1] = Xb
Xinput[2] = Xc
Xinput[3] = Xd
Xinput[4] = Xe
Xinput[5] = Xf

start_time_mLLE = time.time()
# Run algorithm
Y_mLLE = mLLE(Xinput, 2, 50, 10.0, 1000)
end_time_mLLE = time.time()
# Collect computational time
running_time_mLLE = end_time_mLLE - start_time_mLLE
