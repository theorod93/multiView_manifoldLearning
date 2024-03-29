# Imports
from Functions.ISOMAP import *

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

## Multi-ISOMAP

# Data input
Xinput = Xi
Xinput[0] = Xa
Xinput[1] = Xb
Xinput[2] = Xc
Xinput[3] = Xd
Xinput[4] = Xe
Xinput[5] = Xf

start_time_multiISOMAP = time.time()
# Run algorithm
n_neighbors_input = 50
Y_multiISOMAP = multi_isomap_path(Xinput, 2, n_neighbors_input, 20.0, 1000)
end_time_multiISOMAP = time.time()
# Collect computational time
running_time_multiISOMAP = end_time_multiISOMAP - start_time_multiISOMAP

# Data input
Xinput = Xi
Xinput[0] = Xa
Xinput[1] = Xb
Xinput[2] = Xc
Xinput[3] = Xd
Xinput[4] = Xe
Xinput[5] = Xf

start_time_mISOMAP = time.time()
# Run algorithm
n_neighbors_input = 50
Y_mISOMAP = m_isomap(Xinput, 2, n_neighbors_input, 10.0, 1000)
end_time_mISOMAP = time.time()
# Collect computational time
running_time_mISOMAP = end_time_mISOMAP - start_time_mISOMAP
