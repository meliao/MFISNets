"""
Defining the strings we use as keys in our HDF5 files.
"""

### Data file name constants
X_VALS = "x_vals"
RHO_VALS = "rho_vals"
THETA_VALS = "theta_vals"
SAMPLE_COMPLETION = "sample_completion"
FILE_COMPLETION = "file_completion"
SEED = "seed"
CONTRAST = "contrast"
NUM_SHAPES = "num_shapes"
GAUSSIAN_LPF_PARAM = "gaussian_lpf_param"
BACKGROUND_MAX_FREQ = "background_max_freq"
BACKGROUND_MAX_RADIUS = "background_max_radius"
Q_CART = "q_cart"
Q_POLAR = "q_polar"
D_RS = "d_rs"
D_MH = "d_mh"
M_VALS = "m_vals"
H_VALS = "h_vals"
Q_POLAR_LPF = "q_polar_lpf"
Q_CART_LPF = "q_cart_lpf"
NU_SF = "nu_sf"
OMEGA_SF = "omega_sf"

# Define which keys are required for which operations.
KEYS_FOR_TRAINING_SAMPLES_MEAS = [Q_POLAR_LPF, D_MH, SAMPLE_COMPLETION]
KEYS_FOR_TRAINING_SAMPLES_ALL = [Q_POLAR_LPF, D_MH, SAMPLE_COMPLETION, Q_POLAR]
KEYS_FOR_TRAINING_METADATA = [X_VALS, RHO_VALS, THETA_VALS, M_VALS, H_VALS]

KEYS_FOR_EXPERIMENT_INFO_OUT = [
    X_VALS,
    RHO_VALS,
    THETA_VALS,
    SAMPLE_COMPLETION,
    CONTRAST,
    NUM_SHAPES,
    GAUSSIAN_LPF_PARAM,
    BACKGROUND_MAX_FREQ,
    BACKGROUND_MAX_RADIUS,
    SEED,
    FILE_COMPLETION,
]
