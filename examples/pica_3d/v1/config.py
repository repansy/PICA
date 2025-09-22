# config.py

# --- Simulation Scenario ---
# 'antipodal_sphere': Agents start on a sphere and travel to the opposite point.
# 'random': Agents start at random positions with random goals.
SCENARIO = 'antipodal_sphere' #'hive_takeoff'\'antipodal_sphere'

# --- Simulation Parameters ---
NUM_AGENTS = 2
TIMESTEP = 0.1  # seconds
SIMULATION_TIME = 300  # seconds
WORLD_SIZE = (50, 50, 50) # meters (x, y, z)

# --- Agent Physical Properties ---
AGENT_RADIUS = 0.5  # meters
MAX_SPEED = 2.0     # meters/second
# A practical way to model state uncertainty (spirit of PRVO).
# This buffer is added to the radius for all collision checks.
UNCERTAINTY_BUFFER = 0.15 # meters

# --- PICA Algorithm Parameters ---
# The number of highest-risk neighbors to perform full PICA optimization on.
PICA_K = 5 
# Damping factor for alpha updates to prevent oscillations (0 to 1).
PICA_BETA_DAMPING = 0.25 
# Small epsilon for numerical stability in analytical solver and projections.
PICA_EPSILON = 1e-5

# --- Risk Assessment Weights ---
# How much weight to give distance vs. time-to-collision.
RISK_W_DIST = 0.4
RISK_W_TTC = 0.6
# Time horizon for collision checks and VO cone truncation.
TTC_HORIZON = 1.0 # seconds 10

# --- Density Calculation Parameters ---
# The "influence radius" for the density calculation.
DENSITY_SIGMA = 5.0 
# Damping factor for smoothing the density value over time (0 to 1).
DENSITY_BETA_SMOOTHING = 0.7 

# --- Visualization ---
VISUALIZE = True
# Update plot every N timesteps to speed up simulation.
PLOT_FREQUENCY = 5 
