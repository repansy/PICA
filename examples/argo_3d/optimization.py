'''
意图: 
    将复杂的数学优化逻辑完全封装在此。我们利用scipy.optimize.minimize这个强大的工业级求解器，
    而不是手写梯度下降，这样更稳定、更高效。cost_function内部清晰地展示了ARGO-3D的成本构成。

改进：
    风险函数拆分: 将动态和静态障碍物的风险计算分开，使逻辑更清晰。

    非对称权重实现: _get_asymmetric_weight提供了一个具体的（尽管是简化的）实现，
    它根据优先级和路权（这里用了简化的2D叉乘来判断左右）来动态调整成本函数中的权重。

    静态障碍物成本: 在成本函数中加入了对静态障碍物的惩罚项。

    _calculate_dynamic_risk中暂时使用了一个简化的、非概率性的风险函数。
    要实现完整的ARGO-3D，需要在这里替换为之前讨论的、基于协方差矩阵和概率积分的复杂计算。
'''


# agent/optimization.py
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
from typing import Dict, List
from utils.argo_structures import State, BeliefState, AgentConfig, Obstacle

# --- Constants for Asymmetry ---
PRIORITY_FACTOR = 1.5      # How much more we penalize risk from higher priority agents
RIGHT_OF_WAY_FACTOR = 2.0  # How much more we penalize violating right-of-way
OBSTACLE_WEIGHT = 10.0     # Static obstacles are very important to avoid
TIME_HORIZON = 5.0         # Look 5 seconds into the future for collision checks
CONFIDENCE_LEVEL = 0.99    # Corresponds to P(collision) threshold of 1 - 0.99 = 1%

# --- New Comprehensive Social Factor Calculation ---
def _calculate_advanced_social_factors(self_state: State, self_config: AgentConfig,
                                     neighbor_beliefs: Dict[int, BeliefState], config_params: Dict):
    """
    Calculates advanced social metrics based on the provided formulas.
    Returns:
        R_A (float): The final comprehensive responsibility factor for self.
        W (np.ndarray): The 3x3 final scaling matrix for self.
    """
    num_neighbors = len(neighbor_beliefs)
    if num_neighbors == 0:
        return 0.5, np.eye(3) # Default: 50% responsibility, no scaling

    # --- Definition 1 & 3: Crowdedness Calculation ---
    # We assume R is sensor range and omega_k = 1 for simplicity.
    R = config_params['SENSOR_RANGE']
    C_A_vec = np.zeros(3)
    sum_omega = num_neighbors
    
    for belief in neighbor_beliefs.values():
        p_rel = belief.mean[0:3] - self_state.pos
        C_A_vec += p_rel
    
    C_A_vec = 1.0 - (C_A_vec / (R * sum_omega))
    C_A = np.mean(C_A_vec) # Definition 3: Comprehensive Crowdedness

    # --- Definition 4: Individual Responsibility Weight (alpha) ---
    alpha_self = max(config_params['ALPHA_0'] * (1 - config_params['GAMMA'] * C_A), config_params['ALPHA_MIN'])

    # --- Definition 5 & 7: Pairwise and Comprehensive Responsibility ---
    # To calculate R_A, we need to estimate alpha for our neighbors.
    # We'll approximate our neighbors' crowdedness based on their local density from our POV.
    alpha_pairwise_list = []
    
    for nid, belief in neighbor_beliefs.items():
        # A simple approximation for neighbor's crowdedness
        # Could be improved with more complex "theory of mind"
        C_neighbor = C_A # Simple approximation: assume they feel as crowded as we do
        alpha_neighbor = max(config_params['ALPHA_0'] * (1 - config_params['GAMMA'] * C_neighbor), config_params['ALPHA_MIN'])

        # Definition 5: Pairwise Responsibility
        alpha_A_given_B = alpha_neighbor / (alpha_self + alpha_neighbor)
        alpha_pairwise_list.append(alpha_A_given_B)
        
    # Definition 6 & 7 (simplified): R_A is the average of our pairwise responsibilities
    R_A = np.mean(alpha_pairwise_list) if alpha_pairwise_list else 0.5
    
    # --- Definition 2: Main Flow Direction (d_flow) ---
    neighbor_velocities = [b.mean[3:6] for b in neighbor_beliefs.values()]
    v_mean = np.mean(neighbor_velocities, axis=0)
    cov_matrix = np.zeros((3, 3))
    for v_k in neighbor_velocities:
        diff = (v_k - v_mean).reshape(3, 1)
        cov_matrix += diff @ diff.T
    cov_matrix /= num_neighbors
    
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    d_flow = eigenvectors[:, np.argmax(eigenvalues)]

    # --- Definition 8 & 9: Final Scaling Matrix W ---
    kappa = config_params['KAPPA']
    lambda_ = config_params['LAMBDA']
    
    W0_diag = 1.0 + kappa * C_A_vec
    W0 = np.diag(W0_diag)
    
    d_flow = d_flow.reshape(3, 1)
    W_flow = (1 - lambda_) * np.eye(3) + lambda_ * (d_flow @ d_flow.T)
    
    W = W0 @ W_flow
    
    return R_A, W


def _calculate_probabilistic_risk(v_candidate: np.ndarray, 
                                 self_state: State, 
                                 self_config: AgentConfig, 
                                 belief: BeliefState, 
                                 asymmetric_T_matrix: np.ndarray) -> float:
    """
    Calculates the collision probability P(collision) for a dynamic agent (neighbor).
    This is the core of the probabilistic risk assessment.
    """
    # 1. Calculate relative state mean
    v_rel = v_candidate - belief.mean[3:6]
    p_rel = belief.mean[0:3] - self_state.pos

    # 2. Calculate Time to Closest Approach (t_cpa) on the mean trajectory
    v_rel_sq_norm = np.dot(v_rel, v_rel)
    if v_rel_sq_norm < 1e-6: # Agents are almost stationary relative to each other
        t_cpa = 0.0
    else:
        t_cpa = -np.dot(p_rel, v_rel) / v_rel_sq_norm
    
    # We only care about collisions within the time horizon
    if not (0 <= t_cpa <= TIME_HORIZON):
        return 0.0

    # 3. Calculate state at CPA
    p_rel_at_cpa = p_rel + v_rel * t_cpa

    # 4. Calculate transformed covariance at CPA
    # Assuming self state has zero uncertainty for simplicity in this step.
    # A more advanced model would include self_covariance.
    # Also assume covariance does not grow significantly over the short t_cpa.
    # This is a reasonable approximation for real-time performance.
    sigma_rel = belief.covariance[0:3, 0:3] # Get the 3x3 position covariance
    sigma_rel_transformed = asymmetric_T_matrix @ sigma_rel @ asymmetric_T_matrix.T
    
    # 5. Mahalanobis Distance and Probability Calculation
    # We are checking if the origin (0,0,0) is inside the uncertainty ellipsoid
    # defined by the combined radius.
    combined_radius = self_config.radius + belief.config.radius
    
    try:
        inv_sigma = np.linalg.inv(sigma_rel_transformed)
    except np.linalg.LinAlgError:
        # If matrix is singular, it's highly certain; a small perturbation can fix it.
        inv_sigma = np.linalg.inv(sigma_rel_transformed + np.eye(3) * 1e-6)

    # Mahalanobis distance squared: measures how many std devs p_rel_at_cpa is from the origin (collision point)
    mahalanobis_dist_sq = p_rel_at_cpa.T @ inv_sigma @ p_rel_at_cpa
    
    # This distance follows a Chi-squared distribution with 3 degrees of freedom (for 3D).
    # We can relate this distance to a probability.
    # To avoid calculating the full integral, we use a common approximation:
    # if the Mahalanobis distance is less than a critical value, collision is likely.
    
    # Find the critical value from Chi2 distribution for our confidence level
    # This value represents the "radius" of our confidence ellipsoid in Mahalanobis space.
    chi2_critical_value = chi2.ppf(CONFIDENCE_LEVEL, df=3)
    
    # Scale the critical value by the combined radius to define the collision boundary
    scaled_boundary = chi2_critical_value * (combined_radius**2)

    # A smooth risk function: 0 if far, 1 if Mahalanobis distance is zero.
    if mahalanobis_dist_sq < scaled_boundary:
        # We are inside the high-confidence collision ellipsoid.
        # Risk ramps up from 0 at the boundary to 1 at the center.
        risk = 1.0 - (mahalanobis_dist_sq / scaled_boundary)
        return np.clip(risk, 0, 1)
    
    return 0.0


def _calculate_static_risk(v_candidate: np.ndarray, self_state: State, 
                           self_config: AgentConfig, obstacle: Obstacle) -> float:
    """
    Calculates a deterministic risk value for a static obstacle.
    """
    p_rel = obstacle.pos - self_state.pos
    v_rel = v_candidate
    
    v_rel_sq_norm = np.dot(v_rel, v_rel)
    if v_rel_sq_norm < 1e-6:
        t_cpa = 0.0
    else:
        t_cpa = -np.dot(p_rel, v_rel) / v_rel_sq_norm
        
    if not (0 <= t_cpa <= TIME_HORIZON):
        return 0.0

    dist_sq_at_cpa = np.linalg.norm(p_rel + v_rel * t_cpa)**2
    combined_radius = self_config.radius + obstacle.radius
    
    if dist_sq_at_cpa < combined_radius**2:
        # Simple risk: 1 if collision is certain, 0 otherwise, smoothed.
        risk = 1.0 - (dist_sq_at_cpa / (combined_radius**2))
        return np.clip(risk, 0, 1)
        
    return 0.0


def solve_argo_3d_velocity(current_state: State, pref_velocity: np.ndarray, 
                           neighbor_beliefs: Dict[int, BeliefState], 
                           nearby_obstacles: List[Obstacle], config: AgentConfig, 
                           config_params: Dict, debug_context: Dict = None):
    """
    Solves the non-convex optimization problem to find the best velocity.
    This is the main entry point for the ARGO-3D decision core.
    """
    # Calculate all social factors once at the beginning
    R_A, W_matrix = _calculate_advanced_social_factors(current_state, config, neighbor_beliefs, config_params)
    
    # --- Populate Debug Context (Part 1) ---
    if debug_context is not None:
        debug_context['R_A'] = R_A
        debug_context['W_matrix'] = W_matrix
        debug_context['v_pref'] = pref_velocity
        debug_context['neighbor_debug'] = {} # Prepare a nested dict for per-neighbor data
    
    def cost_function(v_candidate: np.ndarray) -> float:
        # Ensure candidate velocity respects max speed (soft constraint can be used, but hard bounds are better)
        speed = np.linalg.norm(v_candidate)
        if speed > config.max_speed:
            # Add a heavy penalty for exceeding max speed
            return 1e9 * (speed - config.max_speed)

        efficiency_cost = np.linalg.norm(v_candidate - pref_velocity)**2
        safety_cost = 0

        # --- Dynamic Agents ---
        for neighbor_id, belief in neighbor_beliefs.items():
            
            # 1. Determine the asymmetric transformation matrix T
            T_matrix = np.eye(3)
            
            # --- 3D Geometric Asymmetry (Right-of-Way) Logic ---
            p_rel = belief.mean[0:3] - current_state.pos
            
            # Use preferred velocity direction for a stable "forward" vector
            forward_dir = pref_velocity / (np.linalg.norm(pref_velocity) + 1e-6)
            if np.linalg.norm(forward_dir) == 0: # If goal is reached
                forward_dir = np.array([1., 0., 0.]) # Default forward
            
            up_vector = np.array([0., 0., 1.])
            # Handle case where forward is parallel to up
            if np.abs(np.dot(forward_dir, up_vector)) > 0.99:
                up_vector = np.array([0., 1., 0.]) # Use Y-axis as up
            
            right_vector = np.cross(forward_dir, up_vector)
            right_vector /= (np.linalg.norm(right_vector) + 1e-6)

            # If the neighbor is significantly on our right, we should give way.
            # We "give way" by stretching our uncertainty ellipsoid towards them,
            # making a collision seem more probable from our perspective.
            if np.dot(p_rel, right_vector) > config.radius:
                 # Create a scaling matrix that stretches along the p_rel direction
                 p_rel_norm = p_rel / (np.linalg.norm(p_rel) + 1e-6)
                 stretch_factor = RIGHT_OF_WAY_FACTOR
                 # Outer product creates a matrix that scales only along p_rel_norm
                 stretch_matrix = (stretch_factor - 1) * np.outer(p_rel_norm, p_rel_norm)
                 T_matrix = np.eye(3) + stretch_matrix
                 
            # Combine transformations: W warps the space based on flow, T warps based on rules.
            combined_transform = W_matrix @ T_matrix
            
            # 2. Calculate probabilistic risk using the T matrix
            risk = _calculate_probabilistic_risk(v_candidate, current_state, config, belief, combined_transform)
            
            # 3. Calculate priority-based weight
            # weight = 1.0
            weight = 1.0 / (R_A + 0.1) # Avoid division by zero
            if belief.config.priority > config.priority:
                weight *= PRIORITY_FACTOR**(belief.config.priority - config.priority)
            
            # 4. Add to total cost
            safety_cost -= weight * np.log(1.0 - risk + 1e-9)

            # --- Populate Debug Context (Part 2 - inside the loop) ---
            if debug_context is not None:
                debug_context['neighbor_debug'][neighbor_id] = {
                    'weight': weight,
                    'T_matrix': T_matrix,
                    'risk_at_v_pref': risk # _calculate_probabilistic_risk(pref_velocity, ...), # Example
                }
            
        # --- Static Obstacles ---
        for obs in nearby_obstacles:
            risk = _calculate_static_risk(v_candidate, current_state, config, obs)
            safety_cost -= OBSTACLE_WEIGHT * np.log(1.0 - risk + 1e-9)
            
        return efficiency_cost + safety_cost

    # --- Run the Optimization ---
    initial_guess = pref_velocity
    
    # Define speed bounds for the optimizer
    bounds = [(-config.max_speed, config.max_speed)] * 3
    
    result = minimize(
        fun=cost_function,
        x0=initial_guess,
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol': 1e-4, 'maxiter': 50} # Tune optimizer params for performance
    )
    
    # If optimization fails, fall back to a safe behavior (e.g., braking)
    optimal_velocity = result.x if result.success else current_state.vel * 0.5
    
    # --- Populate Debug Context (Part 3 - final result) ---
    if debug_context is not None:
        debug_context['v_optimal'] = optimal_velocity
        debug_context['optim_success'] = result.success
        
    return optimal_velocity
    