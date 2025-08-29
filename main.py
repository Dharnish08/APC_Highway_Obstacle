print('Hello!')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import casadi as ca

# ==============================================================================
# 1. SETUP: Parameters and Initial Conditions
# ==============================================================================
def get_params():
    """
    Defines all model and simulation parameters in a dictionary.
    """
    params = {}
    
    # Model parameters from Table 1
    params['l_car'] = 3.0   # [m] length of car
    params['w_car'] = 2.0   # [m] width of car
    # Assuming CoG is in the middle (lr = lf) as stated
    params['l_r'] = params['l_car'] / 2.0
    params['l_f'] = params['l_car'] / 2.0
    
    # Environment parameters
    params['w_lane'] = 4.0  # [m] lane width
    params['obs_x'] = 12.0 # [m] x-position of obstacle
    params['obs_y'] = 2.0   # [m] y-position of obstacle (center of right lane)
    params['obs_r'] = 1.0   # [m] radius of obstacle
    params['obs_safe_r'] = params['obs_r'] + 1.2 # [m] radius + safety margin
    
    # Initial state x0 = [x, y, psi, v]
    # Convert 120 km/h to m/s
    v0_kmh = 120.0
    params['x0'] = np.array([0.0, 2.5, 0.0, v0_kmh * 1000 / 3600]) 
    
    # Initial input u0 = [a, delta_f]
    params['u0'] = np.array([0.0, 0.0])

    # State and input bounds from Table 2
    params['psi_min'], params['psi_max'] = -np.pi/2, np.pi/2
    v_max_kmh = 130.0
    params['v_min'], params['v_max'] = 0.0, v_max_kmh * 1000 / 3600
    params['a_min'], params['a_max'] = -10.0, 3.0
    params['delta_f_min'], params['delta_f_max'] = -np.pi/2, np.pi/2
    
    # MPC parameters
    params['N'] = 20       # Prediction horizon (N=3 is too short, see discussion)
    params['T_s'] = 0.05    # [s] Sampling time
    params['N_sim'] = 30   # Number of simulation steps

    # Input rate change limits (for MPC subtask iii)
    params['delta_u_max'] = np.array([1.0, np.pi/20]) # [a_rate, delta_f_rate]
    
    return params

# ==============================================================================
# 2. SYSTEM MODEL
# ==============================================================================
def get_kinematic_model(params):
    """
    Defines the kinematic bicycle model using CasADi symbolic variables.
    """
    # Unpack parameters
    l_r = params['l_r']
    l_f = params['l_f']
    
    # Define symbolic states
    x_pos = ca.SX.sym('x_pos')
    y_pos = ca.SX.sym('y_pos')
    psi = ca.SX.sym('psi')
    v = ca.SX.sym('v')
    states = ca.vertcat(x_pos, y_pos, psi, v)
    n_states = states.numel()

    # Define symbolic controls
    a = ca.SX.sym('a')
    delta_f = ca.SX.sym('delta_f')
    controls = ca.vertcat(a, delta_f)
    n_controls = controls.numel()

    # Define the model equations
    # Side-slip angle beta
    beta = ca.arctan(l_r / (l_r + l_f) * ca.tan(delta_f))
    
    # State derivatives (RHS of ODE)
    x_dot = v * ca.cos(psi + beta)
    y_dot = v * ca.sin(psi + beta)
    psi_dot = (v / l_r) * ca.sin(beta)
    v_dot = a
    
    rhs = ca.vertcat(x_dot, y_dot, psi_dot, v_dot)

    # Create a CasADi function for the model
    model_func = ca.Function('f', [states, controls], [rhs])
    
    return model_func, states, controls

# ==============================================================================
# 3. PART 1: OPEN-LOOP SIMULATION
# ==============================================================================
def run_open_loop_simulation(params, model_func, states, controls):
    """
    Runs the open-loop simulation as described in the first mandatory task.
    """
    print("--- Running Open-Loop Simulation ---")
    
    # Simulation setup
    T_s = params['T_s']
    N_sim = params['N_sim']
    x0 = params['x0']
    
    # Create an integrator
    dae = {'x': states, 'p': controls, 'ode': model_func(states, controls)}
    opts = {'tf': T_s, 'simplify': True, 'number_of_finite_elements': 4}
    F = ca.integrator('F', 'collocation', dae, opts)

    # Prepare history arrays
    x_history = np.zeros((N_sim + 1, states.numel()))
    u_history = np.zeros((N_sim, controls.numel()))
    x_history[0, :] = x0
    
    # Inputs for the two halves of the simulation
    u1 = np.array([0.1, 5e-4 * np.pi])
    u2 = np.array([-0.1, -5e-4 * np.pi])
    
    # Simulation loop
    current_x = x0
    for i in range(N_sim):
        if i < N_sim / 2:
            current_u = u1
        else:
            current_u = u2
            
        # Integrate one step
        res = F(x0=current_x, p=current_u)
        current_x = np.array(res['xf']).flatten()
        
        # Store results
        u_history[i, :] = current_u
        x_history[i+1, :] = current_x

    print("Open-loop simulation finished.")
    
    # System behavior explanation:
    # First half (t=0 to 2.5s): Positive acceleration and a slight positive steering angle.
    # The car accelerates and starts turning left (positive y direction).
    # Second half (t=2.5s to 5s): Negative acceleration and a slight negative steering angle.
    # The car decelerates and starts turning back to the right.
    # The trajectory shows a gentle S-curve.

    return x_history, u_history

# ==============================================================================
# 4. PART 2: MPC CONTROLLER SIMULATION
# ==============================================================================
def run_mpc_simulation(params, model_func, states, controls):
    """
    Runs the MPC simulation, incorporating all subtasks.
    """
    print("--- Running MPC Simulation ---")

    # --- MPC Configuration ---
    USE_INPUT_RATE_CONSTRAINT = True  # Subtask iii
    USE_OBSTACLE_AVOIDANCE = True     # Subtask iv
    USE_CAR_SHAPE_APPROX = True       # Subtask iv extension
    
    # --- MPC Setup ---
    T_s = params['T_s']
    N = params['N']
    N_sim = params['N_sim']
    n_states = states.numel()
    n_controls = controls.numel()

    # Create an integrator for MPC
    dae = {'x': states, 'p': controls, 'ode': model_func(states, controls)}
    opts = {'tf': T_s}
    F = ca.integrator('F', 'rk', dae, opts) # RK4 is faster for MPC

    # Create the optimization problem
    opti = ca.Opti()
    
    # Decision variables
    X = opti.variable(n_states, N + 1)
    U = opti.variable(n_controls, N)
    
    # Parameters
    x0_param = opti.parameter(n_states)      # Initial state
    u_prev_param = opti.parameter(n_controls) # Previous input

    # --- Objective Function ---
    cost = 0
    
    # Weights for the cost function terms
    Q_y = 10.0     # Penalty on y-position deviation
    Q_v = 1.0      # Penalty for low velocity (to encourage movement)
    R_u = 0.1      # Penalty on control inputs
    R_delta_u = 100.0  # Penalty on control input rate of change
    
    y_ref = params['w_lane'] / 2.0  # Center of the right lane

    for k in range(N):
        # Subtask ii: Keep car in center of the right lane
        cost += Q_y * (X[1, k] - y_ref)**2
        
        # Subtask i: Minimize traveling time (by maximizing forward velocity)
        cost -= Q_v * X[3, k] 
        
        # Penalize control effort
        cost += R_u * ca.sumsqr(U[:, k])
        
        # Subtask iii: Penalize input rate of change
        if k > 0:
            cost += R_delta_u * ca.sumsqr(U[:, k] - U[:, k-1])
        else:
            cost += R_delta_u * ca.sumsqr(U[:, 0] - u_prev_param)

    opti.minimize(cost)

    # --- Constraints ---
    for k in range(N):
        # Dynamics constraint (system model)
        opti.subject_to(X[:, k+1] == F(x0=X[:, k], p=U[:, k])['xf'])

        # Subtask iv: Obstacle avoidance
        if USE_OBSTACLE_AVOIDANCE:
            obs_x, obs_y = params['obs_x'], params['obs_y']
            
            # Approximate car shape with a bounding circle
            if USE_CAR_SHAPE_APPROX:
                car_radius = np.sqrt((params['l_car']/2)**2 + (params['w_car']/2)**2)
                effective_radius = params['obs_safe_r'] + car_radius
            else: # Model car as a point (as per hint for i-iii)
                effective_radius = params['obs_safe_r']

            dist_sq = (X[0, k] - obs_x)**2 + (X[1, k] - obs_y)**2
            opti.subject_to(dist_sq >= effective_radius**2)

    # State and control bounds
    opti.subject_to(opti.bounded(params['v_min'], X[3, :], params['v_max']))
    opti.subject_to(opti.bounded(params['psi_min'], X[2, :], params['psi_max']))
    
    # Subtask iv: Stay on the highway (2 lanes)
    opti.subject_to(opti.bounded(0, X[1, :], 2 * params['w_lane']))
    
    opti.subject_to(opti.bounded(params['a_min'], U[0, :], params['a_max']))
    opti.subject_to(opti.bounded(params['delta_f_min'], U[1, :], params['delta_f_max']))

    # Subtask iii: Input rate of change hard constraints
    if USE_INPUT_RATE_CONSTRAINT:
        delta_u_max = params['delta_u_max']
        opti.subject_to(opti.bounded(-delta_u_max[0], U[0, 0] - u_prev_param[0], delta_u_max[0]))
        opti.subject_to(opti.bounded(-delta_u_max[1], U[1, 0] - u_prev_param[1], delta_u_max[1]))
        for k in range(N - 1):
            opti.subject_to(opti.bounded(-delta_u_max[0], U[0, k+1] - U[0, k], delta_u_max[0]))
            opti.subject_to(opti.bounded(-delta_u_max[1], U[1, k+1] - U[1, k], delta_u_max[1]))

    # Initial condition constraint
    opti.subject_to(X[:, 0] == x0_param)
    
    # --- Solver Setup ---
    # Suppress IPOPT output for cleaner terminal
    p_opts = {"expand": True} 
    s_opts = {"max_iter": 1000, "print_level": 0, "sb": "yes"}
    opti.solver('ipopt', p_opts, s_opts)
    
    # --- MPC Simulation Loop ---
    x0 = params['x0']
    u0 = params['u0']
    
    x_history = np.zeros((N_sim + 1, n_states))
    u_history = np.zeros((N_sim, n_controls))
    x_history[0, :] = x0
    
    # Store predicted trajectories for one step for plotting
    predicted_X_hist = []

    current_x = x0
    current_u = u0
    
    for i in range(N_sim):
        print(f"Simulating step {i+1}/{N_sim}", end='\r')
        
        # Set parameter values
        opti.set_value(x0_param, current_x)
        opti.set_value(u_prev_param, current_u)
        
        # Solve the NLP
        try:
            sol = opti.solve()
            
            # Get the first optimal control input
            u_optimal = sol.value(U[:, 0])
            
            # Store predicted trajectory for plotting
            if i == 50: # Choose a step to visualize prediction
                 predicted_X_hist = sol.value(X)
        
        except RuntimeError:
            print(f"\nOptimization failed at step {i}. Using previous control input.")
            u_optimal = current_u # Failsafe

        # Apply the control to the real system (integrator)
        res = F(x0=current_x, p=u_optimal)
        current_x = np.array(res['xf']).flatten()
        current_u = u_optimal

        # Store results
        u_history[i, :] = current_u
        x_history[i+1, :] = current_x
        
        # Set initial guess for next iteration for faster convergence
        opti.set_initial(X, np.roll(sol.value(X), -1, axis=1))
        opti.set_initial(U, np.roll(sol.value(U), -1, axis=1))

    print("\nMPC simulation finished.")

    # Discussion on N=3:
    # A prediction horizon of N=3 with T_s=0.05s means the controller only sees 0.15s into the future.
    # At 120 km/h (~33 m/s), the car travels ~5m in this time. This is extremely myopic.
    # The car would only "see" the obstacle when it's very close, leading to extremely aggressive,
    # jerky, and likely infeasible maneuvers. It would probably fail to avoid the obstacle.
    # Minimal N: To plan a smooth lane change, the car needs to see the obstacle from at least 
    # 50-60m away. 60m / (33 m/s * 0.05 s/step) ≈ 36 steps. A horizon of N=20 to N=40 is more
    # appropriate. N=20 is a reasonable compromise for performance. A larger N leads to
    # smoother, more optimal paths but increases computation time.

    return x_history, u_history, predicted_X_hist


# ==============================================================================
# 5. PLOTTING
# ==============================================================================
def plot_results(params, x_history, u_history, predicted_X=None, sim_type='MPC'):
    """
    Plots the simulation results.
    """
    # Unpack data
    t = np.arange(x_history.shape[0]) * params['T_s']
    x_pos, y_pos, psi, v = x_history.T
    
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f'Self-Driving Car Simulation Results ({sim_type})', fontsize=16)

    # Plot 1: Highway Trajectory
    ax1 = fig.add_subplot(2, 2, (1, 3))
    ax1.plot(x_pos, y_pos, 'b-', label='Car Trajectory (CoG)')
    
    # Plot predicted trajectory if available
    if predicted_X is not None and len(predicted_X) > 0:
        ax1.plot(predicted_X[0, :], predicted_X[1, :], 'g--', label='Predicted Trajectory')
    
    # Plot highway lanes
    ax1.axhline(0, color='k', linestyle='-')
    ax1.axhline(params['w_lane'], color='k', linestyle='--')
    ax1.axhline(2 * params['w_lane'], color='k', linestyle='-')
    
    # Plot obstacle
    obstacle = patches.Circle((params['obs_x'], params['obs_y']), radius=params['obs_r'], fc='r', label='Obstacle')
    ax1.add_patch(obstacle)
    obstacle_safe = patches.Circle((params['obs_x'], params['obs_y']), radius=params['obs_safe_r'], ec='r', fill=False, linestyle='--', label='Safety Margin')
    ax1.add_patch(obstacle_safe)

    ax1.set_xlabel('Position x [m]')
    ax1.set_ylabel('Position y [m]')
    ax1.set_title('Highway Obstacle Avoidance')
    ax1.legend()
    ax1.grid(True)
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_ylim(-1, 2 * params['w_lane'] + 1)
    
    # Plot 2: States over time
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(t, x_pos, label='x [m]')
    ax2.plot(t, y_pos, label='y [m]')
    ax2.plot(t, psi, label='psi [rad]')
    ax2.plot(t, v, label='v [m/s]')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('State Values')
    ax2.set_title('States vs. Time')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Inputs over time
    ax3 = fig.add_subplot(2, 2, 4)
    time_u = t[:-1]
    ax3.step(time_u, u_history[:, 0], where='post', label='a [m/s^2]')
    ax3.step(time_u, u_history[:, 1], where='post', label='delta_f [rad]')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Input Values')
    ax3.set_title('Inputs vs. Time')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# ==============================================================================
# 6. MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':
    # Get parameters and model
    params = get_params()
    model_func, states, controls = get_kinematic_model(params)
    
    # --- CHOOSE SIMULATION TO RUN ---
    # simulation_to_run = 'open_loop'
    simulation_to_run = 'mpc'

    if simulation_to_run == 'open_loop':
        x_hist, u_hist = run_open_loop_simulation(params, model_func, states, controls)
        plot_results(params, x_hist, u_hist, sim_type='Open-Loop')
    elif simulation_to_run == 'mpc':
        x_hist, u_hist, predicted_X = run_mpc_simulation(params, model_func, states, controls)
        plot_results(params, x_hist, u_hist, predicted_X, sim_type='MPC')
    else:
        print("Invalid simulation choice. Please choose 'open_loop' or 'mpc'.")