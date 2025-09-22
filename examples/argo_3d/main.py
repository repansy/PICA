# main.py
import numpy as np
from tqdm import tqdm
import csv

from simulator.argo_simulator import Simulator
from agent.argo_agent import Agent
from utils.argo_structures import State
import config
from enviroments import visualizer

def main():
    """Main function to run the simulation."""
    print("Initializing ARGO-3D Simulation...")

    # --- Load Scenario ---
    scenario_agents = config.get_scenario_agents()
    scenario_obstacles = config.get_scenario_obstacles()
    
    # --- Create Agent Instances ---
    agents = {}
    agent_ids_sorted = sorted(scenario_agents.keys()) # Ensure consistent order
    for agent_id, params in scenario_agents.items():
        initial_state = State(pos=params["initial_pos"], vel=np.zeros(3))
        agents[agent_id] = Agent(
            id=agent_id,
            initial_state=initial_state,
            config=params["config"],
            goal=params["goal"],
            config_params=config.CONFIG_PARAMS
        )

    # --- Initialize Simulator ---
    sim = Simulator(
        agents=agents,
        obstacles=scenario_obstacles,
        sensor_range=config.SENSOR_RANGE,
        comm_loss_prob=config.COMM_LOSS_PROB
    )
    
    # --- Prepare for Logging ---
    trajectory_data = []
    # Generate the header based on the user's format
    header = []
    for agent_id in agent_ids_sorted:
        header.extend([f'Agent{agent_id}_x', f'Agent{agent_id}_y', f'Agent{agent_id}_z'])
    trajectory_data.append(header)

    # --- Run Simulation Loop ---
    num_steps = int(config.SIM_DURATION_SECONDS / config.SIM_TIME_STEP)
    print(f"Running simulation for {config.SIM_DURATION_SECONDS} seconds ({num_steps} steps)...")
    
    for step in tqdm(range(num_steps)):
        sim.run_step(config.SIM_TIME_STEP)
        
        # Optional: Add visualization hooks here
        '''        
        if step == 50: # At the 50th step (5 seconds in)
            print("Generating visualization for Agent 0...")
            visualizer.plot_velocity_space_risk(sim, agent_id=0, time=sim.time)
        '''
            
        # For example, store agent trajectories for plotting later
        current_step_positions = []
        for agent_id in agent_ids_sorted:
            agent = sim.agents[agent_id]
            current_step_positions.extend(agent.state.pos)
        trajectory_data.append(current_step_positions)
        
        # Check for completion
        if all(agent.is_at_goal for agent in sim.agents.values()):
            print("All agents have reached their goals. Simulation finished.")
            break

    print("Simulation complete.")
    # --- Write Log Data to CSV File ---
    output_filename = "trajectory.csv"
    print(f"Writing trajectory data to {output_filename}...")
    with open(output_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(trajectory_data)
    print("Data successfully saved.")

if __name__ == "__main__":
    main()