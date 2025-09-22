# main.py
import random
import time
import numpy as np
from agent.pica_agent import Agent
# from agent.orca_agent import OrcaAgent as Agent

from simulator.pica_simulator import Simulator
from utils.pica_structures import Vector3D
import config as cfg
import matplotlib.pyplot as plt

def setup_scenario():
    """Creates agents and their goals based on the configured scenario."""
    agents = []
    center = Vector3D(cfg.WORLD_SIZE[0] / 2, cfg.WORLD_SIZE[1] / 2, cfg.WORLD_SIZE[2] / 2)
    
    # --- SCENARIO 1: Crossroads (tests priority and inertia) ---
    if cfg.SCENARIO == 'crossroads':
        print("Setting up 'crossroads' scenario...")
        num_heavy = cfg.NUM_AGENTS // 2
        # num_agile = cfg.NUM_AGENTS - num_heavy
        
        # Group A: Heavy, high-priority, moving along X-axis
        heavy_inertia = np.diag([0.5, 5.0, 5.0]) # Hard to move sideways/up-down
        for i in range(num_heavy):
            offset = Vector3D(0, random.uniform(-5, 5), random.uniform(-5, 5))
            start_pos = Vector3D(5, center.y, center.z) + offset
            goal_pos = Vector3D(cfg.WORLD_SIZE[0] - 5, center.y, center.z) + offset
            agents.append(Agent(id=i, pos=start_pos, goal=goal_pos, priority=2.0, inertia_matrix=heavy_inertia))
            
        # Group B: Agile, low-priority, moving along Y-axis ?? 
        for i in range(num_heavy, cfg.NUM_AGENTS):
            offset = Vector3D(random.uniform(-5, 5), 0, random.uniform(-5, 5))
            start_pos = Vector3D(center.x, 5, center.z) + offset
            goal_pos = Vector3D(center.x, cfg.WORLD_SIZE[1] - 5, center.z) + offset
            agents.append(Agent(id=i, pos=start_pos, goal=goal_pos, priority=1.0))

    # --- SCENARIO 2: Merge Flow (tests density and risk perception) ---
    elif cfg.SCENARIO == 'merge_flow':
        print("Setting up 'merge_flow' scenario...")
        z_inertia = np.diag([3.0, 3.0, 0.5])
        # Main flow: A dense group of low-priority agents
        for i in range(cfg.NUM_AGENTS - 1):
            offset = Vector3D(random.uniform(-10, 10), random.uniform(-2, 2), random.uniform(-2, 2))
            start_pos = Vector3D(5, center.y, center.z) + offset
            goal_pos = Vector3D(cfg.WORLD_SIZE[0] - 5, center.y, center.z) + offset
            agents.append(Agent(id=i, pos=start_pos, goal=goal_pos, inertia_matrix=z_inertia, priority=1.0))
            
        # Merging agent: A single high-priority agent
        # start_pos = Vector3D(center.x - 10, cfg.WORLD_SIZE[1] - 10, center.z)
        start_pos = Vector3D(5, center.y, center.z)
        goal_pos = Vector3D(cfg.WORLD_SIZE[0] - 5, center.y, center.z)
        agents.append(Agent(id=cfg.NUM_AGENTS - 1, pos=start_pos, goal=goal_pos, inertia_matrix=z_inertia, priority=3.0))

    # --- SCENARIO 3: Hive Takeoff (tests dense Z-axis coordination) ---
    elif cfg.SCENARIO == 'hive_takeoff':
        print("Setting up 'hive_takeoff' scenario...")
        platform_radius = 5.0
        z_takeoff = 5.0
        # Z-axis specialized inertia: easy to move up/down, hard to move sideways
        z_inertia = np.diag([3.0, 3.0, 0.5])
        
        for i in range(cfg.NUM_AGENTS):
            # Start on a tight platform on the ground
            start_offset = Vector3D(random.uniform(-platform_radius, platform_radius),
                                    random.uniform(-platform_radius, platform_radius),
                                    0)
            start_pos = center + start_offset
            start_pos.z = z_takeoff
            
            # Goal at a high altitude and spread out
            goal_offset = Vector3D(random.uniform(-15, 15), random.uniform(-15, 15), 0)
            goal_pos = center + goal_offset
            goal_pos.z = random.uniform(cfg.WORLD_SIZE[2] * 0.7, cfg.WORLD_SIZE[2] * 0.9)
            
            # Assign mixed priorities to see emergent sequencing
            priority = 1.0 + (i % 3) # 1.0, 2.0, 3.0
            agents.append(Agent(id=i, pos=start_pos, goal=goal_pos, inertia_matrix=z_inertia, priority=priority))
            #TODO：这里是没有惯性说明参数的
            
    # --- Baseline Scenarios ---
    elif cfg.SCENARIO == 'antipodal_sphere':
        print("Setting up 'antipodal_sphere' scenario...")
        radius = min(cfg.WORLD_SIZE) * 0.5
        z_inertia = np.diag([1.0, 1.0, 1.0])
        
        for i in range(cfg.NUM_AGENTS):
            phi = random.uniform(0, 2 * math.pi)
            costheta = random.uniform(-1, 1)
            theta = math.acos(costheta)
            
            x = radius * math.sin(theta) * math.cos(phi)
            y = radius * math.sin(theta) * math.sin(phi)
            z = radius * math.cos(theta)

            start_pos = center + Vector3D(x, y, z)
            goal_pos = center - Vector3D(x, y, z)
            
            priority = 1.0 + (i % 3) * 0.5
            agents.append(Agent(id=i, pos=start_pos, goal=goal_pos, inertia_matrix=z_inertia, priority=priority))

    # --- Baseline Hybrid Scenarios ---
    elif cfg.SCENARIO == 'hybrid_sphere':
        print("Setting up 'hive_takeoff' scenario...")
        platform_radius = 5.0
        z_takeoff = 5.0
        # Z-axis specialized inertia: easy to move up/down, hard to move sideways
        z_inertia = np.diag([3.0, 3.0, 0.5])
        
        for i in range(cfg.NUM_AGENTS):
            # Start on a tight platform on the ground
            start_offset = Vector3D(random.uniform(-platform_radius, platform_radius),
                                    random.uniform(-platform_radius, platform_radius),
                                    0)
            start_pos = center + start_offset
            start_pos.z = z_takeoff
            
            # Goal at a high altitude and spread out
            goal_offset = Vector3D(random.uniform(-15, 15), random.uniform(-15, 15), 0)
            goal_pos = center + goal_offset
            goal_pos.z = random.uniform(cfg.WORLD_SIZE[2] * 0.7, cfg.WORLD_SIZE[2] * 0.9)
            priority = 1.0
            agents.append(Agent(id=i, pos=start_pos, goal=goal_pos, inertia_matrix=z_inertia, priority=priority))            
            '''
            if i % 3 != 0:
                agents.append(Agent(id=i, pos=start_pos, goal=goal_pos, inertia_matrix=z_inertia, priority=priority))
            else:
                from agent.orca_agent import OrcaAgent
                agents.append(OrcaAgent(id=i, pos=start_pos, goal=goal_pos, inertia_matrix=z_inertia, priority=priority))
            '''
            
    
    else: # Default to 'random' if scenario name is invalid
        print("Setting up 'random' scenario...")
        for i in range(cfg.NUM_AGENTS):
            start_pos = Vector3D(random.uniform(0, cfg.WORLD_SIZE[0]),
                                 random.uniform(0, cfg.WORLD_SIZE[1]),
                                 random.uniform(0, cfg.WORLD_SIZE[2]))
            goal_pos = Vector3D(random.uniform(0, cfg.WORLD_SIZE[0]),
                                random.uniform(0, cfg.WORLD_SIZE[1]),
                                random.uniform(0, cfg.WORLD_SIZE[2]))
            agents.append(Agent(id=i, pos=start_pos, goal=goal_pos, inertia_matrix=z_inertia, priority=1.0))
    return agents

if __name__ == "__main__":
    import math
    
    print("Initializing 3D PICA Simulation...")
    agents = setup_scenario()
    simulator = Simulator(agents)

    start_time = time.time()
    
    # --- Main Simulation Loop ---
    while simulator.time < cfg.SIMULATION_TIME:
        print(f"Simulating... Time: {simulator.time:.2f}s", end='\r')
        simulator.step()
        
        # Check for early exit condition
        if simulator.all_agents_at_goal():
            print("\nAll agents have reached their goals!")
            break

    end_time = time.time()
    
    print(f"\nSimulation finished in {end_time - start_time:.2f} real-world seconds.")
    print(f"Completed {simulator.time:.2f} simulation seconds.")

    if cfg.VISUALIZE:
        print("Closing visualization.")
        plt.ioff()
        plt.show()