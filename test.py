# main.py
import os
from enviroments import config as cfg
from simulator.pica_simulator import Simulator
# from simulator.orca_simulator import Simulator
from enviroments.scenario_test import SphereScenario_factory
import time

def main():
    """主函数，根据config选择并运行仿真"""

    # 从工厂中获取对应的场景生成函数
    setup_function = SphereScenario_factory.get(cfg.SCENARIO)
    
    if not setup_function:
        raise ValueError(f"Unknown scenario '{cfg.SCENARIO}' in config file.")

    # 1. 初始化智能体
    agents = setup_function()

    # 2. 初始化仿真器
    sim = Simulator(agents)

    # 3. 运行仿真主循环
    while sim.time < cfg.SIMULATION_TIME:
        print(f"\rSimulation Time: {sim.time:.2f}s", end="")
        
        sim.step()

        if sim.all_agents_at_goal():
            print("\nAll agents have reached their goals!")
            break
            
    print(f"\nSimulation finished at time {sim.time:.2f}s.")
    print(f"Total collision events: {sim.total_collision_events}")

    if cfg.VISUALIZE:
        input("Press Enter to close the plot...")

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("compute time:", end - start)
    