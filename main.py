# main.py
import os
import random
from enviroments import config as cfg
from simulator.pica_simulator import Simulator
# from simulator.orca_simulator import Simulator
from enviroments.scenario_test import SphereScenario_factory, CircleScenario_factory
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


def batch_run_s_scenarios(seeds=None):
    if seeds is None:
        seeds = [42]  # 默认使用一个seed
        
    scenarios = list(SphereScenario_factory.keys())
    output_dir = os.path.join(cfg.RESULT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    
    for scenario in scenarios:
        for _, seed in enumerate(seeds):
            print(f"\n===== 运行场景: {scenario}, Seed: {seed} =====")
            # 设置当前场景
            cfg.SCENARIO = scenario
            cfg.TRAJECTORY_FILE = os.path.join(output_dir, f"{scenario}{seed}_trajectory.csv")
            cfg.TRAJECTORY_FILE_2 = os.path.join(output_dir, f"{scenario}{seed}_alpha.csv")
            cfg.TRAJECTORY_FILE_3 = os.path.join(output_dir, f"{scenario}{seed}_RMPsetting.csv")
           
            # 初始化场景智能体
            agents = SphereScenario_factory[scenario](seed=seed)
            # 运行仿真
            sim = Simulator(agents)
            while sim.time < cfg.SIMULATION_TIME:
                sim.step()
                if sim.all_agents_at_goal():
                    print(f"所有智能体到达目标，提前结束场景 {scenario} (seed={seed})")
                    break
            print(f"\nSimulation finished at time {sim.time:.2f}s.")
            print(f"Total collision events: {sim.total_collision_events}")
            print(f"场景 {scenario} (seed={seed})完成，CSV文件: {cfg.TRAJECTORY_FILE}")
            
            if cfg.VISUALIZE:
                input("Press Enter to close the plot...")

def batch_run_c_scenarios(seeds=None):
    if seeds is None:
        seeds = [42]  # 默认使用一个seed
    scenarios = list(CircleScenario_factory.keys())
    # 输出目录（确保存在）
    output_dir = os.path.join(cfg.RESULT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    
    for scenario in scenarios:
        for _, seed in enumerate(seeds):
            print(f"\n===== 运行场景: {scenario}, Seed: {seed} =====")
            # 设置当前场景
            cfg.SCENARIO = scenario
            cfg.TRAJECTORY_FILE = os.path.join(output_dir, f"{scenario}{seed}_trajectory.csv")
            cfg.TRAJECTORY_FILE_2 = os.path.join(output_dir, f"{scenario}{seed}_alpha.csv")
            cfg.TRAJECTORY_FILE_3 = os.path.join(output_dir, f"{scenario}{seed}_RMPsetting.csv")
            
            # 初始化场景智能体
            agents = CircleScenario_factory[scenario](seed=seed)
            # 运行仿真
            sim = Simulator(agents)
            while sim.time < cfg.SIMULATION_TIME:
                sim.step()
                if sim.all_agents_at_goal():
                    print(f"所有智能体到达目标，提前结束场景 {scenario} (seed={seed})")
                    break
            print(f"\nSimulation finished at time {sim.time:.2f}s.")
            print(f"Total collision events: {sim.total_collision_events}")
            print(f"场景 {scenario} (seed={seed})完成，CSV文件: {cfg.TRAJECTORY_FILE}")
            
            if cfg.VISUALIZE:
                input("Press Enter to close the plot...")

def batch_run_random_seeds(num_runs=10):
    """运行多次随机seed的实验"""
    random_seeds = [random.randint(0, 100) for _ in range(num_runs)]
    # batch_run_s_scenarios(seeds=random_seeds)
    batch_run_c_scenarios(seeds=random_seeds)
    
if __name__ == "__main__":
    batch_run_random_seeds(num_runs=10)
    # batch_run_c_scenarios()
    # batch_run_s_scenarios()
    # 42 44 1 2 24 43/45 22 24 14 41