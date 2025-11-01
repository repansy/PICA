# main.py
# import enviroments.config as cfg
import os
from enviroments import config as cfg
from simulator.pica_simulator import Simulator
# from simulator.orca_simulator import Simulator
from enviroments.scenario import scenario_factory
from enviroments.scenario_plane import plane_scenario_factory
from enviroments.scenario_test import HeterogeneousSphereScenario_factory

def main():
    """主函数，根据config选择并运行仿真"""

    # 从工厂中获取对应的场景生成函数
    setup_function = scenario_factory.get(cfg.SCENARIO)
    
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

def batch_run_plane_scenarios():
    # 所有场景名称（从scenario_factory中获取）
    scenarios = list(plane_scenario_factory.keys())
    # 输出目录（确保存在）
    output_dir = os.path.join(cfg.RESULT_DIR, "plane_scenarios")
    os.makedirs(output_dir, exist_ok=True)
    
    for scenario in scenarios:
        print(f"\n===== 运行场景: {scenario} =====")
        # 设置当前场景
        cfg.SCENARIO = scenario
        # 设置带场景名的CSV输出路径
        cfg.TRAJECTORY_FILE = os.path.join(output_dir, f"{scenario}_trajectory.csv")
        # 初始化场景智能体
        agents = plane_scenario_factory[scenario]()
        # 运行仿真
        sim = Simulator(agents)
        while sim.time < cfg.SIMULATION_TIME:
            sim.step()
            if sim.all_agents_at_goal():
                print(f"所有智能体到达目标，提前结束场景 {scenario}")
                break
        print(f"场景 {scenario} 完成，CSV文件: {cfg.TRAJECTORY_FILE}")

def batch_run_scenarios():
    # 所有场景名称（从scenario_factory中获取）
    scenarios = list(scenario_factory.keys())
    # 输出目录（确保存在）
    output_dir = os.path.join(cfg.RESULT_DIR, "scenarios")
    os.makedirs(output_dir, exist_ok=True)
    
    for scenario in scenarios:
        print(f"\n===== 运行场景: {scenario} =====")
        # 设置当前场景
        cfg.SCENARIO = scenario
        # 设置带场景名的CSV输出路径
        cfg.TRAJECTORY_FILE = os.path.join(output_dir, f"{scenario}_trajectory.csv")
        # 初始化场景智能体
        agents = scenario_factory[scenario]()
        # 运行仿真
        sim = Simulator(agents)
        while sim.time < cfg.SIMULATION_TIME:
            sim.step()
            if sim.all_agents_at_goal():
                print(f"所有智能体到达目标，提前结束场景 {scenario}")
                break
        print(f"场景 {scenario} 完成，CSV文件: {cfg.TRAJECTORY_FILE}")

def batch_run_hscenarios():
    scenarios = list(HeterogeneousSphereScenario_factory.keys())
    # 输出目录（确保存在）
    output_dir = os.path.join(cfg.RESULT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    
    for scenario in scenarios:
        print(f"\n===== 运行场景: {scenario} =====")
        # 设置当前场景
        cfg.SCENARIO = scenario
        # 设置带场景名的CSV输出路径
        cfg.TRAJECTORY_FILE = os.path.join(output_dir, f"{scenario}_trajectory.csv")
        cfg.TRAJECTORY_FILE_2 = os.path.join(output_dir, f"{scenario}_alpha.csv")
        cfg.TRAJECTORY_FILE_3 = os.path.join(output_dir, f"{scenario}_RMPsetting.csv")

        # 初始化场景智能体
        agents = HeterogeneousSphereScenario_factory[scenario]()
        # 运行仿真
        sim = Simulator(agents)
        while sim.time < cfg.SIMULATION_TIME:
            sim.step()
            if sim.all_agents_at_goal():
                print(f"所有智能体到达目标，提前结束场景 {scenario}")
                break
        print(f"场景 {scenario} 完成，CSV文件: {cfg.TRAJECTORY_FILE}")


if __name__ == "__main__":
    
    # main()
    # batch_run_plane_scenarios()
    # batch_run_scenarios()
    batch_run_hscenarios()