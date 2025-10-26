# main.py
# import enviroments.config as cfg
from examples.pica_3d.v2 import config as cfg
from simulator.pica_simulator import Simulator
from enviroments.scenario import scenario_factory
from enviroments.scenario_plane import plane_scenario_factory
import os

from utils.pica_structures import Vector3D
from enviroments.scenario_test import HeterogeneousSphereScenario 

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


if __name__ == "__main__":
    
    # main()
    # batch_run_plane_scenarios()
    # batch_run_scenarios()

    # 示例 1: 离散三级场景 (高/中/低)
    print("--- 正在创建离散三级场景 (30% 高, 50% 中, 20% 低) ---")
    discrete_groups = [
        {
            'ratio': 0.3, 
            'params': {'radius': 0.5, 'P': 0.9, 'M': Vector3D(1.0, 1.0, 1.0)}
        },
        {
            'ratio': 0.5, 
            'params': {'radius': 0.5, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}
        },
        {
            'ratio': 0.2, 
            'params': {'radius': 0.5, 'P': 0.1, 'M': Vector3D(1.0, 1.0, 1.0)}
        },
    ]
    discrete_scenario = HeterogeneousSphereScenario(agent_groups=discrete_groups, num_agents=10)
    discrete_agents = discrete_scenario.create_agents()
    print(f"成功创建 {len(discrete_agents)} 个智能体。\n")

    # 示例 2: 角色扮演场景 (重型 vs 敏捷)
    print("--- 正在创建角色扮演场景 (40% 重型, 60% 敏捷) ---")
    role_based_groups = [
        {
            'ratio': 0.4, 
            'params': {'radius': 0.8, 'P': 0.8, 'M': Vector3D(5.0, 5.0, 5.0)} # 重型: R大, P高, M大
        },
        {
            'ratio': 0.6, 
            'params': {'radius': 0.4, 'P': 0.3, 'M': Vector3D(0.5, 0.5, 0.5)} # 敏捷: R小, P低, M小
        },
    ]
    role_scenario = HeterogeneousSphereScenario(agent_groups=role_based_groups, num_agents=10)
    role_agents = role_scenario.create_agents()
    print(f"成功创建 {len(role_agents)} 个智能体。\n")

    # 示例 3: 单一异质性测试 (仅半径不同)
    print("--- 正在创建仅半径不同的场景 (50% 大, 50% 小) ---")
    radius_groups = [
        {
            'ratio': 0.5, 
            'params': {'radius': 1.0, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}
        },
        {
            'ratio': 0.5, 
            'params': {'radius': 0.3, 'P': 0.5, 'M': Vector3D(1.0, 1.0, 1.0)}
        },
    ]
    radius_scenario = HeterogeneousSphereScenario(agent_groups=radius_groups, num_agents=10)
    radius_agents = radius_scenario.create_agents()
    print(f"成功创建 {len(radius_agents)} 个智能体。\n")