import os
from enviroments import config as cfg
from simulator.pica_simulator import Simulator
# from simulator.orca_simulator import Simulator
from examples.pica_3d.v2.scenario_sphere import scenario_factory
from examples.pica_3d.v2.scenario_plane import plane_scenario_factory

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