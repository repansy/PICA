import time
import numpy as np
import os
from simulator.pica2d_simulator import Simulator  # 2D仿真器
from enviroments.scenario_2d import scenario_factory_2d
from enviroments.scenario_plane import plane_scenario_factory
import examples.pica_2d.v2.config as cfg


def main():
    """2D仿真主函数"""
    print("Initializing 2D PICA Simulation...")
    
    # 1. 获取场景智能体
    setup_function = scenario_factory_2d.get(cfg.SCENARIO_2D)
    if not setup_function:
        raise ValueError(f"Unknown 2D scenario: {cfg.SCENARIO_2D}")
    agents = setup_function()
    
    # 2. 初始化2D仿真器
    sim = Simulator(agents)
    
    # 3. 运行仿真循环
    start_time = time.time()
    while sim.time < cfg.SIMULATION_TIME:
        print(f"Simulating 2D... Time: {sim.time:.2f}s", end='\r')
        sim.step()  # 2D仿真步长更新
        
        # 检查是否所有智能体到达目标
        if sim.all_agents_at_goal():
            print("\nAll agents reached goal!")
            break
    
    # 5. 输出统计信息
    end_time = time.time()
    print(f"\nSimulation finished in {end_time - start_time:.2f}s (real time)")
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
        cfg.SCENARIO_2D = scenario
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

if __name__ == "__main__":
    # main()
    batch_run_plane_scenarios()