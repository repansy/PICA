import time
import numpy as np
from simulator.pica2d_simulator import Simulator  # 2D仿真器
from enviroments.scenario_2d import scenario_factory_2d
from enviroments.visualizer import plot_2d_trajectories  # 复用2D可视化函数
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
    
    # 3. 记录轨迹数据（时间步，智能体，坐标）
    trajectory_data = []
    trajectory_data.append([f"Agent{i}_x" for i in range(cfg.NUM_AGENTS)] + 
                          [f"Agent{i}_y" for i in range(cfg.NUM_AGENTS)])
    
    # 4. 运行仿真循环
    start_time = time.time()
    while sim.time < cfg.SIMULATION_TIME:
        print(f"Simulating 2D... Time: {sim.time:.2f}s", end='\r')
        sim.step()  # 2D仿真步长更新
        
        # 记录当前位置
        current_pos = []
        for agent in agents:
            current_pos.extend([agent.pos.x, agent.pos.y])
        trajectory_data.append(current_pos)
        
        # 检查是否所有智能体到达目标
        if sim.all_agents_at_goal():
            print("\nAll agents reached goal!")
            break
    
    # 5. 输出统计信息
    end_time = time.time()
    print(f"\nSimulation finished in {end_time - start_time:.2f}s (real time)")
    print(f"Total collision events: {sim.total_collision_events}")
    
    # 6. 保存轨迹并可视化
    if cfg.SAVE_TRAJECTORY:
        np.savetxt("2d_trajectory.csv", trajectory_data, delimiter=",", fmt="%s")
        print("Trajectory saved to 2d_trajectory.csv")
    
    if cfg.VISUALIZE:
        # 转换轨迹数据为numpy数组（timesteps, agents, 2）
        pos_array = np.array(trajectory_data[1:], dtype=np.float32).reshape(-1, cfg.NUM_AGENTS, 2)
        plot_2d_trajectories(pos_array, plane='xy')  # 2D仅需xy平面


if __name__ == "__main__":
    main()