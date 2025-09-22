main.main()
└── sim.run_step(dt)
    ├── sim._get_measurements()             # [SIMULATOR] 使用KD-Tree模拟感知
    │   └── kdtree.query_ball_point()
    └── for agent in sim.agents:
        ├── agent.update_beliefs(measurements, t, dt) # [AGENT]
        │   ├── belief_manager.predict_all(t)
        │   └── belief_manager.update_from_measurement(...)
        │       └── (EKF数学逻辑)
        └── agent.compute_new_velocity(obstacles)   # [AGENT]
            ├── agent.get_pref_velocity()
            └── optimization.solve_argo_3d_velocity(...) # [OPTIMIZATION]
                └── scipy.optimize.minimize(fun=cost_function, ...)
                    └── cost_function(v_candidate)  # 这是被反复调用的核心
                        ├── _calculate_local_metrics(...)
                        ├── _calculate_probabilistic_risk(...)
                        │   └── (概率、马氏距离、卡方分布的数学逻辑)
                        └── _calculate_static_risk(...)

好的，我们来梳理并完整展示ARGO-3D算法在一个决策周期内的完整计算循环。这个流程将清晰地串联起从不完美的感知到最终的控制指令的全过程。

---

### **ARGO-3D 算法核心流程梳理 (Conceptual Flow)**

ARGO-3D的单步决策可以分解为三个核心阶段，这构成了智能体的“思考”过程：

1.  **信念更新 (Belief Update): “我现在对世界有什么看法？”**
    *   **目标：** 将新的、不完美的观测数据，与自己对世界已有的记忆（信念模型）相融合，形成一个对当前及未来邻居状态的、最精确的概率性估计。
    *   **机制：** 扩展卡尔曼滤波器 (EKF) / 无迹卡尔曼滤波器 (UKF)。

2.  **风险评估 (Risk Assessment): “如果我采取某个行动，未来会怎样？”**
    *   **目标：** 将信念更新阶段得到的概率分布，转化为一个在自身决策空间（速度空间）中的“风险地图”。这张地图不仅反映物理碰撞概率，还融入了社会规则和非对称交互原则。
    *   **机制：** 构建非对称、概率性的成本函数 `J(v_A)`。

3.  **决策优化 (Decision Optimization): “在所有可能性中，我应该选择哪个行动？”**
    *   **目标：** 在“风险地图”上，系统性地寻找一个点（一个速度向量），该点既能高效地飞向目标，又能将总风险降至最低，同时还必须满足自身的物理限制。
    *   **机制：** 非线性梯度下降优化。

---

### **单时间步计算循环详解 (A Single Time Step `Δt`)**

假设我们是无人机 **Agent A**，当前仿真时间为 `t`，决策周期为 `Δt`。

#### **输入 (Inputs at time `t`)**

1.  **自身状态 (Agent A's own state):**
    *   精确位置 `p_A(t)`
    *   精确速度 `v_A(t)`
    *   静态配置 `config_A` (半径, 最大速度, 最大加速度等)
    *   任务目标点 `goal_A`

2.  **内部记忆 (Agent A's internal memory):**
    *   `BeliefManager` 中存储的对所有已知邻居 `B_i` 的信念模型 `BeliefState_i(t)`，包含：
        *   上一时刻的均值 `x̂_i(t)` 和协方差 `Σ_i(t)`
        *   上次更新时间 `t_update_i`

3.  **外部观测 (Sensor/Communication data):**
    *   一个字典 `measurements`，包含了在 `[t, t+Δt]` 时间段内从模拟器（现实中是传感器）收到的、关于部分邻居的、带噪声和延迟的观测数据。
    *   *例如: `{ 7: State(pos=(10.2, 5.1, 20.4), vel=(2.1, -1.0, 0.1)), ... }`，表示收到了来自ID为7的邻居的信息。*

---

#### **计算循环 (Computation Cycle)**

**【阶段一：信念更新 (Belief Update)】**

1.  **时间演化 (Prediction Step):** 对`BeliefManager`中**所有**已知的邻居，无论本周期是否收到其新信息，都执行一次EKF的预测步。
    *   **For each neighbor `B_i` in `BeliefManager`:**
        *   `Δt_predict = t + Δt - t_update_i` (计算自上次更新以来的时间差)
        *   使用运动模型（如匀速模型的状态转移矩阵`F`）更新均值：`x̂_i_predicted = F(Δt_predict) * x̂_i(t)`
        *   增加不确定性：`Σ_i_predicted = F * Σ_i(t) * F^T + Q` (Q为过程噪声)
        *   更新内存：`BeliefState_i` -> `(x̂_i_predicted, Σ_i_predicted, t_update_i)`

2.  **数据融合 (Update Step):** 对本周期**收到新信息**的邻居，执行一次EKF的更新步。
    *   **For each neighbor `B_j` in `measurements`:**
        *   从`measurements`中获取其观测值 `z_j` 和观测噪声 `R_j`。
        *   使用`BeliefState_j`中**已预测过**的 `x̂_j_predicted` 和 `Σ_j_predicted`，结合`z_j`，计算卡尔曼增益 `K`。
        *   修正均值：`x̂_j_updated = x̂_j_predicted + K * (z_j - H * x̂_j_predicted)` (H为观测矩阵)
        *   减小不确定性：`Σ_j_updated = (I - K * H) * Σ_j_predicted`
        *   更新内存：`BeliefState_j` -> `(x̂_j_updated, Σ_j_updated, t + Δt)` (更新时间戳)

3.  **记忆管理 (Memory Management):**
    *   调用`BeliefManager.forget_lost_neighbors(t + Δt)`，移除那些长时间未收到任何信息的邻居。

**【阶段二：风险评估 (Risk Assessment)】**

4.  **计算期望速度 (Preferred Velocity):**
    *   `direction_to_goal = normalize(goal_A - p_A(t))`
    *   `v_pref = direction_to_goal * config_A.max_speed`
    *   *（可选高阶优化）可在此处根据邻居的平均航向等信息微调`v_pref`*

5.  **定义成本函数 (Define Cost Function):**
    *   在内存中构建一个Python函数`cost_function(v_candidate)`，该函数接受一个3D速度向量 `v_candidate` 作为输入。
    *   **Inside `cost_function(v_candidate)`:**
        a. **计算效率成本:** `efficiency_cost = ||v_candidate - v_pref||²`
        b. **初始化安全成本:** `safety_cost = 0`
        c. **循环遍历所有邻居:**
            *   **For each neighbor `B_i` in the (now updated) `BeliefManager`:**
                i.   从 `BeliefState_i` 中提取其概率分布 `(x̂_i, Σ_i)`。
                ii.  **计算非对称变换矩阵 `T_i`:** 根据 `p_A` 和 `p̂_i` 的相对位置，以及 `config_A` 和 `config_i` 的优先级，确定一个3x3的变换矩阵。
                iii. **计算风险 `f_Bi(v_candidate)`:** 调用一个内部函数，该函数执行以下操作：
                    *   计算相对速度 `v_rel = v_candidate - v̂_i`。
                    *   计算最近点时刻 `t_cpa`。
                    *   计算`t_cpa`时刻的相对位置均值 `μ_cpa` 和**扭曲后**的协方差 `Σ_rel_transformed = T_i * (Σ_A + Σ_i) * T_i^T`。
                    *   计算高斯分布`N(μ_cpa, Σ_rel_transformed)`在半径为`r_A + r_i`的3D球内的积分，得到碰撞概率 `P_coll`。
                    *   返回 `risk = P_coll`。
                iv.  **计算非对称权重 `w_i`:** 根据优先级、路权等规则确定。
                v.   **累加安全成本:** `safety_cost -= w_i * log(1 - risk + 1e-9)`
        d. **返回总成本:** `return efficiency_cost + safety_cost`

**【阶段三：决策优化 (Decision Optimization)】**

6.  **执行优化 (Execute Optimization):**
    *   **初始猜测:** `v_initial_guess = v_A(t)` 或 `v_pref`。
    *   **定义边界:** 根据`config_A.max_speed`定义速度边界`bounds`。
    *   **调用求解器:**
        ```python
        result = scipy.optimize.minimize(
            fun=cost_function,
            x0=v_initial_guess,
            bounds=bounds,
            method='L-BFGS-B' 
        )
        ```
    *   **提取最优速度:** `v_optimal_raw = result.x`

7.  **应用动力学约束 (Apply Kinematic Constraints):**
    *   计算从当前速度`v_A(t)`到`v_optimal_raw`所需的速度变化 `Δv = v_optimal_raw - v_A(t)`。
    *   计算所需加速度 `a_required = Δv / Δt`。
    *   如果`||a_required|| > config_A.max_accel`，则将`a_required`缩放到最大加速度限制内：`a_clipped = normalize(a_required) * config_A.max_accel`。
    *   **计算最终可执行的速度指令:** `v_command(t+Δt) = v_A(t) + a_clipped * Δt`。

---

#### **输出 (Outputs at the end of the cycle)**

*   **最终控制指令 (Final Command):** `v_command(t+Δt)`。这个指令将被发送给模拟器的物理引擎（或真实无人机的飞控），用于更新下一个时间步的真值状态。

这个完整的循环，从处理杂乱的现实数据开始，通过严谨的概率推理和非线性优化，最终得出了一个安全、高效且符合物理规律的决策。