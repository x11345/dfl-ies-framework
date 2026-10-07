# energy_model.py (15分钟版：步长从1小时改为15分钟，默认96时段)
from pyomo.environ import *

def solve_energy_model(demand, heat_demand, cool_demand,
                       price, pv,
                       co2_limit,
                       co2_grid=0.65,
                       co2_gas=0.2,
                       cop_hp=3.5,
                       cop_c=4.0,
                       dt=0.25):  # <--- 新增：时间步长（小时），默认15分钟=0.25h
    """
    版本 7.0 (15分钟步长版):
    - 时间步长由1小时改为15分钟，适应负荷预测的96点/天。
    - 能量相关的约束（SOC变化、储能容量）已乘以 dt 因子。
    - 其他设备模型不变，功率变量仍为瞬时kW，经济计算考虑时段长度。
    """

    model = ConcreteModel()

    # ===== 1. 时间集合 =====
    # 自动根据输入数据的长度确定时段数（支持任意长度，典型为96）
    n_steps = len(demand)
    model.T = RangeSet(1, n_steps)   # <--- 修改：原为固定24

    # ===== 2. 参数 =====
    # 将字典或列表转换为以1..n_steps为索引的Param
    def _to_dict(arr):
        if isinstance(arr, dict):
            return arr
        elif isinstance(arr, (list, tuple)):
            return {i+1: arr[i] for i in range(len(arr))}
        else:
            raise TypeError("demand/price等参数应为字典或列表")

    model.demand = Param(model.T, initialize=_to_dict(demand))
    model.heat_demand = Param(model.T, initialize=_to_dict(heat_demand))
    model.price = Param(model.T, initialize=_to_dict(price))
    model.pv = Param(model.T, initialize=_to_dict(pv))
    model.cool_demand = Param(model.T, initialize=_to_dict(cool_demand))

    # ===== 3. 设备参数 =====
    # 根据宁德通用制造业园区负荷规模（峰值约80MW）调整
    # 3.1 电池（功率单位：kW，能量单位：kWh）
    E_max = 30000.0  # 电池容量 (kWh)  原100 → 10000 (10MWh)
    P_bat_max = 12000.0  # 最大充放电功率 (kW) 原30 → 5000 (5MW)
    eta_ch = 0.98
    eta_dis = 0.98
    SOC_init = 5000.0  # 初始SOC (kWh) 原50 → 5000 (50%)

    # 3.2 CHP (保留，作为基荷或补充)
    eta_e = 0.35
    eta_h = 0.45

    # 3.3 热泵 (替代原锅炉)
    # COP参数已在函数参数中定义

    # 3.4 蓄热罐 (能量单位：kWh_th)
    H_tank_max = 5000.0  # 储热容量 (kWh_th)  原80 → 5000
    H_tank_rate = 2000.0  # 最大充放热功率 (kW_th) 原20 → 2000
    eta_h_in = 0.95
    eta_h_out = 0.95
    S_heat_init = 2500.0  # 初始储热量 原20 → 2500

    # 3.5 冷储能参数 (能量单位：kWh_c)
    C_tank_max = 8000.0  # 储冷容量 (kWh_c)  原100 → 8000
    C_tank_rate = 3000.0  # 最大充放冷功率 (kW_c) 原30 → 3000
    eta_c_in = 0.95
    eta_c_out = 0.95
    S_cool_init = 4000.0  # 初始储冷量 原20 → 4000

    # 3.6 经济参数 (价格单位：元/kWh 或 元/kWh_th)
    gas_price = 0.35       # 天然气价格 (元/kWh 燃料热值)
    sell_price = 0.30      # 售电价格 (元/kWh)

    # ===== 4. 决策变量 =====
    model.P_buy = Var(model.T, domain=NonNegativeReals)      # 购电功率 (kW)
    model.P_sell = Var(model.T, domain=NonNegativeReals)     # 售电功率 (kW)
    model.u_grid = Var(model.T, domain=Binary)               # 购电/售电互斥

    # 计算电网交互上限（用于二元约束）
    P_grid_max = max(model.demand[t] for t in model.T) \
                 + max(model.cool_demand[t] for t in model.T) / cop_c \
                 + max(model.heat_demand[t] for t in model.T) / cop_hp \
                 + P_bat_max

    model.buy_limit = Constraint(model.T,
        rule=lambda m, t: m.P_buy[t] <= P_grid_max * m.u_grid[t])
    model.sell_limit = Constraint(model.T,
        rule=lambda m, t: m.P_sell[t] <= P_grid_max * (1 - m.u_grid[t]))

    # 电池变量 (功率 kW)
    model.P_ch = Var(model.T, domain=NonNegativeReals)
    model.P_dis = Var(model.T, domain=NonNegativeReals)
    model.SOC = Var(model.T, bounds=(0, E_max))   # 储能状态 (kWh)
    model.u_bat = Var(model.T, domain=Binary)

    # CHP 变量
    model.F_chp = Var(model.T, domain=NonNegativeReals)      # 燃料功率 (kW)
    model.P_chp = Var(model.T, domain=NonNegativeReals, bounds=(0, 15000))      # 电功率 (kW)
    model.H_chp = Var(model.T, domain=NonNegativeReals)      # 热功率 (kW_th)

    # 热泵变量
    model.P_hp = Var(model.T, domain=NonNegativeReals, bounds=(0, 8000))       # 热泵耗电 (kW)
    model.H_hp = Var(model.T, domain=NonNegativeReals)       # 热泵产热 (kW_th)

    # 储热变量
    model.H_store_in = Var(model.T, domain=NonNegativeReals) # 充热功率 (kW_th)
    model.H_store_out = Var(model.T, domain=NonNegativeReals)# 放热功率 (kW_th)
    model.S_heat = Var(model.T, bounds=(0, H_tank_max))      # 储热量 (kWh_th)
    model.u_heat = Var(model.T, domain=Binary)               # 充放互斥

    # 制冷变量
    model.P_chiller = Var(model.T, domain=NonNegativeReals, bounds=(0, 10000))  # 制冷机耗电 (kW)
    model.C_chiller = Var(model.T, domain=NonNegativeReals)  # 制冷功率 (kW_c)

    # 储冷变量
    model.C_store_in = Var(model.T, domain=NonNegativeReals) # 充冷功率 (kW_c)
    model.C_store_out = Var(model.T, domain=NonNegativeReals)# 放冷功率 (kW_c)
    model.S_cool = Var(model.T, bounds=(0, C_tank_max))      # 储冷量 (kWh_c)
    model.u_cool = Var(model.T, domain=Binary)               # 充放互斥

    # ===== 5. 目标函数 =====
    carryover_value = 0.8

    def total_cost(m):
        # 1. 累加所有时段的常规成本（只把与 t 相关的放进 sum）
        base_cost = sum(
            (m.P_buy[t] * m.price[t] * dt) -
            (m.P_sell[t] * m.price[t] * 0.7 * dt) +
            (m.F_chp[t] * gas_price * dt)
            for t in m.T
        )

        # 2. 单独计算终端价值（没有 for 循环）
        #erminal_reward = (m.SOC[n_steps] - SOC_init) * carryover_value

        # 3. 返回总结果
        return base_cost

    model.obj = Objective(rule=total_cost, sense=minimize)
    # ===== 6. 约束 =====

    # 电力平衡 (功率瞬时平衡)
    def power_balance(m, t):
        return (m.P_buy[t] + m.P_dis[t] + m.pv[t] + m.P_chp[t] ==
                m.demand[t] + m.P_ch[t] + m.P_sell[t] + m.P_hp[t] + m.P_chiller[t])
    model.power_balance = Constraint(model.T, rule=power_balance)

    # 热力平衡 (功率瞬时平衡)
    def heat_balance(m, t):
        return (m.H_chp[t] + m.H_hp[t] + m.H_store_out[t] ==
                m.heat_demand[t] + m.H_store_in[t])
    model.heat_balance = Constraint(model.T, rule=heat_balance)

    # 冷平衡 (功率瞬时平衡)
    def cool_balance(m, t):
        return (m.C_chiller[t] + m.C_store_out[t] ==
                m.cool_demand[t] + m.C_store_in[t])
    model.cool_balance = Constraint(model.T, rule=cool_balance)

    # 设备转换约束
    model.chp_elec = Constraint(model.T, rule=lambda m, t: m.P_chp[t] == eta_e * m.F_chp[t])
    model.chp_heat = Constraint(model.T, rule=lambda m, t: m.H_chp[t] == eta_h * m.F_chp[t])
    model.hp_convert = Constraint(model.T, rule=lambda m, t: m.H_hp[t] == cop_hp * m.P_hp[t])
    model.chiller_convert = Constraint(model.T, rule=lambda m, t: m.C_chiller[t] == cop_c * m.P_chiller[t])

    # 储电 SOC 约束 (加入 dt 因子)
    def soc_balance(m, t):
        if t == 1:
            return m.SOC[t] == SOC_init + \
                   (eta_ch * m.P_ch[t] - m.P_dis[t] / eta_dis) * dt   # <--- 修改：乘dt
        else:
            return m.SOC[t] == m.SOC[t-1] + \
                   (eta_ch * m.P_ch[t] - m.P_dis[t] / eta_dis) * dt   # <--- 修改：乘dt
    model.soc_balance = Constraint(model.T, rule=soc_balance)

    # 储热 SOC 约束 (乘dt)
    def heat_soc_balance(m, t):
        if t == 1:
            return m.S_heat[t] == S_heat_init + \
                   (eta_h_in * m.H_store_in[t] - m.H_store_out[t] / eta_h_out) * dt   # <--- 乘dt
        else:
            return m.S_heat[t] == m.S_heat[t-1] + \
                   (eta_h_in * m.H_store_in[t] - m.H_store_out[t] / eta_h_out) * dt   # <--- 乘dt
    model.heat_soc_balance = Constraint(model.T, rule=heat_soc_balance)

    # 储冷 SOC 约束 (乘dt)
    def cool_soc_balance(m, t):
        if t == 1:
            return m.S_cool[t] == S_cool_init + \
                   (eta_c_in * m.C_store_in[t] - m.C_store_out[t] / eta_c_out) * dt   # <--- 乘dt
        else:
            return m.S_cool[t] == m.S_cool[t-1] + \
                   (eta_c_in * m.C_store_in[t] - m.C_store_out[t] / eta_c_out) * dt   # <--- 乘dt
    model.cool_soc_balance = Constraint(model.T, rule=cool_soc_balance)

    # 终端SOC约束 (改为最后时段 n_steps)
    model.soc_terminal = Constraint(expr=model.SOC[n_steps] == SOC_init)      # <--- 修改：24 -> n_steps
    model.heat_terminal = Constraint(expr=model.S_heat[n_steps] == S_heat_init)
    model.cool_terminal = Constraint(expr=model.S_cool[n_steps] == S_cool_init)

    # 功率限制（储能充放速率限制）
    model.heat_in_limit = Constraint(model.T,
        rule=lambda m, t: m.H_store_in[t] <= H_tank_rate * m.u_heat[t])
    model.heat_out_limit = Constraint(model.T,
        rule=lambda m, t: m.H_store_out[t] <= H_tank_rate * (1 - m.u_heat[t]))
    model.charge_limit = Constraint(model.T,
        rule=lambda m, t: m.P_ch[t] <= P_bat_max * m.u_bat[t])
    model.discharge_limit = Constraint(model.T,
        rule=lambda m, t: m.P_dis[t] <= P_bat_max * (1 - m.u_bat[t]))
    model.cool_in_limit = Constraint(model.T,
        rule=lambda m, t: m.C_store_in[t] <= C_tank_rate * m.u_cool[t])
    model.cool_out_limit = Constraint(model.T,
        rule=lambda m, t: m.C_store_out[t] <= C_tank_rate * (1 - m.u_cool[t]))

    # SOC上限已通过Var bounds定义

    # 碳排放约束 (注意：电量需乘dt转化为能量)
    def carbon_emission_rule(m):
        total_carbon = sum(
            (m.P_buy[t] * dt) * co2_grid +      # <--- 修改：购电量(kWh)乘以碳排放系数(kg/kWh)
            (m.F_chp[t] * dt) * co2_gas         # <--- 修改：燃气消耗量(kWh)乘以碳排放系数
            for t in m.T
        )
        return total_carbon <= co2_limit
    model.carbon_constraint = Constraint(rule=carbon_emission_rule)

    # ===== 7. 求解 =====
    solver = SolverFactory('gurobi', solver_io='python')
    results = solver.solve(model, tee=False)

    if (results.solver.status == SolverStatus.ok) and (
            results.solver.termination_condition == TerminationCondition.optimal):
        print("\n✅ 模型求解成功！正在提取结果...")
    elif results.solver.termination_condition == TerminationCondition.infeasible:
        print("\n❌ 模型无解 (Infeasible)！")
        return None
    else:
        print(f"\n❌ 求解失败！状态: {results.solver.status}, 终止条件: {results.solver.termination_condition}")
        return None

    # ===== 8. 输出 =====
    # 经济指标（乘dt还原为实际能量费用）
    cost_buy_elec = sum(value(model.P_buy[t]) * value(model.price[t]) * dt for t in model.T)
    rev_sell_elec = sum(value(model.P_sell[t]) * value(model.price[t]) * 0.7 * dt for t in model.T)
    cost_gas_chp = sum(value(model.F_chp[t]) * gas_price * dt for t in model.T)

    carbon_grid = sum(value(model.P_buy[t]) * dt * co2_grid for t in model.T)
    carbon_gas_chp = sum(value(model.F_chp[t]) * dt * co2_gas for t in model.T)
    total_actual_carbon = carbon_grid + carbon_gas_chp

    result = {
        # 电网
        "P_buy": {t: value(model.P_buy[t]) for t in model.T},
        "P_sell": {t: value(model.P_sell[t]) for t in model.T},
        # 电池
        "P_ch": {t: value(model.P_ch[t]) for t in model.T},
        "P_dis": {t: value(model.P_dis[t]) for t in model.T},
        "SOC": {t: value(model.SOC[t]) for t in model.T},
        # CHP
        "P_chp": {t: value(model.P_chp[t]) for t in model.T},
        "H_chp": {t: value(model.H_chp[t]) for t in model.T},
        "F_chp": {t: value(model.F_chp[t]) for t in model.T},
        # 热泵
        "P_hp": {t: value(model.P_hp[t]) for t in model.T},
        "H_hp": {t: value(model.H_hp[t]) for t in model.T},
        # 储热
        "H_store_in": {t: value(model.H_store_in[t]) for t in model.T},
        "H_store_out": {t: value(model.H_store_out[t]) for t in model.T},
        "S_heat": {t: value(model.S_heat[t]) for t in model.T},
        # 制冷
        "P_chiller": {t: value(model.P_chiller[t]) for t in model.T},
        "C_chiller": {t: value(model.C_chiller[t]) for t in model.T},
        # 储冷
        "C_store_in": {t: value(model.C_store_in[t]) for t in model.T},
        "C_store_out": {t: value(model.C_store_out[t]) for t in model.T},
        "S_cool": {t: value(model.S_cool[t]) for t in model.T},
        # 经济与碳排放汇总
        "Cost_Buy_Elec": cost_buy_elec,
        "Rev_Sell_Elec": rev_sell_elec,
        "Cost_Gas_CHP": cost_gas_chp,
        "Carbon_Grid": carbon_grid,
        "Carbon_Gas": carbon_gas_chp,
        "Total_Carbon": total_actual_carbon,
        "n_steps": n_steps,          # <--- 新增：返回时段数便于外部使用
        "dt": dt                      # <--- 新增：返回时间步长
    }

    return result