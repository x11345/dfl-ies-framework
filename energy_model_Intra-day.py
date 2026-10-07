import pandas as pd
from pyomo.environ import *


def run_ibp_from_excel(xlsx_path, actual_demand, actual_heat, actual_cool,
                       price, actual_pv, co2_limit,
                       penalty_buy_factor=1.5, penalty_dump_price=0.05,sell_extra_factor = 0.5,
                       soc_dev_margin=0.4,
                       dt=0.25):  # <--- 新增：时间步长（小时）
    """
    通过读取日前模型生成的 Excel 结果，自动运行日内平衡模型 (IBP)
    15分钟步长版：日前计划Excel应为96行数据
    """
    try:
        if xlsx_path.endswith('.csv'):
            df_da = pd.read_csv(xlsx_path)
        else:
            df_da = pd.read_excel(xlsx_path, sheet_name='Hourly_Result')
    except Exception as e:
        print(f"❌ 读取日前计划文件失败: {e}")
        return None

    # 1. 提取需要被“锚定”或“跟踪”的日前计划
    da_plan = {
        "P_buy": df_da["Buy_Power"].values,
        "P_sell": df_da["Sell_Power"].values,
        "P_chp": df_da["CHP_Electric"].values,
        "H_chp": df_da["CHP_Heat"].values,
        "F_chp": df_da["CHP_Gas"].values,
        "SOC": df_da["Battery_SOC"].values
    }

    print(f"✅ 成功加载日前计划，共 {len(df_da)} 个时间步。正在进行日内实时平衡计算...")

    # 2. 运行日内模型
    return solve_ibp_model(
        actual_demand=actual_demand,
        actual_heat=actual_heat,
        actual_cool=actual_cool,
        price=price,
        actual_pv=actual_pv,
        da_plan=da_plan,
        co2_limit=co2_limit,
        penalty_buy_factor=penalty_buy_factor,
        penalty_dump_price=penalty_dump_price,
        soc_dev_margin=soc_dev_margin,
        sell_extra_factor=sell_extra_factor,
        dt=dt                     # <--- 新增：传入时间步长
    )


def solve_ibp_model(actual_demand, actual_heat, actual_cool,
                    price, actual_pv, da_plan,
                    co2_limit, penalty_buy_factor=1.5, penalty_dump_price=0.05,sell_extra_factor = 0.5,
                    soc_dev_margin=0.4,
                    dt=0.25,                           # <--- 新增：时间步长参数
                    co2_grid=0.65, co2_gas=0.2, cop_hp=3.5, cop_c=4.0):
    """
    日内实时平衡模型 (IBP) 核心求解器
    15分钟步长版：所有能量相关约束已加入 dt 因子
    """
    model = ConcreteModel()
    n_steps = len(actual_demand)
    model.T = RangeSet(1, n_steps)                     # 自动适应时段数

    def _to_dict(arr):
        return {i + 1: arr[i] for i in range(len(arr))} if isinstance(arr, (list, tuple, pd.Series)) or hasattr(arr, "shape") else arr

    # ===== 1. 真实参数 =====
    model.demand = Param(model.T, initialize=_to_dict(actual_demand))
    model.heat_demand = Param(model.T, initialize=_to_dict(actual_heat))
    model.cool_demand = Param(model.T, initialize=_to_dict(actual_cool))
    model.price = Param(model.T, initialize=_to_dict(price))
    model.pv = Param(model.T, initialize=_to_dict(actual_pv))

    # ===== 2. 日前计划参数 =====
    model.da_P_chp = Param(model.T, initialize=_to_dict(da_plan["P_chp"]))
    model.da_H_chp = Param(model.T, initialize=_to_dict(da_plan["H_chp"]))
    model.da_F_chp = Param(model.T, initialize=_to_dict(da_plan["F_chp"]))
    model.da_P_buy = Param(model.T, initialize=_to_dict(da_plan["P_buy"]))
    model.da_P_sell = Param(model.T, initialize=_to_dict(da_plan["P_sell"]))
    model.da_SOC = Param(model.T, initialize=_to_dict(da_plan["SOC"]))

    # 3.1 电池（功率单位：kW，能量单位：kWh）
    E_max = 30000.0  # 电池容量 (kWh)  原100 → 10000 (10MWh)
    P_bat_max = 12000.0  # 最大充放电功率 (kW) 原30 → 5000 (5MW)
    eta_ch = 0.98
    eta_dis = 0.98
    SOC_init = 5000.0  # 初始SOC (kWh) 原50 → 5000 (50%)


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

    gas_price, sell_price = 0.35, 0.30

    # ===== 4. 实时可调决策变量 =====
    model.P_ch = Var(model.T, domain=NonNegativeReals, bounds=(0, P_bat_max))
    model.P_dis = Var(model.T, domain=NonNegativeReals, bounds=(0, P_bat_max))
    model.SOC = Var(model.T, bounds=(0, E_max))

    model.H_store_in = Var(model.T, domain=NonNegativeReals, bounds=(0, H_tank_rate))
    model.H_store_out = Var(model.T, domain=NonNegativeReals, bounds=(0, H_tank_rate))
    model.S_heat = Var(model.T, bounds=(0, H_tank_max))

    model.C_store_in = Var(model.T, domain=NonNegativeReals, bounds=(0, C_tank_rate))
    model.C_store_out = Var(model.T, domain=NonNegativeReals, bounds=(0, C_tank_rate))
    model.S_cool = Var(model.T, bounds=(0, C_tank_max))

    model.P_hp = Var(model.T, domain=NonNegativeReals, bounds=(0, 8000))
    model.H_hp = Var(model.T, domain=NonNegativeReals)
    model.P_chiller = Var(model.T, domain=NonNegativeReals, bounds=(0, 10000))
    model.C_chiller = Var(model.T, domain=NonNegativeReals)

    # ===== 5. 紧急惩罚变量 =====
    model.P_buy_extra = Var(model.T, domain=NonNegativeReals)  # 紧急购电
    model.P_dump = Var(model.T, domain=NonNegativeReals)       # 弃电
    model.H_dump = Var(model.T, domain=NonNegativeReals)       # 弃热
    model.C_dump = Var(model.T, domain=NonNegativeReals)       # 弃冷
    # --- 在 Var 定义区域添加 ---
    model.P_sell_extra = Var(model.T, domain=NonNegativeReals)  # 日内额外售电量 (kW)
    # ===== 6. 目标函数 =====
    carryover_value = 0.8

    def real_time_cost(m):
        # 1. 日常基础成本
        base_cost = sum(
            (m.da_P_buy[t] * m.price[t] * dt) -
            (m.da_P_sell[t] * m.price[t] * 0.7 * dt) +
            (m.da_F_chp[t] * gas_price * dt)
            for t in m.T
        )

        # 2. 惩罚成本
        penalty_cost = sum(
            m.P_buy_extra[t] * (m.price[t] * penalty_buy_factor * dt) -
            m.P_sell_extra[t] * (m.price[t] * sell_extra_factor * dt) +
            m.P_dump[t] * (penalty_dump_price * dt) +
            m.H_dump[t] * (0.05 * dt) +
            m.C_dump[t] * (0.05 * dt)
            for t in m.T
        )

        # 3. 单独计算终端价值
        #terminal_reward = (m.SOC[n_steps] - SOC_init) * carryover_value

        # 4. 返回总结果
        return base_cost + penalty_cost

    model.obj = Objective(rule=real_time_cost, sense=minimize)
    # ===== 7. 约束条件 =====
    # 能量平衡（功率瞬时平衡）
    def power_balance_rt(m, t):
        supply = m.da_P_buy[t] + m.P_buy_extra[t] + m.P_dis[t] + m.pv[t] + m.da_P_chp[t]
        demand_side = m.demand[t] + m.P_ch[t] + m.da_P_sell[t] + m.P_sell_extra[t] + m.P_hp[t] + m.P_chiller[t] + m.P_dump[t]
        return supply == demand_side
    model.power_balance = Constraint(model.T, rule=power_balance_rt)

    def heat_balance_rt(m, t):
        return (m.da_H_chp[t] + m.H_hp[t] + m.H_store_out[t] ==
                m.heat_demand[t] + m.H_store_in[t] + m.H_dump[t])
    model.heat_balance = Constraint(model.T, rule=heat_balance_rt)

    def cool_balance_rt(m, t):
        return (m.C_chiller[t] + m.C_store_out[t] ==
                m.cool_demand[t] + m.C_store_in[t] + m.C_dump[t])
    model.cool_balance = Constraint(model.T, rule=cool_balance_rt)

    model.hp_convert = Constraint(model.T, rule=lambda m, t: m.H_hp[t] == cop_hp * m.P_hp[t])
    model.chiller_convert = Constraint(model.T, rule=lambda m, t: m.C_chiller[t] == cop_c * m.P_chiller[t])

    # SOC 动态更新（加入 dt 因子）
    def soc_balance(m, t):
        if t == 1:
            return m.SOC[t] == SOC_init + (eta_ch * m.P_ch[t] - m.P_dis[t] / eta_dis) * dt   # <--- 乘dt
        else:
            return m.SOC[t] == m.SOC[t-1] + (eta_ch * m.P_ch[t] - m.P_dis[t] / eta_dis) * dt # <--- 乘dt
    model.soc_balance = Constraint(model.T, rule=soc_balance)

    # 电池 SOC 路径跟踪约束
    def soc_tracking_upper(m, t):
        return m.SOC[t] <= m.da_SOC[t] + m.da_SOC[t] * soc_dev_margin
    model.soc_track_up = Constraint(model.T, rule=soc_tracking_upper)

    def soc_tracking_lower(m, t):
        return m.SOC[t] >= m.da_SOC[t] - m.da_SOC[t] * soc_dev_margin
    model.soc_track_low = Constraint(model.T, rule=soc_tracking_lower)

    # 储热 SOC 约束（乘 dt）
    def heat_soc_balance(m, t):
        if t == 1:
            return m.S_heat[t] == S_heat_init + (eta_h_in * m.H_store_in[t] - m.H_store_out[t] / eta_h_out) * dt   # <--- 乘dt
        else:
            return m.S_heat[t] == m.S_heat[t-1] + (eta_h_in * m.H_store_in[t] - m.H_store_out[t] / eta_h_out) * dt # <--- 乘dt
    model.heat_soc_balance = Constraint(model.T, rule=heat_soc_balance)

    # 储冷 SOC 约束（乘 dt）
    def cool_soc_balance(m, t):
        if t == 1:
            return m.S_cool[t] == S_cool_init + (eta_c_in * m.C_store_in[t] - m.C_store_out[t] / eta_c_out) * dt   # <--- 乘dt
        else:
            return m.S_cool[t] == m.S_cool[t-1] + (eta_c_in * m.C_store_in[t] - m.C_store_out[t] / eta_c_out) * dt # <--- 乘dt
    model.cool_soc_balance = Constraint(model.T, rule=cool_soc_balance)

    # 终端 SOC 约束（时段数改为 n_steps）
    model.soc_terminal = Constraint(expr=model.SOC[n_steps] == SOC_init)       # <--- 修改：n_steps
    model.heat_terminal = Constraint(expr=model.S_heat[n_steps] == S_heat_init)
    model.cool_terminal = Constraint(expr=model.S_cool[n_steps] == S_cool_init)

    # 碳排放约束（加入 dt 因子）
    def carbon_emission_rule(m):
        total_carbon = sum(
            (m.da_P_buy[t] + m.P_buy_extra[t]) * dt * co2_grid +    # <--- 乘dt
            m.da_F_chp[t] * dt * co2_gas                             # <--- 乘dt
            for t in m.T
        )
        return total_carbon <= co2_limit
    model.carbon_constraint = Constraint(rule=carbon_emission_rule)

    # ===== 8. 求解 =====
    solver = SolverFactory('gurobi', solver_io='python')
    results = solver.solve(model, tee=False)

    if (results.solver.status != SolverStatus.ok) or (
            results.solver.termination_condition != TerminationCondition.optimal):
        print("\n❌ [日内模型] 求解失败，可能是惩罚约束太紧或容忍度过小。")
        return None

    # ====== 输出逐时数据 ======
    hourly_data = {
        "Hour": [t for t in model.T],
        "Price": [value(model.price[t]) for t in model.T],
        "Load_Elec": [value(model.demand[t]) for t in model.T],
        "PV": [value(model.pv[t]) for t in model.T],

        "Buy_Power_DA": [value(model.da_P_buy[t]) for t in model.T],
        "Buy_Power_Extra": [value(model.P_buy_extra[t]) for t in model.T],
        "Buy_Power_Total": [value(model.da_P_buy[t]) + value(model.P_buy_extra[t]) for t in model.T],
        "Sell_Power_DA": [value(model.da_P_sell[t]) for t in model.T],
        "Sell_Power_Extra": [value(model.P_sell_extra[t]) for t in model.T],
        "Dump_Power": [value(model.P_dump[t]) for t in model.T],

        "Battery_Charge": [value(model.P_ch[t]) for t in model.T],
        "Battery_Discharge": [value(model.P_dis[t]) for t in model.T],
        "Battery_SOC": [value(model.SOC[t]) for t in model.T],

        "CHP_Electric": [value(model.da_P_chp[t]) for t in model.T],
        "CHP_Heat": [value(model.da_H_chp[t]) for t in model.T],
        "CHP_Gas": [value(model.da_F_chp[t]) for t in model.T],

        "Heat_Load": [value(model.heat_demand[t]) for t in model.T],
        "HeatPump_Electric": [value(model.P_hp[t]) for t in model.T],
        "HeatPump_Heat": [value(model.H_hp[t]) for t in model.T],
        "Heat_Store_In": [value(model.H_store_in[t]) for t in model.T],
        "Heat_Store_Out": [value(model.H_store_out[t]) for t in model.T],
        "Heat_SOC": [value(model.S_heat[t]) for t in model.T],
        "Heat_Dump": [value(model.H_dump[t]) for t in model.T],

        "Cool_Load": [value(model.cool_demand[t]) for t in model.T],
        "Chiller_Electric": [value(model.P_chiller[t]) for t in model.T],
        "Chiller_Cooling": [value(model.C_chiller[t]) for t in model.T],
        "Cool_Store_In": [value(model.C_store_in[t]) for t in model.T],
        "Cool_Store_Out": [value(model.C_store_out[t]) for t in model.T],
        "Cool_SOC": [value(model.S_cool[t]) for t in model.T],
        "Cool_Dump": [value(model.C_dump[t]) for t in model.T]
    }

    return {
        "Total_Actual_Cost": value(model.obj),
        "P_buy_extra": {t: value(model.P_buy_extra[t]) for t in model.T},
        "P_sell_extra": {t: value(model.P_sell_extra[t]) for t in model.T},
        "P_dump": {t: value(model.P_dump[t]) for t in model.T},
        "Hourly_Data": hourly_data
    }