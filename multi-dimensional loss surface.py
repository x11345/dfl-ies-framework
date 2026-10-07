import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from energy_model import solve_energy_model
from ibp_model import solve_ibp_model

def main():
    print("🚀 开始构建 误差-成本 决策损失曲线（从Excel读取基准数据）...\n")

    # ===== 读取基准真实数据 =====
    INPUT_FILE = "ningde_2022_7_16_elec_heat_cool_pv_price.xlsx"   # <--- 新：基准数据文件
    df_true = pd.read_excel(INPUT_FILE, sheet_name='Timeseries')
    n_steps = len(df_true)
    if n_steps != 96:
        raise ValueError(
            f"当前数据共有 {n_steps} 个时间点，"
            f"但当前时间段划分要求必须为96点。"
        )
    dt = 0.25
    T = range(1, n_steps + 1)

    true_price = {i+1: df_true.loc[i, 'Price'] for i in range(n_steps)}
    true_demand = {i+1: df_true.loc[i, 'Pure_Electric_Load'] for i in range(n_steps)}
    true_heat = {i+1: df_true.loc[i, 'Heat_Load'] for i in range(n_steps)}
    true_cool = {i+1: df_true.loc[i, 'Cool_Load'] for i in range(n_steps)}
    true_pv = {i+1: df_true.loc[i, 'PV'] for i in range(n_steps)}

    co2_limit = 20000000.0

    # ===== 计算基准最优成本 =====
    print("⏳ 正在计算完美预测下的基准最优成本 (C_opt)...")
    optimal_plan = solve_energy_model(
        demand=true_demand, heat_demand=true_heat, cool_demand=true_cool,
        price=true_price, pv=true_pv, co2_limit=co2_limit, dt=dt
    )
    if not optimal_plan:
        print("❌ 基准模型求解失败，请检查初始参数。")
        return
    c_opt = optimal_plan["Cost_Buy_Elec"] - optimal_plan["Rev_Sell_Elec"] + optimal_plan["Cost_Gas_CHP"]
    print(f"✅ 基准最优成本 (C_opt) = {c_opt:.2f} 元\n")

    # ===== 制造双重预测误差并计算决策损失 (2D Surface) =====
    # 建议初始测试时步长设大一点，例如0.05，否则组合数量庞大，求解时间极长
    scale_pv_list = np.arange(0.6, 1.41, 0.02)
    scale_load_list = np.arange(0.6, 1.41, 0.02)
    # ===== 8个等长时间段，每段12个15min点 = 3小时 =====
    time_segments = {
        1: range(1, 13),  # 00:00–03:00
        2: range(13, 25),  # 03:00–06:00
        3: range(25, 37),  # 06:00–09:00
        4: range(37, 49),  # 09:00–12:00
        5: range(49, 61),  # 12:00–15:00
        6: range(61, 73),  # 15:00–18:00
        7: range(73, 85),  # 18:00–21:00
        8: range(85, 97)  # 21:00–24:00
    }

    WINDOW_STEPS = 12
    results_list = []

    # 获取总组合数用于进度监控
    total_iterations = (
            len(time_segments)
            * len(scale_pv_list)
            * len(scale_load_list)
    )
    current_iter = 0

    for segment_id, segment_times in time_segments.items():

        print(f"\n========== 开始计算时间段 {segment_id} ==========")

        for s_pv in scale_pv_list:

            s_pv = round(s_pv, 2)

            for s_load in scale_load_list:

                s_load = round(s_load, 2)

                current_iter += 1

                print(
                    f"🔄 进度 {current_iter}/{total_iterations} | "
                    f"时段 {segment_id} | "
                    f"PV比例 {s_pv:.2f} | "
                    f"Load比例 {s_load:.2f}"
                )

                # ==========================================
                # 1. 构造预测数据
                # 默认所有时刻均为真实值
                # ==========================================
                pred_pv = true_pv.copy()
                pred_demand = true_demand.copy()

                # ==========================================
                # 2. 只在当前3小时窗口注入预测误差
                # ==========================================
                for t in segment_times:
                    pred_pv[t] = true_pv[t] * s_pv

                    pred_demand[t] = (
                            true_demand[t] * s_load
                    )

                # ==========================================
                # 3. 运行日前调度模型
                # ==========================================
                da_plan_raw = solve_energy_model(
                    demand=pred_demand,
                    heat_demand=true_heat,
                    cool_demand=true_cool,
                    price=true_price,
                    pv=pred_pv,
                    co2_limit=co2_limit,
                    dt=dt
                )

                if not da_plan_raw:
                    print(
                        f"   ⚠️ 时段 {segment_id} "
                        f"(PV:{s_pv}, Load:{s_load}) "
                        f"日前无解，跳过。"
                    )

                    continue

                # ==========================================
                # 4. 提取日前计划
                # ==========================================
                da_plan_for_ibp = {

                    "P_buy": [
                        da_plan_raw["P_buy"][t]
                        for t in T
                    ],

                    "P_sell": [
                        da_plan_raw["P_sell"][t]
                        for t in T
                    ],

                    "P_chp": [
                        da_plan_raw["P_chp"][t]
                        for t in T
                    ],

                    "H_chp": [
                        da_plan_raw["H_chp"][t]
                        for t in T
                    ],

                    "F_chp": [
                        da_plan_raw["F_chp"][t]
                        for t in T
                    ],

                    "SOC": [
                        da_plan_raw["SOC"][t]
                        for t in T
                    ]
                }

                # ==========================================
                # 5. 准备真实数据
                # ==========================================
                list_true_demand = [
                    true_demand[t]
                    for t in T
                ]

                list_true_heat = [
                    true_heat[t]
                    for t in T
                ]

                list_true_cool = [
                    true_cool[t]
                    for t in T
                ]

                list_true_price = [
                    true_price[t]
                    for t in T
                ]

                list_true_pv = [
                    true_pv[t]
                    for t in T
                ]

                # ==========================================
                # 6. 运行日内实时模型
                # ==========================================
                rt_run = solve_ibp_model(

                    actual_demand=list_true_demand,

                    actual_heat=list_true_heat,

                    actual_cool=list_true_cool,

                    price=list_true_price,

                    actual_pv=list_true_pv,

                    da_plan=da_plan_for_ibp,

                    co2_limit=co2_limit,

                    penalty_buy_factor=1.5,

                    penalty_dump_price=0.05,

                    sell_extra_factor=0.85,

                    soc_dev_margin=0.4,

                    dt=dt
                )

                # ==========================================
                # 7. 计算决策损失
                # ==========================================
                if not rt_run:

                    decision_loss = np.nan

                    print(
                        "   ⚠️ 日内模型求解失败"
                    )

                else:

                    c_real = (
                        rt_run["Total_Actual_Cost"]
                    )

                    decision_loss = (
                            c_real - c_opt
                    )

                    print(
                        f"   💰 决策损失 "
                        f"(ΔC): {decision_loss:.2f}"
                    )

                    # ======================================
                    # 8. 保存结果
                    # ======================================
                    results_list.append({

                        "Time_Segment": segment_id,

                        "PV_Scale": s_pv,

                        "Load_Scale": s_load,

                        "Decision_Loss": decision_loss
                    })

    # 保存3D曲面数据
    df_results = pd.DataFrame(results_list)
    df_results.to_csv("Joint_Decision_Loss_Surface(ningde2022.7.16)_pv扩容_3d.csv", index=False)
    print("\n✅ 双重误差测试完成！数据已保存。")

    # ===== 分别绘制8个时间段的决策损失曲面 =====

    df_clean = df_results.dropna()

    for segment_id in sorted(df_clean["Time_Segment"].unique()):

        df_seg = df_clean[
            df_clean["Time_Segment"] == segment_id
            ]

        X = np.sort(df_seg["PV_Scale"].unique())
        Y = np.sort(df_seg["Load_Scale"].unique())

        X_mesh, Y_mesh = np.meshgrid(X, Y)

        Z_mesh = np.full_like(X_mesh, np.nan, dtype=float)

        for i in range(len(Y)):
            for j in range(len(X)):

                loss_val = df_seg[
                    (df_seg["PV_Scale"] == X[j]) &
                    (df_seg["Load_Scale"] == Y[i])
                    ]["Decision_Loss"]

                if not loss_val.empty:
                    Z_mesh[i, j] = loss_val.iloc[0]

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(
            X_mesh,
            Y_mesh,
            Z_mesh,
            cmap='viridis',
            edgecolor='none',
            alpha=0.9
        )

        ax.set_title(
            f'Decision Loss Surface - Time Segment {segment_id}'
        )

        ax.set_xlabel('PV Prediction Scale')
        ax.set_ylabel('Load Prediction Scale')
        ax.set_zlabel('Decision Loss (¥)')

        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.savefig(
            f"Decision_Loss_Surface_Segment_{segment_id}.png",
            dpi=150
        )

        plt.show()

if __name__ == "__main__":
    main()