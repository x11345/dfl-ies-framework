import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json


def asym_huber_1d(gamma, alpha, beta, delta_over, delta_under):
    """单变量的不对称 Huber 核心逻辑"""
    err = gamma - 1.0
    if err >= 0:  # 高估
        if err <= delta_over:
            return 0.5 * alpha * (err ** 2)
        else:
            return alpha * delta_over * (err - 0.5 * delta_over)
    else:  # 低估
        abs_err = abs(err)
        if abs_err <= delta_under:
            return 0.5 * beta * (abs_err ** 2)
        else:
            return beta * delta_under * (abs_err - 0.5 * delta_under)


def joint_loss_surface(X, alpha_p, beta_p, d_over_p, d_under_p,
                       alpha_l, beta_l, d_over_l, d_under_l, C):
    """
    二维联合损失函数，X 是一个形状为 (2, N) 的数组
    X[0] 是 gamma_pv (光伏预测比例)
    X[1] 是 gamma_load (负荷预测比例)
    """
    gamma_pv = X[0]
    gamma_load = X[1]

    losses = np.zeros(len(gamma_pv))

    for i in range(len(gamma_pv)):
        # 1. 光伏的独立损失
        loss_pv = asym_huber_1d(gamma_pv[i], alpha_p, beta_p, d_over_p, d_under_p)
        # 2. 负荷的独立损失
        loss_load = asym_huber_1d(gamma_load[i], alpha_l, beta_l, d_over_l, d_under_l)
        # 3. 耦合项 (交叉项)
        err_pv = gamma_pv[i] - 1.0
        err_load = gamma_load[i] - 1.0
        cross_term = C * err_pv * err_load

        # 保证拟合的 loss 基础大于等于 0 (经济损失非负)
        losses[i] = max(0, loss_pv + loss_load + cross_term)

    return losses


def main():
    print("🚀 开始拟合 2D 误差-成本决策损失曲面...\n")

    # 1. 读取数据
    csv_file = "Joint_Decision_Loss_Surface(ningde2022.7.16)_pv扩容_3d.csv"
    df = pd.read_csv(csv_file).dropna()
    # =====================================================
    # 对8个时间段分别拟合决策损失函数
    # =====================================================

    all_params = {}

    # 初始猜测
    initial_guess = [
        1.0, 1.0, 0.1, 0.1,
        1.0, 1.0, 0.1, 0.1,
        0.0
    ]

    # 参数边界
    bounds = (
        [0.0, 0.0, 0.01, 0.01,
         0.0, 0.0, 0.01, 0.01,
         -np.inf],

        [np.inf, np.inf, 0.5, 0.5,
         np.inf, np.inf, 0.5, 0.5,
         np.inf]
    )

    for segment_id in sorted(df["Time_Segment"].unique()):
        print(f"\n========== 拟合时间段 {segment_id} ==========")

        df_seg = df[
            df["Time_Segment"] == segment_id
            ].dropna()

        X_pv = df_seg["PV_Scale"].values
        X_load = df_seg["Load_Scale"].values
        Y_loss_raw = df_seg["Decision_Loss"].values

        X_data = np.vstack((X_pv, X_load))

        y_scale_factor = (
            Y_loss_raw.max()
            if Y_loss_raw.max() > 0
            else 1.0
        )

        Y_loss_scaled = Y_loss_raw / y_scale_factor

        popt, pcov = curve_fit(
            joint_loss_surface,
            X_data,
            Y_loss_scaled,
            p0=initial_guess,
            bounds=bounds,
            maxfev=20000
        )

        # 恢复经济损失量纲
        ap = popt[0] * y_scale_factor
        bp = popt[1] * y_scale_factor
        dop = popt[2]
        dup = popt[3]

        al = popt[4] * y_scale_factor
        bl = popt[5] * y_scale_factor
        dol = popt[6]
        dul = popt[7]

        C_opt = popt[8] * y_scale_factor

        print(
            f"[PV] α={ap:.4f}, β={bp:.4f}, "
            f"D_over={dop:.4f}, D_under={dup:.4f}"
        )

        print(
            f"[Load] α={al:.4f}, β={bl:.4f}, "
            f"D_over={dol:.4f}, D_under={dul:.4f}"
        )

        print(f"[Coupling] C={C_opt:.4f}")

        all_params[str(segment_id)] = {
            "pv": {
                "alpha": ap,
                "beta": bp,
                "delta_over": dop,
                "delta_under": dup
            },

            "load": {
                "alpha": al,
                "beta": bl,
                "delta_over": dol,
                "delta_under": dul
            },

            "coupling_C": C_opt
        }


    print("✅ 拟合成功！提取参数如下：")
    print(f"  [光伏] Alpha: {ap:.2f}, Beta: {bp:.2f}, D_over: {dop:.2f}, D_under: {dup:.2f}")
    print(f"  [负荷] Alpha: {al:.2f}, Beta: {bl:.2f}, D_over: {dol:.2f}, D_under: {dul:.2f}")
    print(f"  [耦合] 交叉系数 C: {C_opt:.2f}")

    # 5. 保存至 JSON 供深度学习框架调用
    params_dict = {
        "time_segments": all_params
    }

    with open(
            "joint_decision_loss_params_time_conditioned.json",
            "w"
    ) as f:

        json.dump(
            params_dict,
            f,
            indent=4
        )

    print(
        "✅ 8个时间段的决策损失参数已保存"
    )
    # =========================================================
    # 6. 可视化验证
    #    对8个时间段分别绘制：
    #    原始仿真数据 + 拟合决策损失曲面
    # =========================================================

    print("\n📊 正在生成8个时间段的3D可视化对比图...")

    for segment_id in sorted(df["Time_Segment"].unique()):

        print(f"   → 正在绘制时间段 {segment_id}")

        # -----------------------------------------------------
        # 1. 提取当前时间段的数据
        # -----------------------------------------------------
        df_seg = df[
            df["Time_Segment"] == segment_id
            ].dropna()

        X_pv_seg = df_seg["PV_Scale"].values
        X_load_seg = df_seg["Load_Scale"].values
        Y_loss_seg = df_seg["Decision_Loss"].values

        # -----------------------------------------------------
        # 2. 获取当前时间段对应的拟合参数
        # -----------------------------------------------------
        params = all_params[str(segment_id)]

        ap = params["pv"]["alpha"]
        bp = params["pv"]["beta"]
        dop = params["pv"]["delta_over"]
        dup = params["pv"]["delta_under"]

        al = params["load"]["alpha"]
        bl = params["load"]["beta"]
        dol = params["load"]["delta_over"]
        dul = params["load"]["delta_under"]

        C_seg = params["coupling_C"]

        # -----------------------------------------------------
        # 3. 注意：
        #    重新使用当前时间段自己的 y_scale_factor
        # -----------------------------------------------------
        y_scale_seg = (
            Y_loss_seg.max()
            if Y_loss_seg.max() > 0
            else 1.0
        )

        # curve_fit中的参数需要重新缩放回去
        popt_seg = np.array([
            ap / y_scale_seg,
            bp / y_scale_seg,
            dop,
            dup,
            al / y_scale_seg,
            bl / y_scale_seg,
            dol,
            dul,
            C_seg / y_scale_seg
        ])

        # -----------------------------------------------------
        # 4. 建立平滑网格
        # -----------------------------------------------------
        pv_grid = np.linspace(
            X_pv_seg.min(),
            X_pv_seg.max(),
            50
        )

        load_grid = np.linspace(
            X_load_seg.min(),
            X_load_seg.max(),
            50
        )

        PV_mesh, LOAD_mesh = np.meshgrid(
            pv_grid,
            load_grid
        )

        # -----------------------------------------------------
        # 5. 计算当前时间段的拟合曲面
        # -----------------------------------------------------
        Z_mesh = np.zeros_like(PV_mesh)

        for i in range(PV_mesh.shape[0]):
            X_row = np.vstack([
                PV_mesh[i, :],
                LOAD_mesh[i, :]
            ])

            Z_row_scaled = joint_loss_surface(
                X_row,
                *popt_seg
            )

            Z_mesh[i, :] = (
                    Z_row_scaled * y_scale_seg
            )

        # -----------------------------------------------------
        # 6. 绘制3D图
        # -----------------------------------------------------
        fig = plt.figure(
            figsize=(12, 8)
        )

        ax = fig.add_subplot(
            111,
            projection='3d'
        )

        # 原始仿真数据
        ax.scatter(
            X_pv_seg,
            X_load_seg,
            Y_loss_seg,
            color='black',
            alpha=0.4,
            s=20,
            label='Raw Simulated Points'
        )

        # 拟合曲面
        surf = ax.plot_surface(
            PV_mesh,
            LOAD_mesh,
            Z_mesh,
            cmap='viridis',
            edgecolor='none',
            alpha=0.7
        )

        # 完美预测点
        ax.scatter(
            [1.0],
            [1.0],
            [0.0],
            color='red',
            s=100,
            marker='*',
            zorder=10,
            label='Perfect Prediction'
        )

        # -----------------------------------------------------
        # 7. 根据时间段设置标题
        # -----------------------------------------------------
        time_labels = {
            1: "00:00–03:00",
            2: "03:00–06:00",
            3: "06:00–09:00",
            4: "09:00–12:00",
            5: "12:00–15:00",
            6: "15:00–18:00",
            7: "18:00–21:00",
            8: "21:00–24:00"
        }

        time_label = time_labels.get(
            int(segment_id),
            f"Segment {segment_id}"
        )

        ax.set_title(
            f'Joint Decision Loss Surface\n'
            f'Time Segment {segment_id} ({time_label})',
            fontsize=14
        )

        ax.set_xlabel(
            'PV Prediction Scale ($\\gamma_{pv}$)',
            fontsize=12
        )

        ax.set_ylabel(
            'Load Prediction Scale ($\\gamma_{load}$)',
            fontsize=12
        )

        ax.set_zlabel(
            'Decision Loss / Extra Cost (¥)',
            fontsize=12
        )

        # -----------------------------------------------------
        # 8. 调整视角
        # -----------------------------------------------------
        ax.view_init(
            elev=25,
            azim=-45
        )

        fig.colorbar(
            surf,
            shrink=0.5,
            aspect=10,
            pad=0.1,
            label='Fitted Loss Magnitude'
        )

        plt.legend()
        plt.tight_layout()

        # -----------------------------------------------------
        # 9. 保存当前时间段图片
        # -----------------------------------------------------
        save_name = (
            f"Joint_Decision_Loss_Fit_"
            f"Segment_{segment_id}.png"
        )

        plt.savefig(
            save_name,
            dpi=300
        )

        plt.show()

        plt.close(fig)

    print(
        "\n✅ 8个时间段的可视化完成！"
    )

if __name__ == "__main__":
    main()