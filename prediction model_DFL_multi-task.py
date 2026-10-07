# =========================================================
# 工业园区 光伏与负荷 多任务联合闭环预测模型 (MTL + Decision Loss)
# 整合了 2D 联合误差决策曲面与动态三方博弈加权机制
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.layers import (
    Input, Dense, Dropout, LSTM, Bidirectional, Conv1D, MaxPooling1D, Attention, Concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# =====================================================
# 加载8个时间段对应的决策损失参数
# =====================================================

DECISION_PARAMS_FILE = \
    "joint_decision_loss_params_time_conditioned.json"

if os.path.exists(DECISION_PARAMS_FILE):

    with open(DECISION_PARAMS_FILE, "r") as f:
        loss_params = json.load(f)

    time_params = loss_params["time_segments"]

    # 按 segment 1~8 转成 TensorFlow 可直接 gather 的数组
    ALPHA_L_TABLE = [
        time_params[str(i)]["load"]["alpha"]
        for i in range(1, 9)
    ]

    BETA_L_TABLE = [
        time_params[str(i)]["load"]["beta"]
        for i in range(1, 9)
    ]

    D_OVER_L_TABLE = [
        time_params[str(i)]["load"]["delta_over"]
        for i in range(1, 9)
    ]

    D_UNDER_L_TABLE = [
        time_params[str(i)]["load"]["delta_under"]
        for i in range(1, 9)
    ]

    ALPHA_P_TABLE = [
        time_params[str(i)]["pv"]["alpha"]
        for i in range(1, 9)
    ]

    BETA_P_TABLE = [
        time_params[str(i)]["pv"]["beta"]
        for i in range(1, 9)
    ]

    D_OVER_P_TABLE = [
        time_params[str(i)]["pv"]["delta_over"]
        for i in range(1, 9)
    ]

    D_UNDER_P_TABLE = [
        time_params[str(i)]["pv"]["delta_under"]
        for i in range(1, 9)
    ]

    C_COUPLING_TABLE = [
        time_params[str(i)]["coupling_C"]
        for i in range(1, 9)
    ]

    print("✅ 成功加载8个时间段的条件化决策损失参数")

else:

    raise FileNotFoundError(
        "找不到 "
        "joint_decision_loss_params_time_conditioned.json"
    )
# ========== [新增核心代码 2: 构建 2D 联合决策损失张量图] ==========
# 设定微小的常数防止除零和夜间光伏异常梯度
EPSILON = 1e-6
PV_NIGHT_THRESHOLD = 0.05  # 光伏容量的 5%，低于此值视为夜间，截断相对误差计算


def calc_asym_huber_tensor(gamma, alpha, beta, d_over, d_under, mask):
    """通用的张量版不对称 Huber 计算"""
    err = gamma - 1.0

    # 高估
    over_mask = tf.cast(err >= 0, tf.float32) * mask
    abs_err_over = tf.abs(err) * over_mask
    loss_over = tf.where(
        abs_err_over <= d_over,
        0.5 * alpha * tf.square(abs_err_over),
        alpha * d_over * (abs_err_over - 0.5 * d_over)
    )

    # 低估
    under_mask = tf.cast(err < 0, tf.float32) * mask
    abs_err_under = tf.abs(err) * under_mask
    loss_under = tf.where(
        abs_err_under <= d_under,
        0.5 * beta * tf.square(abs_err_under),
        beta * d_under * (abs_err_under - 0.5 * d_under)
    )
    return loss_over + loss_under


def joint_decision_loss(y_true_norm, y_pred_norm):
    """计算二维光伏与负荷联合经济决策损失"""
    # 第3列保存当前预测目标所属的时间段
    segment_id = tf.cast(
        y_true_norm[:, 2],
        tf.int32
    )

    # JSON是1~8，TensorFlow索引改为0~7
    segment_idx = segment_id - 1
    alpha_l = tf.gather(
        ALPHA_L_TABLE,
        segment_idx
    )

    beta_l = tf.gather(
        BETA_L_TABLE,
        segment_idx
    )

    d_over_l = tf.gather(
        D_OVER_L_TABLE,
        segment_idx
    )

    d_under_l = tf.gather(
        D_UNDER_L_TABLE,
        segment_idx
    )

    alpha_p = tf.gather(
        ALPHA_P_TABLE,
        segment_idx
    )

    beta_p = tf.gather(
        BETA_P_TABLE,
        segment_idx
    )

    d_over_p = tf.gather(
        D_OVER_P_TABLE,
        segment_idx
    )

    d_under_p = tf.gather(
        D_UNDER_P_TABLE,
        segment_idx
    )

    c_coupling = tf.gather(
        C_COUPLING_TABLE,
        segment_idx
    )
    # 1. 还原物理量纲 (假设目标为 Load 放在列 0，PV 放在列 1)
    y_true_load = (y_true_norm[:, 0] - MIN_LOAD) / SCALE_LOAD
    y_pred_load = (y_pred_norm[:, 0] - MIN_LOAD) / SCALE_LOAD

    y_true_pv = (y_true_norm[:, 1] - MIN_PV) / SCALE_PV
    y_pred_pv = (y_pred_norm[:, 1] - MIN_PV) / SCALE_PV

    # 2. 计算相对误差 Gamma
    gamma_load = y_pred_load / (y_true_load + EPSILON)
    gamma_pv = y_pred_pv / (y_true_pv + EPSILON)

    # 3. 构造夜间掩码 (防止光伏夜间除 0 导致梯度爆炸)
    # 当真实光伏接近 0 时，mask 为 0，阻断光伏相关的决策损失梯度传导
    pv_valid_mask = tf.cast(y_true_pv > PV_NIGHT_THRESHOLD, tf.float32)
    load_valid_mask = tf.ones_like(gamma_load)  # 负荷全天有效

    # 4. 计算独立分量
    loss_load = calc_asym_huber_tensor(
        gamma_load,
        alpha_l,
        beta_l,
        d_over_l,
        d_under_l,
        load_valid_mask
    )
    loss_pv = calc_asym_huber_tensor(
        gamma_pv,
        alpha_p,
        beta_p,
        d_over_p,
        d_under_p,
        pv_valid_mask
    )

    cross_term = (
            c_coupling
            * (gamma_load - 1.0)
            * (gamma_pv - 1.0)
            * pv_valid_mask
    )

    # 6. 总成本非负约束
    total_dec_loss = tf.maximum(0.0, loss_load + loss_pv + cross_term)

    return tf.reduce_mean(total_dec_loss)


# ========== [修改核心代码 3: 动态三方博弈联合损失] ==========
ALIGNMENT_FACTOR = 0.01  # 强制拉平经济量级与归一化误差量级


def combined_mtl_loss(y_true, y_pred):
    # 1. 计算纯统计损失 (Load 用 Huber 抗尖峰，PV 用 MSE 抓平滑)
    stat_loss_load = tf.keras.losses.Huber()(y_true[:, 0], y_pred[:, 0])
    stat_loss_pv = tf.keras.losses.MeanSquaredError()(y_true[:, 1], y_pred[:, 1])
    total_stat_loss = stat_loss_load + stat_loss_pv

    # 2. 计算经济损失并对齐量级
    dec_loss = joint_decision_loss(y_true, y_pred) * ALIGNMENT_FACTOR

    # 3. 三方博弈的梯度截断计算
    total_loss_val = total_stat_loss + dec_loss + 1e-8

    weight_stat = tf.stop_gradient(dec_loss / total_loss_val)
    weight_dec = tf.stop_gradient(total_stat_loss / total_loss_val)

    # 4. 联合梯度回传
    return weight_stat * total_stat_loss + weight_dec * dec_loss


# 辅助度量函数 (用于 TensorBoard 或训练进度输出)
def metric_stat_loss(y_true, y_pred):
    return tf.keras.losses.Huber()(y_true[:, 0], y_pred[:, 0]) + tf.keras.losses.MeanSquaredError()(y_true[:, 1], y_pred[:, 1])


def metric_dec_loss(y_true, y_pred):
    return joint_decision_loss(y_true, y_pred)


# =========================
# 常规数据处理模块 (整合 MTL 版本)
# =========================
print("-> 读取数据与特征工程...")
file_path = "宁德通用制造业21.3.23-22.8.31_heat_cool_pv.xlsx"
df = pd.read_excel(file_path)
df.columns = df.columns.str.strip()
df["time"] = pd.to_datetime(df["time"])
# =====================================================
# 根据一天中的时间确定所属时间段
# 1~8，每段3小时
# =====================================================

def get_time_segment(timestamp):

    minutes = (
        timestamp.hour * 60
        + timestamp.minute
    )

    segment_id = minutes // 180 + 1

    # 防止24:00边界问题
    return min(segment_id, 8)


time_segment_all = df["time"].apply(
    get_time_segment
).values
df.fillna(method='ffill', inplace=True)

df["hour"] = df["time"].dt.hour
df["is_workday"] = (df["time"].dt.weekday < 5).astype(int)
df["future_irradiance"] = df["global_tilted_irradiance_instant"].shift(-1)
df["future_temperature"] = df["temperature_2m"].shift(-1)
df = df.dropna().reset_index(drop=True)

# 确保目标变量在最前两列
features = [
    "load", "pv", "temperature_2m", "future_irradiance", "future_temperature", "is_workday"
]
data = df[features].values

train_size = int(len(data) * 0.9)
train_data, test_data = data[:train_size], data[train_size:]
train_segments = time_segment_all[:train_size]
test_segments = time_segment_all[train_size:]
scaler = MinMaxScaler(feature_range=(0, 1))  # 注意这里改为 0-1
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

# 提取反归一化常量供张量图使用
SCALE_LOAD = scaler.scale_[0]
MIN_LOAD = scaler.min_[0]
SCALE_PV = scaler.scale_[1]
MIN_PV = scaler.min_[1]

TIME_STEPS = 96


def create_mtl_dataset(
    dataset,
    time_steps,
    time_segments
):

    X = []
    y = []

    for i in range(len(dataset) - time_steps):

        target_index = i + time_steps

        X.append(
            dataset[i:target_index, :]
        )

        # Load、PV
        y.append([
            dataset[target_index, 0],
            dataset[target_index, 1],

            # 时间段编号
            time_segments[target_index]
        ])

    return np.array(X), np.array(y)


X_train, y_train = create_mtl_dataset(
    train_scaled,
    TIME_STEPS,
    train_segments
)

X_test, y_test = create_mtl_dataset(
    test_scaled,
    TIME_STEPS,
    test_segments
)

# ========== [修改核心代码 4: MTL 架构合并为联合输出层] ==========
inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))

# 共享主干
x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(inputs)
x = MaxPooling1D(pool_size=2)(x)
x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = Dropout(0.2)(x)
shared_features = Bidirectional(LSTM(32, return_sequences=True))(x)

# 负荷分支
att_load = Attention()([shared_features, shared_features])
x_load = att_load[:, -1, :]
x_load = Dense(32, activation='relu')(x_load)

# 光伏分支
att_pv = Attention()([shared_features, shared_features])
x_pv = att_pv[:, -1, :]
x_pv = Dense(32, activation='relu')(x_pv)

# 【关键】将双分支拼接，实现形状为 (batch, 2) 的联合输出，适配联合损失张量
merged_features = Concatenate()([x_load, x_pv])
outputs = Dense(2, name='joint_output')(merged_features)

model = Model(inputs, outputs)

# =====================================================
# Stage 1：开环特征预训练 (纯统计寻优)
# =====================================================
print("\n==============================")
print("Stage 1：MTL 统计特征预训练 (MSE + Huber)")
print("==============================")


# 在 Stage 1 我们只用纯统计误差，建立物理基准
def stage1_stat_loss(y_true, y_pred):
    return tf.keras.losses.Huber()(y_true[:, 0], y_pred[:, 0]) + tf.keras.losses.MeanSquaredError()(y_true[:, 1], y_pred[:, 1])


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=stage1_stat_loss,
    metrics=[metric_stat_loss, metric_dec_loss]
)

early_stop_stage1 = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history_stage1 = model.fit(
    X_train, y_train, epochs=10, batch_size=64, validation_split=0.1,
    callbacks=[early_stop_stage1], verbose=1
)
model.save_weights("stage1_mtl_pretrain.weights.h5")

# =====================================================
# Stage 2：闭环联合决策微调 (经济向漂移)
# =====================================================
print("\n==============================")
print("Stage 2：双重不确定性闭环微调 (2D 经济代价引入)")
print("==============================")

model.load_weights("stage1_mtl_pretrain.weights.h5")

# 切换为我们编写的包含交叉项的动态博弈损失函数
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss=combined_mtl_loss,
    metrics=[metric_stat_loss, metric_dec_loss]
)

early_stop_stage2 = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history_stage2 = model.fit(
    X_train, y_train, epochs=30, batch_size=64, validation_split=0.1,
    callbacks=[early_stop_stage2], verbose=1
)

import seaborn as sns

# =====================================================
# 准备阶段：执行预测与反归一化 (Stage 1 & Stage 2)
# =====================================================
print("\n-> 正在执行预测与数据反归一化...")

# 1. 定义统一的双输出反归一化函数
def inverse_transform_2d(scaler, y_pred_matrix, n_features):
    temp_matrix = np.zeros((len(y_pred_matrix), n_features))
    temp_matrix[:, 0] = y_pred_matrix[:, 0]  # Load
    temp_matrix[:, 1] = y_pred_matrix[:, 1]  # PV
    inversed = scaler.inverse_transform(temp_matrix)
    return inversed[:, 0], inversed[:, 1]

# 2. 获取 Stage 2 (闭环) 预测结果并反归一化
# 注意：此时模型内存里装的刚好是 Stage 2 训练完的最新权重
predictions_stage2 = model.predict(X_test)
y_pred_load, y_pred_pv = inverse_transform_2d(scaler, predictions_stage2, len(features))
y_test_target = y_test[:, :2]

y_test_load, y_test_pv = inverse_transform_2d(
    scaler,
    y_test_target,
    len(features)
)

# =====================================================
# PV物理约束：夜间无光伏出力
# =====================================================

# 1. 光伏非负
y_pred_pv[y_pred_pv < 0] = 0

# 2. 获取测试集对应时间
start_idx = train_size + TIME_STEPS
test_times = df["time"].iloc[
    start_idx : start_idx + len(y_pred_pv)
].reset_index(drop=True)

# 3. 夜间判断：19:30 ~ 次日05:30
test_hours = (
    test_times.dt.hour
    + test_times.dt.minute / 60.0
)

night_mask = (
    (test_hours >= 19)
    | (test_hours < 5.5)
)

# 4. 夜间PV强制为0
y_pred_pv[night_mask] = 0

# 3. 获取 Stage 1 (开环) 预测结果并反归一化，用于画图对比
print("-> 提取 Stage 1 预测结果以构建对比图表...")
model.load_weights("stage1_mtl_pretrain.weights.h5")
predictions_stage1 = model.predict(X_test, verbose=0)
y_pred_load_s1, y_pred_pv_s1 = inverse_transform_2d(scaler, predictions_stage1, len(features))

# 物理约束：光伏非负 (Stage 1)
y_pred_pv_s1[y_pred_pv_s1 < 0] = 0
y_pred_pv_s1[night_mask] = 0
# 4. 提前计算好物理误差，供后续画分布图使用 (正数 = 高估，负数 = 低估)
err_load_s1 = y_pred_load_s1 - y_test_load
err_load_s2 = y_pred_load - y_test_load

err_pv_s1 = y_pred_pv_s1 - y_test_pv
err_pv_s2 = y_pred_pv - y_test_pv

# =====================================================
# 模块一：负荷预测 - 传统评估指标
# =====================================================
print("\n========== [1] 负荷预测 (Load) 传统评估指标 ==========")
mae_load = mean_absolute_error(y_test_load, y_pred_load)
rmse_load = np.sqrt(mean_squared_error(y_test_load, y_pred_load))
# 负荷通常不为0，加微小 epsilon 防除零
mape_load = np.mean(np.abs((y_test_load - y_pred_load) / (y_test_load + 1e-6))) * 100
r2_load = r2_score(y_test_load, y_pred_load)

print(f"MAE : {mae_load:.2f} kW")
print(f"RMSE: {rmse_load:.2f} kW")
print(f"MAPE: {mape_load:.2f} %")
print(f"R²  : {r2_load:.3f}")


# =====================================================
# 模块二：负荷预测 - 深度可视化
# =====================================================
# 2.1 负荷时序拟合曲线
plt.figure(figsize=(14, 5))
plot_range = min(96 * 3, len(y_test_load))  # 展示约3天的数据
plt.plot(y_test_load[:plot_range], label="Actual Load", color='steelblue', linewidth=1.5)
plt.plot(y_pred_load_s1[:plot_range], label="Predicted Load (Stage 1: Open-Loop)", color='gray', linestyle='-.', alpha=0.7)
plt.plot(y_pred_load[:plot_range], label="Predicted Load (Stage 2: Closed-Loop)", color='darkorange', linestyle='--', linewidth=2)
plt.title("Load Forecast: Actual vs Predicted (Time Series)")
plt.xlabel("Time Step (15-min)")
plt.ylabel("Load (kW)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 2.2 负荷误差分布偏移 (核心闭环证明)
plt.figure(figsize=(10, 5))
sns.histplot(err_load_s1, bins=60, kde=True, color='gray', alpha=0.4, label='Stage 1 (Statistical Focus)')
sns.histplot(err_load_s2, bins=60, kde=True, color='red', alpha=0.5, label='Stage 2 (Decision Focus)')
plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
plt.title("Load Prediction Error Distribution Shift\n(Right = Overestimation, Left = Underestimation)")
plt.xlabel("Load Error (kW)")
plt.ylabel("Density")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 2.3 负荷散点图
plt.figure(figsize=(6, 6))
plt.scatter(y_test_load, y_pred_load_s1, alpha=0.3, color='gray', label='Stage 1', s=10)
plt.scatter(y_test_load, y_pred_load, alpha=0.4, color='darkorange', label='Stage 2', s=10)
min_val, max_val = min(np.min(y_test_load), np.min(y_pred_load)), max(np.max(y_test_load), np.max(y_pred_load))
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='y = x')
plt.title("Load: Actual vs. Predicted")
plt.xlabel("Actual Load (kW)")
plt.ylabel("Predicted Load (kW)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()


# =====================================================
# 模块三：光伏预测 - 传统评估指标
# =====================================================
print("\n========== [3] 光伏预测 (PV) 传统评估指标 ==========")
# 【关键学术处理】：排除夜间 0 值或极小值计算 MAPE，否则数值会爆炸
pv_valid_mask = y_test_pv > (SCALE_PV * 0.01) # 真实发电量大于装机容量 1% 的时刻才计入 MAPE

mae_pv = mean_absolute_error(y_test_pv, y_pred_pv)
rmse_pv = np.sqrt(mean_squared_error(y_test_pv, y_pred_pv))
mape_pv = np.mean(np.abs((y_test_pv[pv_valid_mask] - y_pred_pv[pv_valid_mask]) / y_test_pv[pv_valid_mask])) * 100
r2_pv = r2_score(y_test_pv, y_pred_pv)

print(f"MAE : {mae_pv:.2f} kW")
print(f"RMSE: {rmse_pv:.2f} kW")
print(f"MAPE: {mape_pv:.2f} % (已屏蔽夜间无效时段)")
print(f"R²  : {r2_pv:.3f}")


# =====================================================
# 模块四：光伏预测 - 深度可视化
# =====================================================
# 4.1 光伏时序拟合曲线
plt.figure(figsize=(14, 5))
plt.plot(y_test_pv[:plot_range], label="Actual PV", color='forestgreen', linewidth=1.5)
plt.plot(y_pred_pv_s1[:plot_range], label="Predicted PV (Stage 1: Open-Loop)", color='gray', linestyle='-.', alpha=0.7)
plt.plot(y_pred_pv[:plot_range], label="Predicted PV (Stage 2: Closed-Loop)", color='gold', linestyle='--', linewidth=2)
plt.title("PV Forecast: Actual vs Predicted (Time Series)")
plt.xlabel("Time Step (15-min)")
plt.ylabel("PV Generation (kW)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 4.2 光伏误差分布偏移
plt.figure(figsize=(10, 5))
# 仅对白天发电时段的误差进行分布统计，更具物理意义
err_pv_s1_day = err_pv_s1[pv_valid_mask]
err_pv_s2_day = err_pv_s2[pv_valid_mask]
sns.histplot(err_pv_s1_day, bins=60, kde=True, color='gray', alpha=0.4, label='Stage 1 (Statistical Focus)')
sns.histplot(err_pv_s2_day, bins=60, kde=True, color='gold', alpha=0.6, label='Stage 2 (Decision Focus)')
plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
plt.title("PV Prediction Error Distribution Shift (Daylight Hours Only)\n(Right = Overestimation, Left = Underestimation)")
plt.xlabel("PV Error (kW)")
plt.ylabel("Density")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 4.3 光伏散点图
plt.figure(figsize=(6, 6))
plt.scatter(y_test_pv, y_pred_pv_s1, alpha=0.3, color='gray', label='Stage 1', s=10)
plt.scatter(y_test_pv, y_pred_pv, alpha=0.4, color='forestgreen', label='Stage 2', s=10)
min_pv_val, max_pv_val = min(np.min(y_test_pv), np.min(y_pred_pv)), max(np.max(y_test_pv), np.max(y_pred_pv))
plt.plot([min_pv_val, max_pv_val], [min_pv_val, max_pv_val], 'k--', lw=2, label='y = x')
plt.title("PV: Actual vs. Predicted")
plt.xlabel("Actual PV (kW)")
plt.ylabel("Predicted PV (kW)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()


# =====================================================
# 最终数据导出 (保持原样)
# =====================================================
print("\n========== 正在导出双维预测数据 ==========")
start_idx = train_size + TIME_STEPS
test_times = df["time"].iloc[start_idx : start_idx + len(y_pred_load)].reset_index(drop=True)

result_df = pd.DataFrame({
    "Time": test_times,
    "Actual_Load": y_test_load,
    "Predicted_Load": y_pred_load,
    "Actual_PV": y_test_pv,
    "Predicted_PV": y_pred_pv
})

output_file = "Dual_ClosedLoop_Forecast_Results_6.9_pv扩容参数（无pv误差）_3d.xlsx"
result_df.to_excel(output_file, index=False)
print(f"✅ 双重不确定性闭环预测结束！结果已保存至: {output_file}")