"""题二：用 PyTorch 自动求导 + 梯度下降拟合二阶函数（最小二乘）。

真实函数：y = 2*x1^2 + 1.5*x1*x2 + 3*sin(x1) + 0.5*cos(2*x2) + e^(-0.5*x1) + 0.8*x2 + 2026 + 噪声
拟合模型：y = a1*x1^2 + a2*x1*x2 + a3*sin(x1) + a4*cos(2*x2) + e^(a5*x1) + a6*x2 + b
输出：拟合曲面、loss 曲线、参数对比、R²。
"""
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无显示器环境下必须用 Agg 后端，否则 plt.show 会阻塞/报错
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

torch.set_num_threads(8)  # CPU 上与别的任务（YOLO 训练）抢核，限制线程数避免互相拖慢
sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK，中文和 R² 会直接报编码错误
torch.manual_seed(2026)
np.random.seed(2026)

# 中文字体，否则图里的中文全是方框
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============ 1. 造数据 ============
SAMPLE_NUM = 5000
x1_np = np.random.uniform(-3.0, 3.0, SAMPLE_NUM)  # 均匀采样，保证 x1^2 / sin / exp 各段都被覆盖
x2_np = np.random.uniform(-3.0, 3.0, SAMPLE_NUM)
noise_np = np.random.uniform(-3.0, 3.0, SAMPLE_NUM)  # 题目要求的 [-3, 3] 均匀噪声

# 真实参数，最后要和学出来的对比
a1_true, a2_true, a3_true, a4_true, a5_true, a6_true, b_true = 2.0, 1.5, 3.0, 0.5, -0.5, 0.8, 2026.0


def true_function(x1, x2):
    """不含噪声的原函数，用来画真值图、算 R²。"""
    return (a1_true * x1 ** 2 + a2_true * x1 * x2 + a3_true * np.sin(x1)
            + a4_true * np.cos(2 * x2) + np.exp(a5_true * x1) + a6_true * x2 + b_true)


y_clean_np = true_function(x1_np, x2_np)  # 无噪声真值
y_true_np = y_clean_np + noise_np         # 训练用的观测值

# 用 float64：目标值在 2026 量级，float32 只有约 7 位有效数字，
# 2026 旁边的小数会被舍掉，拟合精度直接损失掉，所以这里统一用 float64。
x1 = torch.tensor(x1_np, dtype=torch.float64)
x2 = torch.tensor(x2_np, dtype=torch.float64)
y_true = torch.tensor(y_true_np, dtype=torch.float64)

# ============ 2. 把 7 个待学参数定义成 nn.Parameter ============
# 一旦是 nn.Parameter（requires_grad=True），前向算出的 loss 就能自动反传到它们身上
a1 = nn.Parameter(torch.zeros(1, dtype=torch.float64))
a2 = nn.Parameter(torch.zeros(1, dtype=torch.float64))
a3 = nn.Parameter(torch.zeros(1, dtype=torch.float64))
a4 = nn.Parameter(torch.zeros(1, dtype=torch.float64))
a5 = nn.Parameter(torch.zeros(1, dtype=torch.float64))
a6 = nn.Parameter(torch.zeros(1, dtype=torch.float64))
# b 从 y 的均值起步：目标在 2026 附近，若从 0 开始，Adam 每步只挪 ≈ 学习率，
# 要上千步才爬到 2026，白白浪费迭代次数。
b = nn.Parameter(torch.tensor([float(y_true.mean())], dtype=torch.float64))

params = [a1, a2, a3, a4, a5, a6, b]


def model(x1, x2):
    """拟合模型：结构按题目给定，参数全部可学。"""
    return (a1 * x1 ** 2 + a2 * x1 * x2 + a3 * torch.sin(x1)
            + a4 * torch.cos(2 * x2) + torch.exp(a5 * x1) + a6 * x2 + b)


# ============ 3. 定义损失 + 优化器 + 训练循环 ============
# 最小二乘 = 均方误差；开方只影响数值大小不影响最优解，所以直接用 MSE
loss_fn = nn.MSELoss()
# Adam 对 7 个参数的量级差异（a 是 0~3，b 是 2026）自适应，不用手工调每个参数的学习率；
# 纯 SGD 在这么悬殊的尺度下要么 b 不收敛要么 a 爆炸。
optimizer = torch.optim.Adam(params, lr=0.1)

EPOCHS = 2000  # 实测 ~130 步 loss 就掉到噪声方差附近，再往后只是极小邻域内的微调
loss_history = []
t_start = time.time()

for epoch in range(EPOCHS):
    y_pred = model(x1, x2)
    loss = loss_fn(y_pred, y_true)

    optimizer.zero_grad()   # 梯度会累加，每步必须先清零
    loss.backward()         # 反向传播：链式法则自动算出 loss 对 7 个参数的偏导
    optimizer.step()        # 沿负梯度方向更新参数

    loss_history.append(loss.item())
    if (epoch + 1) % 500 == 0:
        print(f"epoch {epoch + 1:5d}  loss = {loss.item():.6f}")

train_seconds = time.time() - t_start

# ============ 4. 打印参数对比 + R² ============
a1_v, a2_v, a3_v, a4_v, a5_v, a6_v, b_v = [p.item() for p in params]
names = ["a1", "a2", "a3", "a4", "a5", "a6", "b"]
fitted = [a1_v, a2_v, a3_v, a4_v, a5_v, a6_v, b_v]
truth = [a1_true, a2_true, a3_true, a4_true, a5_true, a6_true, b_true]

print("\n参数对比（真实值 vs 梯度下降学到的值）")
print(f"{'参数':<6}{'真实值':>12}{'拟合值':>14}{'相对误差':>12}")
for name, fit_v, true_v in zip(names, fitted, truth):
    rel = abs(fit_v - true_v) / max(abs(true_v), 1e-12)
    print(f"{name:<6}{true_v:>12.6f}{fit_v:>14.6f}{rel:>11.2%}")

# R² = 1 - SS_res / SS_tot，SS_tot 用观测值 y 的均值算（对"带噪声数据"的标准定义）
with torch.no_grad():  # 评估阶段不需要建图，关掉可以省内存/时间
    y_fit = model(x1, x2)
    y_fit_np = y_fit.numpy()
ss_res = float(np.sum((y_true_np - y_fit_np) ** 2))
ss_tot = float(np.sum((y_true_np - y_true_np.mean()) ** 2))
r2 = 1.0 - ss_res / ss_tot

print(f"\nR² = {r2:.6f}")
print(f"RMSE = {np.sqrt(ss_res / SAMPLE_NUM):.6f}   （噪声标准差约 {np.sqrt(3.0):.4f}，说明已拟合到噪声水平）")
print(f"训练 {EPOCHS} 步，耗时 {train_seconds:.2f} 秒")

# ============ 5. 画图 ============
# 5.1 真实曲面 vs 拟合曲面（并排两张 3D 图）
grid = np.linspace(-3.0, 3.0, 50)
gx1, gx2 = np.meshgrid(grid, grid)
gz_true = true_function(gx1, gx2)  # 不含噪声的真实函数
with torch.no_grad():
    gz_fit = model(torch.tensor(gx1, dtype=torch.float64),
                   torch.tensor(gx2, dtype=torch.float64)).numpy()

z_min = min(gz_true.min(), gz_fit.min())
z_max = max(gz_true.max(), gz_fit.max())

fig = plt.figure(figsize=(14, 5.6))
for idx, (gz, title) in enumerate([(gz_true, "真实函数（不含噪声）"),
                                  (gz_fit, "PyTorch 拟合函数")], start=1):
    ax = fig.add_subplot(1, 2, idx, projection="3d")
    surf = ax.plot_surface(gx1, gx2, gz, cmap="viridis", vmin=z_min, vmax=z_max,
                           rcount=50, ccount=50,  # 默认只抽 50 行且步长不整齐，曲面会出现锯齿接缝
                           linewidth=0, antialiased=True)
    ax.set_zlim(z_min, z_max)  # 两张图锁同一 z 轴范围，才能一眼看出形状是否一致
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    ax.set_title(title)
    ax.view_init(elev=65, azim=-125)  # 视角压低时曲面会自遮挡，mplot3d 按面片排序会出现"撕裂"接缝，抬高视角可避开
fig.colorbar(surf, ax=fig.axes, shrink=0.6, pad=0.02)  # 两张图共用一条色标，颜色才能直接对比
fig.suptitle(f"题二：真实函数与拟合函数对比（R² = {r2:.4f}）", fontsize=14)
fig.savefig(OUT_DIR / "q2_ground_truth_vs_fit.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# 5.2 loss 曲线（loss 从 60 多掉到 3，跨数量级，横纵轴都用对数才看得清下降过程和平台）
fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.plot(np.arange(1, EPOCHS + 1), loss_history, color="tab:blue", linewidth=1.2)
ax.axhline(3.0, color="tab:red", linestyle="--", linewidth=1.0,
           label="理论下界 = 噪声方差 3.0")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("iteration (log scale)")
ax.set_ylabel("MSE loss (log scale)")
ax.set_title("题二：梯度下降的损失曲线")
ax.legend()
ax.grid(True, alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(OUT_DIR / "q2_loss.png", dpi=200)
plt.close(fig)

print(f"\n图片已保存到：{OUT_DIR}")
print("  - q2_ground_truth_vs_fit.png")
print("  - q2_loss.png")
