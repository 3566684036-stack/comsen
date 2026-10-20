"""题三：用 PyTorch 搭建 MLP 识别 MNIST 手写数字（纯 CPU）。

数据：本地 parquet（60000 张 28x28 灰度图，image 列是 PNG 字节流），没有官方测试集，
      所以自己按 50000 / 10000 划分训练集与测试集。
输出：训练曲线（Loss / Train Acc / Test Acc）、16 张识别结果可视化。
"""
import io
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无显示器环境，用非交互后端直接存图
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(8)  # CPU 上还有别的任务在跑，限制线程数避免互相抢核
sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK，中文输出会报编码错误
torch.manual_seed(2026)
np.random.seed(2026)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[2]  # 往上三层就是"算法组招新题目"目录
DATA_PATH = Path(r"D:\新建文件夹\算法组招新题目\task3\train-00000-of-00001.parquet")   # 数据集未复制进作业目录，这里指向原位置
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 128
EPOCHS = 8
LR = 1e-3

# ============ 1. 读数据 ============
df = pd.read_parquet(DATA_PATH)
# image 列是 HuggingFace 风格字典 {'bytes': b'...PNG...', 'path': ...}，要自己解 PNG
images = np.stack([np.array(Image.open(io.BytesIO(item["bytes"])))
                   for item in df["image"]])  # -> (60000, 28, 28) uint8
labels = df["label"].to_numpy()
print(f"读入 {images.shape[0]} 张图片，shape={images.shape}，标签范围 {labels.min()}~{labels.max()}")

x_all = torch.tensor(images, dtype=torch.float32).reshape(len(images), 28 * 28)  # MLP 只吃一维向量
y_all = torch.tensor(labels, dtype=torch.long)
# 像素原来在 0~255，先缩到 0~1，再减 MNIST 全局均值除以标准差：
# 这样输入落在 0 附近、各维尺度一致，梯度不会被大数值放大，Adam 收敛更快。
x_all = (x_all / 255.0 - 0.1307) / 0.3081

# 原数据没有测试集，自己划分：随机打乱后前 50000 张训练、后 10000 张测试
perm = torch.randperm(len(x_all))
train_idx, test_idx = perm[:50000], perm[50000:]
x_train, y_train = x_all[train_idx], y_all[train_idx]
x_test, y_test = x_all[test_idx], y_all[test_idx]
print(f"训练集 {len(x_train)} 张，测试集 {len(x_test)} 张")

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
# shuffle=True 每个 epoch 打乱顺序，避免模型记住样本出现的先后顺序

# ============ 2. 搭模型 ============
# 784(输入) -> 256 -> 128 -> 10(输出)，两层隐藏层足够拟合手写数字，再深在 CPU 上没必要
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),  # 激活函数引入非线性，否则多层线性堆叠等价于一层线性
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10),  # 输出 10 个 logits，不加 softmax（CrossEntropyLoss 内部已经做了）
)
print(model)
print(f"可训练参数总量：{sum(p.numel() for p in model.parameters())}")

# 多分类用交叉熵：它把 softmax + 负对数似然合成一步，数值上比"先 softmax 再手写 log"更稳
loss_fn = nn.CrossEntropyLoss()
# Adam 对每个参数自适应调整步长，学习率不用像 SGD 那样仔细调；lr=1e-3 是它的常用默认值
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ============ 3. 训练 + 每个 epoch 评估 ============
train_loss_list, train_acc_list, test_acc_list = [], [], []
t_start = time.time()

for epoch in range(EPOCHS):
    model.train()  # 切换到训练模式（本模型没有 Dropout/BN，但这行是习惯，也提醒自己别漏）
    running_loss, correct, total = 0.0, 0, 0

    for x_batch, y_batch in train_loader:
        logits = model(x_batch)            # 前向：784 -> 256 -> 128 -> 10
        loss = loss_fn(logits, y_batch)

        optimizer.zero_grad()  # 梯度默认累加，不清零就会把上一步的梯度也算进来
        loss.backward()        # 反向传播：链式法则从 loss 一路求到每层权重
        optimizer.step()       # 用 Adam 规则更新权重和偏置

        running_loss += loss.item() * x_batch.size(0)  # 乘回 batch 大小，才能算整个 epoch 的均值
        correct += (logits.argmax(dim=1) == y_batch).sum().item()  # argmax 取 logits 最大那类当预测
        total += x_batch.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    model.eval()  # 评估模式，语义上关闭训练期才用的行为
    with torch.no_grad():  # 只做前向，不需要建计算图，省内存也更快
        test_logits = model(x_test)  # 10000 张一次性前向，CPU 也只要零点几秒
        test_acc = (test_logits.argmax(dim=1) == y_test).float().mean().item()

    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    print(f"epoch {epoch + 1}/{EPOCHS}  loss={train_loss:.4f}  "
          f"train_acc={train_acc:.4f}  test_acc={test_acc:.4f}  "
          f"用时={time.time() - t_start:.1f}s")

total_seconds = time.time() - t_start

# ============ 4. 打印报告要用的数字 ============
test_pred = test_logits.argmax(dim=1)
acc = (test_pred == y_test).float().mean().item()
per_class_acc = [(test_pred[y_test == c] == c).float().mean().item() for c in range(10)]
print(f"\n最终测试集准确率：{acc:.4f}（{int(acc * len(y_test))}/{len(y_test)}）")
print("各类别准确率：" + "  ".join(f"{c}:{v:.3f}" for c, v in enumerate(per_class_acc)))
print(f"总训练耗时：{total_seconds:.1f} 秒（{EPOCHS} 个 epoch，平均 {total_seconds / EPOCHS:.1f} 秒/epoch）")

# ============ 5. 画训练曲线 ============
epochs = range(1, EPOCHS + 1)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

axes[0].plot(epochs, train_loss_list, marker="o", color="tab:red")
axes[0].set_title("训练损失 Loss")
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("CrossEntropy loss")

axes[1].plot(epochs, train_acc_list, marker="o", color="tab:blue")
axes[1].set_title("训练准确率 Train Acc")
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("accuracy")

axes[2].plot(epochs, test_acc_list, marker="o", color="tab:green")
axes[2].set_title("测试准确率 Test Acc")
axes[2].set_xlabel("epoch")
axes[2].set_ylabel("accuracy")

for ax in axes:
    ax.set_xticks(list(epochs))
    ax.grid(True, alpha=0.3)
fig.suptitle(f"题三：MLP 在 MNIST 上的训练过程（最终测试准确率 {acc:.2%}）", fontsize=14)
fig.tight_layout()
fig.savefig(OUT_DIR / "q3_curves.png", dpi=200)
plt.close(fig)

# ============ 6. 画识别结果（12 张分对的 + 4 张分错的） ============
is_correct = test_pred == y_test
idx_right = torch.nonzero(is_correct).flatten()[:12]  # 测试集内部下标
idx_wrong = torch.nonzero(~is_correct).flatten()[:4]
show_idx = torch.cat([idx_right, idx_wrong])
show_images = torch.tensor(images)[test_idx][show_idx].numpy()  # 回到原始 0~255 像素才好显示

fig, axes = plt.subplots(4, 4, figsize=(9.5, 10.5))
show_idx_list = show_idx.tolist()
for k, ax in enumerate(axes.ravel()):
    i = show_idx_list[k]  # i 是测试集内部下标
    ax.imshow(show_images[k], cmap="gray")
    true_label, pred_label = int(y_test[i]), int(test_pred[i])
    ok = true_label == pred_label
    ax.set_title(f"真实 {true_label} / 预测 {pred_label}", fontsize=11,
                 color="green" if ok else "red")  # 绿色=分对，红色=分错，一眼看出错在哪
    ax.axis("off")
fig.suptitle("题三：测试集识别结果（绿色为正确，红色为错误）", fontsize=14)
fig.tight_layout()
fig.savefig(OUT_DIR / "q3_predictions.png", dpi=200)
plt.close(fig)

print(f"\n图片已保存到：{OUT_DIR}")
print("  - q3_curves.png")
print("  - q3_predictions.png")
