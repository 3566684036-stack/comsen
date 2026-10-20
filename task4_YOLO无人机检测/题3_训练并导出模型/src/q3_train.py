# 题3：用题目给的无人机预训练权重 drone.pt 在无人机数据集上做迁移学习训练，并保存模型与训练结果图
import os
import shutil
import time
import zipfile
import torch
from ultralytics import YOLO

torch.set_num_threads(6)  # 机器上还有其他任务，限制 CPU 线程

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # task4_solution/
T4 = r"D:\新建文件夹\算法组招新题目\task4_yolo"                        # 题目素材目录
ZIP = os.path.join(T4, "dataset", "drone_bird.zip")
DATA_DIR = os.path.join(T4, "dataset", "drone_bird")
YAML = os.path.join(T4, "dataset", "drone_bird_abs.yaml")             # 自己写的绝对路径 yaml
RUNS = os.path.join(T4, "dataset", "runs")
OUT_IMG = os.path.join(ROOT, "outputs", "images")
OUT_W = os.path.join(ROOT, "outputs", "weights")
for d in (OUT_IMG, OUT_W):
    os.makedirs(d, exist_ok=True)

# 1) 解压数据集（只解压一次，225MB，约 10s）
if not os.path.isdir(os.path.join(DATA_DIR, "train", "images")):
    print("解压数据集到", DATA_DIR)
    zipfile.ZipFile(ZIP).extractall(DATA_DIR)

# 2) 数据集自带的 data.yaml 里写的是 ../train/images 这种相对路径，跨目录调用容易找不到图，
#    所以这里按 绝对路径 重新生成一份，训练时只认这份。
with open(YAML, "w", encoding="utf-8") as f:
    f.write("train: D:/新建文件夹/算法组招新题目/task4_yolo/dataset/drone_bird/train/images\n"
            "val: D:/新建文件夹/算法组招新题目/task4_yolo/dataset/drone_bird/valid/images\n"
            "test: D:/新建文件夹/算法组招新题目/task4_yolo/dataset/drone_bird/test/images\n"
            "\nnc: 2\nnames: ['bird', 'drone']\n")

# 3) 训练耗时先估过：imgsz=320、batch=8 在 CPU(6线程) 上约 0.06 s/张，
#    训练集 3893 张 -> 每轮训练约 4 分钟，每轮验证(649张)约 15 s，6 轮合计约 25 分钟，控制在 40 分钟内。
t0 = time.time()
model = YOLO(os.path.join(T4, "drone.pt"))   # 题目给的无人机预训练权重（yolo11n 结构，2 类 Bird/Drone）
model.train(
    data=YAML,
    epochs=6,          # 轮数
    imgsz=320,         # 输入分辨率，CPU 上 320 比 640 快 4 倍左右
    batch=8,
    workers=2,
    device="cpu",
    project=RUNS,      # 训练中间产物（曲线、权重）都放这里，最后只把要交的图复制走
    name="train",
    exist_ok=True,
    seed=0,            # 固定随机种子，结果可复现
)
train_sec = time.time() - t0
print("\n训练总耗时：%.1f 分钟" % (train_sec / 60))

# 4) best.pt 复制到 outputs/weights/
run_dir = str(model.trainer.save_dir)
shutil.copy(os.path.join(run_dir, "weights", "best.pt"), os.path.join(OUT_W, "best.pt"))

# 5) 训练曲线（loss 与 mAP）、混淆矩阵复制到 outputs/images/ 并改成看得懂的名字
for src, dst in [("results.png", "train_results.png"),
                 ("confusion_matrix.png", "confusion_matrix.png"),
                 ("confusion_matrix_normalized.png", "confusion_matrix_normalized.png")]:
    shutil.copy(os.path.join(run_dir, src), os.path.join(OUT_IMG, dst))

# 6) 用最好的权重在 测试集 上验证，打印 mAP50 / mAP50-95 / precision / recall
best = YOLO(os.path.join(OUT_W, "best.pt"))
m = best.val(data=YAML, split="test", imgsz=320, batch=8, workers=2, device="cpu",
             plots=False, project=RUNS, name="val_test", exist_ok=True)
print("\n===== 测试集指标（best.pt）=====")
print("mAP50    = %.4f" % m.box.map50)
print("mAP50-95 = %.4f" % m.box.map)
print("precision= %.4f" % m.box.mp)
print("recall   = %.4f" % m.box.mr)
print("权重与图片已保存到：", OUT_W, "|", OUT_IMG)
