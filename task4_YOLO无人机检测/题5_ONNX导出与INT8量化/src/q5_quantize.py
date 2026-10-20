# 题5（提高）：导出 ONNX、做 INT8 动态量化，并对比 模型大小 / 推理速度 / 精度
import os
os.environ["YOLO_AUTOINSTALL"] = "false"   # 禁止 ultralytics 自动 pip 安装依赖，避免动到环境
import time
import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from onnxruntime.quantization import quantize_dynamic, QuantType
from ultralytics import YOLO

torch.set_num_threads(6)  # 机器上还有其他任务，限制 CPU 线程

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEST = os.path.join(ROOT, "outputs", "weights", "best.pt")
ONNX = os.path.join(ROOT, "outputs", "weights", "best.onnx")
INT8 = os.path.join(ROOT, "outputs", "weights", "best_int8.onnx")
YAML = r"D:\新建文件夹\算法组招新题目\task4_yolo\dataset\drone_bird_abs.yaml"
FRAMES = r"D:\新建文件夹\算法组招新题目\task4_yolo\example\example.mp4"
RUNS = r"D:\新建文件夹\算法组招新题目\task4_yolo\dataset\runs"
IMG = os.path.join(ROOT, "outputs", "images", "quant_compare.png")
IMGSZ = 320
mb = lambda p: os.path.getsize(p) / 1024**2
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]   # 图上要显示中文
plt.rcParams["axes.unicode_minus"] = False

# 1) 导出 ONNX（fp32）。ultralytics 会把 best.onnx 生成在 best.pt 同目录
onnx_path = YOLO(BEST).export(format="onnx", imgsz=IMGSZ, simplify=True, opset=12)
print("ONNX 已导出：", onnx_path)

# 2) INT8 动态量化：权重存成 uint8，激活值在推理时按每批数据动态量化。
#    这里传 onnx.ModelProto 而不是文件路径：量化库内部会用 C++ 接口在中文路径下建临时文件，
#    直接传路径会因 "新建文件夹" 这类非 ASCII 路径失败。
quantize_dynamic(onnx.load(onnx_path), INT8, weight_type=QuantType.QUInt8)
print("INT8 已生成：", INT8)

# 3) 速度：三个后端喂同一批输入（视频取 20 帧，缩放到 320x320），都只测纯前向，不含前后处理
cap = cv2.VideoCapture(FRAMES)
frames, i = [], 0
while len(frames) < 20:
    ok, f = cap.read()
    if not ok:
        break
    if i % 15 == 0:
        frames.append(cv2.resize(f, (IMGSZ, IMGSZ)))
    i += 1
cap.release()
x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().div(255).contiguous()  # (N,3,H,W)

def bench(fn, n=30, rounds=3):
    # 同机还有别的任务在跑，单次计时抖动较大，这里每轮跑 n 次、重复 rounds 轮后取中位数
    for _ in range(2):      # 预热 2 次
        fn()
    costs = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        costs.append((time.perf_counter() - t0) / n * 1000)
    costs.sort()
    return costs[len(costs) // 2]

torch_model = YOLO(BEST).model.fuse()        # fuse() 把 BN 折进 Conv，和 ultralytics 推理时的实际状态一致
def run_pt():                                # 单帧前向，和 onnx 侧 batch=1 对齐；关掉梯度记录
    with torch.inference_mode():
        torch_model(x[:1])

so = ort.SessionOptions()
so.intra_op_num_threads = 6                  # 和 pytorch 侧一样给 6 线程，比较才公平
so.inter_op_num_threads = 1
def ort_runner(path):
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return lambda: sess.run(None, {name: x[:1].numpy()})

ms = {"best.pt": bench(run_pt), "best.onnx": bench(ort_runner(ONNX)), "best_int8.onnx": bench(ort_runner(INT8))}

# 4) 精度：三份模型都在同一个测试集(196 张)上跑 ultralytics 的 val，得到 mAP50 / mAP50-95
maps = {}
for p in (BEST, ONNX, INT8):
    r = YOLO(p).val(data=YAML, split="test", imgsz=IMGSZ, batch=8, workers=2, device="cpu",
                    plots=False, project=RUNS, name="quant_val", exist_ok=True)
    maps[os.path.basename(p)] = (r.box.map50, r.box.map)

# 5) 打印对比表
print("\n%-15s %10s %12s %8s %10s %10s" % ("模型", "大小(MB)", "单帧(ms)", "fps", "mAP50", "mAP50-95"))
for name in ms:
    print("%-15s %10.2f %12.1f %8.1f %10.4f %10.4f" % (
        name, mb(os.path.join(ROOT, "outputs", "weights", name)), ms[name], 1000 / ms[name], *maps[name]))

# 6) 柱状图：大小与单帧耗时对比
names = list(ms)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
axes[0].bar(names, [mb(os.path.join(ROOT, "outputs", "weights", n)) for n in names], color=["#4C72B0", "#DD8452", "#55A868"])
axes[0].set_title("模型大小 (MB)")
axes[1].bar(names, [ms[n] for n in names], color=["#4C72B0", "#DD8452", "#55A868"])
axes[1].set_title("单帧前向耗时 (ms, CPU 6 线程)")
for ax, vals in zip(axes, [[mb(os.path.join(ROOT, "outputs", "weights", n)) for n in names], [ms[n] for n in names]]):
    for j, v in enumerate(vals):
        ax.text(j, v, "%.2f" % v, ha="center", va="bottom", fontsize=9)
for ax in axes:
    ax.set_xticks(range(3))
    ax.set_xticklabels(names, fontsize=8)
    ax.grid(axis="y", ls="--", alpha=0.4)
plt.tight_layout()
plt.savefig(IMG, dpi=150)
print("对比图已保存：", IMG)
