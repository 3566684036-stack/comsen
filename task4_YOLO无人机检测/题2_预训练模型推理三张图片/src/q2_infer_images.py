# 题2：用 YOLO 官方预训练权重对 example 下的三张图片推理并保存结果
import os
import urllib.request
import torch
from ultralytics import YOLO

torch.set_num_threads(6)  # 机器上还有其他任务，限制 CPU 线程

# 路径：本脚本在 src/ 下，项目根目录是上一级
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMP = r"D:\新建文件夹\算法组招新题目\task4_yolo\example"
OUT = os.path.join(ROOT, "outputs", "images")
os.makedirs(OUT, exist_ok=True)

# 官方 COCO 预训练权重 yolo11n.pt（80 类，约 5.4MB）。
# 本机连不上 github.com，ultralytics 的自动下载会超时，所以缺文件时从 hf-mirror 下同一份官方权重。
WEIGHTS = os.path.join(ROOT, "yolo11n.pt")
if not os.path.exists(WEIGHTS):
    url = "https://hf-mirror.com/Ultralytics/YOLO11/resolve/main/yolo11n.pt"
    print("本地没有 yolo11n.pt，正在下载：", url)
    urllib.request.urlretrieve(url, WEIGHTS)
model = YOLO(WEIGHTS)
print("模型类别数：", len(model.names), " 权重大小：%.1f MB" % (os.path.getsize(WEIGHTS) / 1024**2))

for name in ["coco-bear.jpg", "coco-bus.jpg", "coco-person.jpg"]:
    # predict() 关键参数：
    #   conf    —— 置信度阈值，低于它的框直接丢弃，默认 0.25；误检多可以调高（如 0.5）
    #   save    —— 是否让 ultralytics 把画好框的图自动存到 runs/detect/predict*/，这里用 False
    #   verbose —— 是否打印 ultralytics 自己的逐框日志，关掉后只保留下面我们自己的输出
    results = model.predict(os.path.join(EXAMP, name), conf=0.25, save=False, verbose=False)

    r = results[0]
    # results[0].save() 把画好框的图存到指定路径，省得去 runs/ 目录里翻
    r.save(os.path.join(OUT, "infer_" + name))

    print("\n[%s] 检出 %d 个目标：" % (name, len(r.boxes)))
    if len(r.boxes) == 0:
        print("   （没有目标超过 conf=0.25）")
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print("   %-12s conf=%.3f  box=[%.0f, %.0f, %.0f, %.0f]" % (
            model.names[int(box.cls)], float(box.conf), x1, y1, x2, y2))

print("\n推理结果图片已保存到：", OUT)
