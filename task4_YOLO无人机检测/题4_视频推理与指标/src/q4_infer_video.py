# 题4：用训练好的 best.pt 对 example.mp4 逐帧推理，保存结果视频，并统计模型大小与推理 fps
import os
import time
import cv2
import torch
from ultralytics import YOLO

torch.set_num_threads(6)  # 机器上还有其他任务，限制 CPU 线程

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_IN = r"D:\新建文件夹\算法组招新题目\task4_yolo\example\example.mp4"
VIDEO_OUT = os.path.join(ROOT, "outputs", "video", "example_result.mp4")
BEST = os.path.join(ROOT, "outputs", "weights", "best.pt")
DRONE = r"D:\新建文件夹\算法组招新题目\task4_yolo\drone.pt"
IMGSZ = 320   # 和训练时的输入尺寸保持一致
os.makedirs(os.path.dirname(VIDEO_OUT), exist_ok=True)

model = YOLO(BEST)

cap = cv2.VideoCapture(VIDEO_IN)
src_fps = cap.get(cv2.CAP_PROP_FPS)
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))

times = []   # 每帧推理耗时(ms)
n = 0        # 帧序号
boxes_sum = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    t0 = time.time()
    # 单帧推理：model.predict 对一帧图返回一个 Results，r.plot() 得到画好框的图
    r = model.predict(frame, imgsz=IMGSZ, conf=0.25, verbose=False)[0]
    dt = (time.time() - t0) * 1000
    if n > 0:                  # 第 0 帧当预热，不计入耗时（避免首次调用的初始化开销）
        times.append(dt)
    boxes_sum += len(r.boxes)
    writer.write(r.plot())     # plot 返回 BGR 图，直接写进视频
    n += 1
cap.release()
writer.release()

avg_ms = sum(times) / len(times)
print("\n===== 视频推理结果 =====")
print("视频：%dx%d, 原始 %d 帧 @ %.1f fps，共检出目标 %d 个" % (w, h, n, src_fps, boxes_sum))
print("best.pt   模型大小 = %.2f MB" % (os.path.getsize(BEST) / 1024**2))
print("drone.pt  模型大小 = %.2f MB" % (os.path.getsize(DRONE) / 1024**2))
print("平均每帧耗时 = %.1f ms（imgsz=%d, CPU）" % (avg_ms, IMGSZ))
print("推理 fps     = %.1f" % (1000.0 / avg_ms))
print("结果视频已保存：", VIDEO_OUT)
