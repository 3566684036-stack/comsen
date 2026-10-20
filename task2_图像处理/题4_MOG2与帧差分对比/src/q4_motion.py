"""task2 题四：example2.mp4 运动目标检测 —— MOG2 背景减除 vs 不做背景减除。

跑法： python src/q4_motion.py
产出： outputs/videos/example2_mog2.mp4、example2_framediff.mp4
       outputs/images/q4_curve.png

两种方法的区别（面试要能讲）：
  A. MOG2 背景建模：给每个像素学一个"平时是什么样"的模型，长期没变的就是背景。
     好处：得到的是**实心**的车辆区域；车慢下来或短暂停住也还能检测到。
  B. 不做背景减除，直接相邻两帧相减：只对**正在动的边缘**有响应。
     好处：不需要预热，第一帧就能用。
     坏处：结果是"空心"的轮廓；车一停就检测不到了。
两种方法用完全相同的后处理，保证唯一变量是"前景怎么来的"。
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")            # 不弹窗，只存图
import matplotlib.pyplot as plt

from common import ASSETS, IMG_DIR, VID_DIR, draw_box, title_bar

MIN_AREA = 450                   # 小于这个面积的轮廓当噪点丢掉


def postprocess(mask):
    """两种方法共用的后处理：开运算去噪 → 闭运算连通 → 找外接框。"""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= MIN_AREA]
    return mask, boxes


def annotate(frame, boxes, text):
    vis = frame.copy()
    for (x, y, w, h) in boxes:
        draw_box(vis, (x, y, w, h), (0, 255, 0))
    title_bar(vis, text)
    return vis


cap = cv2.VideoCapture(str(ASSETS / "example2.mp4"))
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("视频: %dx%d, %.1f fps, %d 帧" % (w, h, fps, n_frames))

# 两个写视频的句柄。尺寸必须和 write 进去的帧完全一致，否则 OpenCV 会**静默失败**。
out_a = cv2.VideoWriter(str(VID_DIR / "example2_mog2.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
out_b = cv2.VideoWriter(str(VID_DIR / "example2_framediff.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

counts_a, counts_b = [], []      # 每帧检出多少个目标，最后画曲线用
prev_gray = None
for i in range(n_frames):
    ok, frame = cap.read()
    if not ok:
        break

    # ---- A: MOG2 ----
    raw = mog2.apply(frame)                       # 每一帧都要喂给它，模型边用边学
    # 输出里 0=背景、127=阴影、255=前景。阴影是"跟着车一起动的暗区"，
    # 不剔掉的话车和影子会连成一大块，框会变得很大。
    fg_a = cv2.threshold(raw, 200, 255, cv2.THRESH_BINARY)[1]
    mask_a, boxes_a = postprocess(fg_a)

    # ---- B: 相邻帧差分（不做背景减除）----
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)      # 先模糊，抑制传感器噪点
    if prev_gray is None:
        fg_b = np.zeros_like(gray)
    else:
        diff = cv2.absdiff(gray, prev_gray)       # 取绝对值：变亮变暗都算变化
        _, fg_b = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    mask_b, boxes_b = postprocess(fg_b)
    prev_gray = gray

    counts_a.append(len(boxes_a))
    counts_b.append(len(boxes_b))

    out_a.write(annotate(frame, boxes_a, "A: MOG2 background subtraction   objects=%d" % len(boxes_a)))
    out_b.write(annotate(frame, boxes_b, "B: frame difference (no background model)   objects=%d" % len(boxes_b)))

    if i % 100 == 0:
        print("  第 %d 帧: MOG2=%d, 帧差分=%d" % (i, len(boxes_a), len(boxes_b)))

cap.release()
out_a.release()
out_b.release()

# ---- 统计 + 曲线 ------------------------------------------------------------
a = np.array(counts_a)
b = np.array(counts_b)
warm = 30                                        # MOG2 前几十帧还在学背景，统计时跳过
print("\n统计（跳过前 %d 帧预热）:" % warm)
print("  MOG2   平均目标数 %.2f，完全漏检 %d 帧" % (a[warm:].mean(), (a[warm:] == 0).sum()))
print("  帧差分 平均目标数 %.2f，完全漏检 %d 帧" % (b[warm:].mean(), (b[warm:] == 0).sum()))
print("  → 平均目标数差不多，但漏检帧数差很多：车慢下来时帧差分就什么都检测不到了。")

fig, ax = plt.subplots(figsize=(11, 3.6))
ax.plot(a, color="#1f77b4", lw=1, label="A: MOG2")
ax.plot(b, color="#d62728", lw=1, label="B: frame difference")
ax.set_xlabel("frame index"); ax.set_ylabel("objects detected")
ax.set_title("example2.mp4 -- objects detected per frame")
ax.legend(); ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(IMG_DIR / "q4_curve.png", dpi=120)
print("已保存: outputs/videos/example2_mog2.mp4, example2_framediff.mp4, outputs/images/q4_curve.png")
