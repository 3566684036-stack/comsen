"""task2 题五：用 OpenCV 的 tracker 追踪 example3.mp4 / example4.mp4 里的小球。

跑法： python src/q5_track.py
产出： outputs/videos/example3_track.mp4、example4_track.mp4
       outputs/images/q5_curve.png

做法（面试要能讲清这一段）：
  1. 先写一个"逐帧检测器"，用颜色阈值直接给出球的绝对位置。两个素材用两套阈值：
     - example3：红色小球，用 HSV 的红色范围。
     - example4：白色乒乓球放在木桌上。这里关键：桌面的**亮度比球还高**，
       按"白/亮"根本分不开；但球受冷光照射 hue≈98（偏蓝），木桌 hue≈19（偏暖），
       所以改用**色调**来分。
  2. 用 cv2.legacy.TrackerCSRT 追踪，它的优点是精度高，缺点是会**漂移**。
  3. 核心改进：每帧都拿"检测框"和"tracker 的框"比 IoU，一旦漂移（IoU 太低）
     就立刻用检测框重新初始化 tracker。这样把"一漂到底"变成"漂了立刻拉回"。
  4. 视频里同时画两条框做对比：
     青色 = 纯 CSRT（从头到尾不重新初始化）；绿色 = 改进版。
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ASSETS, IMG_DIR, VID_DIR, title_bar

# 环境自检：TrackerCSRT 在 cv2.legacy 里，而 OpenCV 5.x 把 legacy 整个删掉了。
# 如果用了 opencv-python 5.x，报错会是一句很费解的 "module 'cv2' has no attribute 'legacy'"，
# 看不出是环境问题，所以这里提前拦住并说清楚怎么修。
if not hasattr(cv2, "legacy") or not hasattr(cv2.legacy, "TrackerCSRT_create"):
    raise SystemExit(
        "当前 OpenCV 是 " + cv2.__version__ + "，里面没有 cv2.legacy.TrackerCSRT。\n"
        "题五需要 opencv-contrib-python 4.x（5.x 已删除 legacy 模块）。修复命令：\n"
        "  pip uninstall -y opencv-python\n"
        "  pip install --force-reinstall --no-deps opencv-contrib-python==4.12.0.88\n"
        "（安装 ultralytics 时会顺带装上 opencv-python 覆盖掉 contrib，注意顺序）"
    )

MAX_FRAMES_TO_COMPARE = None      # 想快速试跑时改成 100


# ---------------------------------------------------------------------------
# 检测器：给一帧，返回球的 (圆心x, 圆心y, 半径)；找不到返回 None
# ---------------------------------------------------------------------------
def detect_red_ball(frame):
    """example3：红色小球，用 HSV 的红色区间。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 红色的 hue 在色环两端（0~10 和 160~179），所以要两段并起来
    mask = cv2.inRange(hsv, (0, 90, 60), (10, 255, 255)) | \
           cv2.inRange(hsv, (160, 90, 60), (179, 255, 255))
    return _biggest_circle(mask)


def detect_white_ball(frame):
    """example4：白色乒乓球。亮度分不开，用色调（hue≈98 偏冷，木桌 hue≈19 偏暖）。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (78, 12, 70), (135, 140, 255))
    return _biggest_circle(mask)


def _biggest_circle(mask):
    """把掩膜修整一下，取最大的那块，返回它最小外接圆的圆心和半径。"""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)                  # 去噪点
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)   # 补空洞
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(biggest)
    if area < 400:                              # 太小，当噪声扔掉
        return None
    # 圆度 = 4πA/P²，正圆等于 1，形状越不规则越小。
    # example4 里桌上的纸巾也是"冷色调的浅色块"，会被同一个 hue 阈值选中，
    # 但它是皱的、形状不规则，圆度远低于球，用这个条件就能把它排除掉。
    perim = cv2.arcLength(biggest, True)
    circularity = 4 * np.pi * area / (perim ** 2) if perim else 0.0
    if circularity < 0.6:
        return None
    (cx, cy), r = cv2.minEnclosingCircle(biggest)
    return float(cx), float(cy), float(r)


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def circle_to_box(cx, cy, r):
    """OpenCV 的 tracker 要的框是 (左上角x, 左上角y, 宽, 高)。"""
    return (int(cx - r), int(cy - r), int(2 * r), int(2 * r))


def iou(a, b):
    """两个框的交并比：重叠面积 / 合并面积，1 表示完全重合，0 表示不重叠。"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)                  # 交集左上角取"较大的那个"
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)   # 右下角取"较小的那个"
    inter = max(0, x2 - x1) * max(0, y2 - y1)          # 不相交时会算出负数，用 max(0,..) 兜住
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def box_center(box):
    return box[0] + box[2] / 2, box[1] + box[3] / 2


# ---------------------------------------------------------------------------
# 主流程：对一个视频跑一遍
# ---------------------------------------------------------------------------
def run(clip_name, detector, out_name, iou_reinit=0.3):
    cap = cv2.VideoCapture(str(ASSETS / clip_name))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("\n=== %s  %dx%d  %d 帧 ===" % (clip_name, w, h, total))

    writer = cv2.VideoWriter(str(VID_DIR / out_name),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    ok, first = cap.read()
    det = detector(first)
    if det is None:
        print("第一帧就检测不到球，检查阈值！")
        return
    det0 = det        # 记住第一帧的球半径，后面用来判断"这个检测结果合不合理"

    # 两个 tracker 用同一个初始框；plain 那个之后永不重新初始化，用来做对比
    plain = cv2.legacy.TrackerCSRT_create()
    plain.init(first, circle_to_box(*det))
    improved = cv2.legacy.TrackerCSRT_create()
    improved.init(first, circle_to_box(*det))

    err_plain, err_improved = [], []
    lost_plain = lost_improved = 0
    reinit_count = 0
    last_good = det        # 最近一次"可信"的检测位置，用来判断下一帧的检测是不是瞬移
    last_good_box = circle_to_box(*det)   # 最近一次可信的框，检测不可用时会拿它兜底
    frame = first
    for idx in range(total):
        if idx > 0:
            ok, frame = cap.read()
            if not ok:
                break

        det = detector(frame)                     # 每一帧都检测，颜色阈值很快（几毫秒）
        # 对检测结果做两道"合理性检查"，否则偶尔的误检会把 tracker 带跑偏：
        #   1) 半径不能突变（球的远近变化没那么快）
        #   2) 位置不能瞬移（球一帧之内不可能跳到很远的地方）
        if det is not None:
            if not (0.5 * det0[2] <= det[2] <= 1.8 * det0[2]):
                det = None
            elif last_good is not None and np.hypot(det[0] - last_good[0],
                                                    det[1] - last_good[1]) > 2.5 * det0[2]:
                det = None
        if det is not None:
            last_good = det                        # 记下最近一次可信的位置

        # ---- 纯 CSRT：只让它自己递推，漂了也不管 ----
        ok_p, box_p = plain.update(frame)
        if not ok_p:
            lost_plain += 1
        # ---- 改进版：检测到了就拿检测框校准，漂了就重新初始化 ----
        ok_i, box_i = improved.update(frame)
        if det is not None:
            det_box = circle_to_box(*det)
            # IoU 太低说明 tracker 已经偏了（或者直接丢了），立刻拉回来
            if not ok_i or iou(box_i, det_box) < iou_reinit:
                improved = cv2.legacy.TrackerCSRT_create()
                improved.init(frame, det_box)
                reinit_count += 1
            box_i, ok_i = det_box, True
            last_good_box = det_box            # 记住这个可信的框
        elif last_good_box is not None and (not ok_i or iou(box_i, last_good_box) < 0.1):
            # 这一帧没有可信检测，而且 tracker 明显跑偏了（比如飘到旁边的纸巾上）：
            # 与其让它继续乱漂，不如保持上一次可信的位置。
            box_i, ok_i = last_good_box, True
        if not ok_i:
            lost_improved += 1

        # ---- 记录误差：和检测器给的球心比（检测器当参考位置）----
        if det is not None:
            ref = (det[0], det[1])
            if ok_p:
                cx, cy = box_center(box_p)
                err_plain.append(np.hypot(cx - ref[0], cy - ref[1]))
            if ok_i:
                cx, cy = box_center(box_i)
                err_improved.append(np.hypot(cx - ref[0], cy - ref[1]))

        # ---- 画图 ----
        vis = frame.copy()
        if det is not None:
            cv2.circle(vis, (int(det[0]), int(det[1])), int(det[2]), (0, 255, 255), 2)
        if ok_p:
            cv2.rectangle(vis, (int(box_p[0]), int(box_p[1])),
                          (int(box_p[0] + box_p[2]), int(box_p[1] + box_p[3])), (255, 200, 0), 2)
        if ok_i:
            cv2.rectangle(vis, (int(box_i[0]), int(box_i[1])),
                          (int(box_i[0] + box_i[2]), int(box_i[1] + box_i[3])), (0, 255, 0), 2)
        title_bar(vis, "frame %d/%d   cyan=CSRT only   green=CSRT+re-detect   yellow=detector"
                  % (idx + 1, total))
        writer.write(vis)

        if MAX_FRAMES_TO_COMPARE and idx >= MAX_FRAMES_TO_COMPARE:
            break
        if idx % 100 == 0:
            print("  第 %d/%d 帧" % (idx, total))

    cap.release()
    writer.release()

    # ---- 统计 ----
    print("  纯 CSRT   : 丢失 %d 帧, 平均中心误差 %.1f 像素, 中位误差 %.1f 像素"
          % (lost_plain, np.mean(err_plain) if err_plain else -1,
             np.median(err_plain) if err_plain else -1))
    print("  改进版    : 丢失 %d 帧, 平均中心误差 %.1f 像素, 中位误差 %.1f 像素, 重新初始化 %d 次"
          % (lost_improved, np.mean(err_improved) if err_improved else -1,
             np.median(err_improved) if err_improved else -1, reinit_count))
    return err_plain, err_improved, lost_plain, lost_improved, reinit_count


e3 = run("example3.mp4", detect_red_ball, "example3_track.mp4")
e4 = run("example4.mp4", detect_white_ball, "example4_track.mp4")

# ---- 把两种方法的误差曲线画出来 ---------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
for ax, (name, res) in zip(axes, (("example3.mp4 (easy)", e3), ("example4.mp4 (hard)", e4))):
    if res is None:
        continue
    ax.plot(res[0], color="#1f77b4", lw=1, label="CSRT only")
    ax.plot(res[1], color="#2ca02c", lw=1, label="CSRT + re-detect")
    ax.set_title(name); ax.set_xlabel("frame"); ax.set_ylabel("centre error (px)")
    ax.legend(); ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(IMG_DIR / "q5_curve.png", dpi=120)
print("\n已保存: outputs/videos/example3_track.mp4, example4_track.mp4, outputs/images/q5_curve.png")
