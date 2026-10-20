"""task2 题三：检测 example1.jpg 的轮廓，并计算轮廓的总面积和总周长。

跑法： python src/q3_contour.py
产出： outputs/images/q3_contour.png

思路（面试要能讲）：
    灰度 + Otsu 在这张图上会失败，因为白色机身和浅蓝天空的**灰度值很接近**。
    真正能把它们分开的是**颜色**：天空是饱和的蓝色。
    所以先取顶部若干行估计出"天空的颜色"，再算每个像素到它的距离，
    距离远的就是飞机。最后对距离图做一次 Otsu 自动定阈值。
"""

import cv2
import numpy as np

from common import ASSETS, IMG_DIR, imread, imwrite, row, title_bar

img = imread(ASSETS / "example1.jpg")
h, w = img.shape[:2]

# 1) 估计天空的颜色：取最上面 40 行的中位数 -----------------------------------
# 用中位数而不是平均值，是因为天空里还有白云，平均值会被少数很亮的云拉高。
sky_color = np.median(img[:40].reshape(-1, 3), axis=0)   # reshape 成 (像素数, 3)，再按列取中位数
print("估计的天空颜色 (B, G, R) :", sky_color)

# 2) 算每个像素到天空颜色的距离 -------------------------------------------------
# 先转 float32：uint8 做减法会下溢（比如 50-200 会绕回 106），结果完全是错的。
diff = img.astype(np.float32) - sky_color                # 广播：(h,w,3) 减去 (3,)
dist = np.linalg.norm(diff, axis=2)                      # 沿通道维求欧氏距离 → (h,w) 单通道
dist8 = np.clip(dist / max(dist.max(), 1e-6) * 255, 0, 255).astype(np.uint8)  # 归一化回 0~255
dist8 = cv2.GaussianBlur(dist8, (5, 5), 0)               # 轻微模糊，去掉噪点

# 3) Otsu 自动定阈值，再做形态学修整 -------------------------------------------
_, mask = cv2.threshold(dist8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)                  # 开运算：去掉小白点
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)   # 闭运算：补上小黑洞

# 4) 找轮廓 -------------------------------------------------------------------
# RETR_EXTERNAL：只要最外层轮廓；CHAIN_APPROX_NONE：轮廓上的点全都留着（算周长更准）
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# 过滤掉面积太小的噪点（比如小于 200 平方像素）
contours = [c for c in contours if cv2.contourArea(c) >= 200]

# 5) 统计总面积、总周长 -------------------------------------------------------
total_area = sum(cv2.contourArea(c) for c in contours)          # 轮廓围成的多边形面积
total_perimeter = sum(cv2.arcLength(c, True) for c in contours)  # True = 轮廓闭合
print("轮廓数量            :", len(contours))
print("总面积              : %.0f 平方像素" % total_area)
print("总周长              : %.2f 像素" % total_perimeter)
print("掩膜里的白点个数    :", int((mask > 0).sum()), "（和总面积互相印证，差几个百分点是正常的）")

# 6) 存图：原图 / 距离图 / 掩膜 / 画了轮廓的原图 -------------------------------
vis = img.copy()
cv2.drawContours(vis, contours, -1, (0, 0, 255), 3)      # -1 表示画所有轮廓

imwrite(IMG_DIR / "q3_contour.png", row([
    title_bar(img.copy(), "original"),
    title_bar(cv2.cvtColor(dist8, cv2.COLOR_GRAY2BGR), "distance to sky colour"),
    title_bar(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), "mask (Otsu)"),
    title_bar(vis, f"contours={len(contours)} area={total_area:.0f} perim={total_perimeter:.0f}"),
], height=300))
print("已保存:", IMG_DIR / "q3_contour.png")
