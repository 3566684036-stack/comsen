"""task2 题二：对图片做缩放、旋转、平移、翻转。

跑法： python src/q2_geometry.py
产出： outputs/images/q2_geometry.png（六种操作拼成一张对比图）
"""

import cv2
import numpy as np

from common import ASSETS, IMG_DIR, imread, imwrite, row, title_bar, pad_width


def scale(img, factor):
    # 缩小用 INTER_AREA（对区域取平均，不会产生摩尔纹）；放大用 INTER_CUBIC（更平滑）
    method = cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=method)


def rotate(img, angle):
    """绕图片中心旋转 angle 度（正数是逆时针）。保持原画布大小，四角会被裁掉。"""
    h, w = img.shape[:2]
    center = (w / 2, h / 2)                              # 注意：中心是 (宽/2, 高/2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)      # 2x3 的仿射矩阵
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(30, 30, 30))


def translate(img, dx, dy):
    """向右移 dx、向下移 dy 个像素。"""
    h, w = img.shape[:2]
    # 仿射矩阵的第三列就是平移量：x' = x + dx, y' = y + dy
    M = np.float32([[1, 0, dx], [0, 1, dy]])             # 必须是 float32，否则 OpenCV 报错
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(30, 30, 30))


def flip(img, how):
    """how='h' 左右翻转，'v' 上下翻转，'both' 两个方向都翻。"""
    # cv2.flip 的第二个参数：1 水平、0 垂直、-1 同时
    return cv2.flip(img, {"h": 1, "v": 0, "both": -1}[how])


# ---------------------------------------------------------------------------
img = imread(ASSETS / "example1.jpg")
h, w = img.shape[:2]
print("原图尺寸:", w, "x", h)

results = [
    (title_bar(img.copy(), "original"), "q2_original.png"),
    (title_bar(scale(img, 0.5), "scale 0.5x (INTER_AREA)"), "q2_scale_05.png"),
    (title_bar(scale(img, 1.5), "scale 1.5x (INTER_CUBIC)"), "q2_scale_15.png"),
    (title_bar(rotate(img, 45), "rotate 45 deg"), "q2_rotate_45.png"),
    (title_bar(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), "rotate 90 deg"), "q2_rotate_90.png"),
    (title_bar(translate(img, 120, 60), "translate (120, 60)"), "q2_translate.png"),
    (title_bar(flip(img, "h"), "flip horizontal"), "q2_flip_h.png"),
    (title_bar(flip(img, "v"), "flip vertical"), "q2_flip_v.png"),
]

# 先分两行拼，再把两行竖着接起来，得到一张 4x2 的对比图
top = row([r[0] for r in results[:4]], height=250)
bottom = row([r[0] for r in results[4:]], height=250)
# 缩放过的图宽度不一样，两行宽度会对不齐，先补黑边到同一宽度再竖着接
width = max(top.shape[1], bottom.shape[1])
imwrite(IMG_DIR / "q2_geometry.png", np.vstack([pad_width(top, width), pad_width(bottom, width)]))
print("已保存:", IMG_DIR / "q2_geometry.png")

# 关键结论（面试时会被问的点）：
print("旋转 90 度用 cv2.rotate 是无损的（只是搬像素），不用插值；")
print("任意角度旋转要靠 warpAffine 插值，而且要保持原画布就会裁掉四个角。")
