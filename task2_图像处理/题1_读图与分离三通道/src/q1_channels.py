"""task2 题一：用 OpenCV 和 Pillow 打开图片，显示，并分离三个通道。

跑法： python src/q1_channels.py
产出： outputs/images/q1_channels.png
"""

import cv2
import numpy as np
from PIL import Image

from common import ASSETS, IMG_DIR, imread, imwrite, row, title_bar

# 想亲眼看窗口就改成 True（脚本会停下来等你按键）；默认 False 只存图，
# 这样批量运行时不会卡住。
SHOW = False

# 1) 两个库各读一次，对比读出来的东西有什么不同 -------------------------------
img_cv = imread(ASSETS / "example1.jpg")                 # cv2 读出来是 (高, 宽, 3)，顺序 BGR
img_pil = np.array(Image.open(ASSETS / "example1.jpg").convert("RGB"))  # PIL 的顺序是 RGB

print("cv2 读到的 shape :", img_cv.shape, " 通道顺序 BGR")
print("PIL 读到的 shape :", img_pil.shape, " 通道顺序 RGB")
print("两个数组完全相同吗      :", np.array_equal(img_cv, img_pil))
print("BGR 转 RGB 之后相同吗   :",
      np.array_equal(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), img_pil))
# 结论：图本身没错，只是通道顺序不同。直接把 cv2 的数组当 RGB 显示，红蓝会颠倒。

# 2) 显示（可选）--------------------------------------------------------------
if SHOW:
    cv2.imshow("example1", cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)                                       # 0 = 等你按任意键再关
    cv2.destroyAllWindows()

# 3) 分离三个通道 -------------------------------------------------------------
# cv2.split 把三通道图拆成三个二维数组，顺序是 B、G、R。
b, g, r = cv2.split(img_cv)
print("单个通道的 shape :", b.shape, "（少了一维，就是一张灰度图）")

# 4) 存对比图：左原图，右边三个通道 -------------------------------------------
tiles = [title_bar(img_cv.copy(), "original (BGR)")]
for name, ch in (("Blue", b), ("Green", g), ("Red", r)):
    # 单通道是二维的，拼图前要转成三通道；这里按灰度显示，方便看通道强弱
    grey = cv2.cvtColor(ch, cv2.COLOR_GRAY2BGR)
    tiles.append(title_bar(grey, f"{name} channel"))

imwrite(IMG_DIR / "q1_channels.png", row(tiles, height=300))
print("已保存:", IMG_DIR / "q1_channels.png")

# 顺带一个观察：B 通道里天空很亮、飞机很暗 → "天空的蓝"主要来自蓝色通道。
print("B 通道平均亮度 : %.1f" % b.mean())
print("R 通道平均亮度 : %.1f" % r.mean())
