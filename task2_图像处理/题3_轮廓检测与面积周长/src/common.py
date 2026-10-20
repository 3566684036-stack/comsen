"""task2 公共小工具：路径、中文路径安全的读写、画框、拼图。

所有脚本都从这里 import，好处是"路径"和"中文路径的坑"只写一次。
"""

from pathlib import Path

import cv2
import numpy as np

# ---- 路径：src/ 的上一级是 task2_solution，再上一级是素材目录 ----
ROOT = Path(__file__).resolve().parent.parent        # task2_solution
ASSETS = Path(r"D:\新建文件夹\算法组招新题目\task2")   # 素材未复制进作业目录，这里指向原位置
IMG_DIR = ROOT / "outputs" / "images"
VID_DIR = ROOT / "outputs" / "videos"
for _d in (IMG_DIR, VID_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def imread(path, flags=cv2.IMREAD_COLOR):
    """读图片。

    为什么不用 cv2.imread？OpenCV 在 Windows 上按系统编码(GBK)打开文件，
    路径里有中文时会**静默返回 None**（不报错），后面 img.shape 就会崩。
    这里改成先用 numpy 把文件读成字节，再让 OpenCV 从内存解码，绕开路径问题。
    """
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)


def imwrite(path, img):
    """写图片，同样绕开中文路径（cv2.imwrite 在中文路径下会静默失败）。"""
    ext = Path(str(path)).suffix                       # 用扩展名决定格式（.png/.jpg）
    ok, buf = cv2.imencode(ext, img)
    buf.tofile(str(path))


def draw_box(img, box, color, text=""):
    """画一个框 + 左上角一行小字。box 的格式是 (x, y, w, h)，和 OpenCV 一致。"""
    x, y, w, h = [int(v) for v in box]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    if text:
        cv2.putText(img, text, (x + 2, max(14, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def title_bar(img, text):
    """在图片顶部加一条黑底，写上这行说明，方便拼图后一眼看出每块是什么。"""
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 255), 1, cv2.LINE_AA)
    return img


def row(images, height=260, gap=6):
    """把几张图缩放到同样高度后横向拼起来（用来做对比图）。"""
    out = []
    for img in images:
        if img.ndim == 2:                              # 灰度图转三通道，否则拼不起来
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        scale = height / img.shape[0]
        out.append(cv2.resize(img, (int(img.shape[1] * scale), height)))
    if gap:
        sep = np.zeros((height, gap, 3), np.uint8)
        out = [x for pair in zip(out, [sep] * len(out)) for x in pair][:-1]
    return np.hstack(out)


def pad_width(img, width, color=(20, 20, 20)):
    """把图右侧补黑边到指定宽度。

    用途：把两行拼图竖着接起来时，两行宽度常常不一样（因为缩放后的图宽度不同），
    直接 np.vstack 会报 "dimensions must match"，补边之后再接就行。
    """
    if img.shape[1] == width:
        return img
    canvas = np.full((img.shape[0], width, 3), color, np.uint8)
    canvas[:, :img.shape[1]] = img
    return canvas


def stats(tag, value):
    """统一格式打印，方便最后一眼抄进报告。"""
    print(f"{tag:<28}{value}")
