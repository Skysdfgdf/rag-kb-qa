"""程序图标：打包成 .ico 用，同时作为托盘图标。

用代码画出来，避免额外维护图片文件。
"""
from pathlib import Path

from PIL import Image, ImageDraw

BLUE = (37, 99, 235, 255)
WHITE = (255, 255, 255, 255)
GREY = (148, 163, 184, 255)


def make_image(size: int = 256) -> Image.Image:
    """画一个「蓝色圆角方块 + 白色文档」的图标。"""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    s = size / 256.0

    def box(x1, y1, x2, y2, r, fill):
        d.rounded_rectangle([x1 * s, y1 * s, x2 * s, y2 * s], radius=r * s, fill=fill)

    box(6, 6, 250, 250, 56, BLUE)          # 底板
    box(70, 42, 186, 214, 16, WHITE)       # 文档
    box(94, 92, 162, 106, 7, BLUE)         # 标题行
    box(94, 126, 162, 140, 7, GREY)        # 正文行
    box(94, 160, 138, 174, 7, GREY)
    return img


def save_ico(path: Path) -> Path:
    """生成多尺寸 ico（Windows 任务栏/exe 图标用）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    make_image(256).save(
        path,
        format="ICO",
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    return path
