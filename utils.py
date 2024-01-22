from moviepy.video.io.bindings import mplfig_to_npimage
from pathlib import Path
from PIL import Image


def to_pil(img):
    if not isinstance(img, Image.Image):
        return Image.fromarray(img)
    return img


def save_image(img, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    to_pil(img).save(str(path))


def show_image(img):
    to_pil(img).show()


def fig_to_array(fig):
    arr = mplfig_to_npimage(fig)
    return arr
