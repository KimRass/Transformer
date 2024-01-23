import cv2
import cmapy
import matplotlib.pyplot as plt
from pathlib import Path

from model import PositionalEncoding
from utils import fig_to_array, save_image


def vis_pos_enc_mat(pos_enc, cmap="Blues"):
    vis = pos_enc.numpy()
    vis -= vis.min()
    vis /= vis.max()
    vis *= 255
    vis = vis.astype("uint8")
    vis = cv2.applyColorMap(vis, cmapy.cmap(cmap))

    fig, axes = plt.subplots(figsize=(10, 6))
    # axes.imshow(vis.transpose(1, 0, 2)[::-1, ...])
    axes.imshow(vis.transpose(1, 0, 2)[::-1, :, ::-1])
    fig.tight_layout()
    arr = fig_to_array(fig)
    return arr


if __name__ == "__main__":
    pos_enc = PositionalEncoding(dim=512, max_len=1024)
    vis = vis_pos_enc_mat(pos_enc.pe_mat)
    save_image(img=vis, path=Path(__file__).parent.resolve()/"pos_encoding.jpg")
