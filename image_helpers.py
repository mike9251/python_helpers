import torch
import cv2
import numpy as np
from pathlib import Path


def tensor2img(x: torch.Tensor):
    img = x.detach().cpu().numpy()
    if len(img.shape) > 3:
      img = img[0]
    img = img.transpose(1, 2, 0)
    img = np.clip(img * 255, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def save_img(img, dst_dir: Path, name: str):
    dst_dir.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(dst_dir / name), img)
