
import os
import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people

lfw = fetch_lfw_people(color=True, resize=1.0, min_faces_per_person=20)
os.makedirs("data/eval", exist_ok=True)

seen, n = set(), 0
for img, label in zip(lfw.images, lfw.target):
    if label in seen:
        continue
    seen.add(label)
    arr = img * 255 if img.max() <= 1.5 else img
    arr = np.clip(arr, 0, 255).astype("uint8")
    cv2.imwrite(f"data/eval/lfw_{n:02d}.jpg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    n += 1
    if n == 30:
        break
print("saved", n, "images") 
