# 음각(검은 바탕에 흰 글씨) 칸 학습용 덧붙임. 9차부터 prep2(aug=True) 안에서 부른다.
# 방각본 첫머리의 '화셜', '슉종'처럼 칸을 검게 파고 글자를 희게 남긴 곳을 흉내 낸다.
# 정답 글자는 그대로 두고 그림만 바꾸므로, 실제 음각 칸도 보통 글자와 같은 글자로 읽게 배운다.
import random
import numpy as np
from PIL import Image, ImageFilter

def invert_span(im, p=0.12):
    """im: 가로로 돌린 행 그림(L, 높이 = 행 너비, 왼쪽이 행의 위쪽). 확률 p로 글자 1~3자 너비를 음각처럼 바꾼다."""
    if random.random() >= p: return im
    a = np.asarray(im).astype(np.int16); h, w = a.shape
    n = random.choice([1, 2, 2, 3]); L = int(h * n * random.uniform(0.85, 1.1))
    if L >= w - 4: return im
    x0 = 0 if random.random() < 0.6 else random.randint(0, w - L - 1)   # 행 첫머리에 많이 나온다
    y0, y1 = int(h * random.uniform(0.0, 0.08)), int(h * random.uniform(0.92, 1.0))
    seg = a[y0:y1, x0:x0 + L]
    bg = random.randint(10, 70); fg = random.randint(185, 250)
    ink = seg < (seg.min() + seg.max()) / 2                      # 원래 먹 자리 → 흰 글씨
    out = np.where(ink, fg, bg).astype(np.int16)
    out += np.random.randint(-12, 13, out.shape)                  # 먹 얼룩
    a[y0:y1, x0:x0 + L] = np.clip(out, 0, 255)
    im = Image.fromarray(a.astype(np.uint8))
    if random.random() < 0.5: im = im.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 0.9)))
    return im
