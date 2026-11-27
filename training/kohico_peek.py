#@title ㉙-1 내려받기 주소가 돌려주는 웹페이지 살펴보기 (키·토큰은 가려서 출력)
import re, glob, os
from google.colab import drive; drive.mount('/content/drive')
RAW = '/content/drive/MyDrive/방각본_판독기/4_말뭉치_비공개/kohico_raw'
f = glob.glob(RAW + '/2023/*.zip')[0]; t = open(f, 'rb').read().decode('utf-8', 'replace')
mask = lambda s: re.sub(r'[0-9A-Za-z_\-]{24,}', '(가림)', s)
print('길이', len(t)); print(mask(re.sub(r'\s+', ' ', t))[:2500])
for p in glob.glob(RAW + '/*/*.zip'):
    if os.path.getsize(p) < 100000: os.remove(p)   # 웹페이지를 잘못 저장한 것이니 지운다
