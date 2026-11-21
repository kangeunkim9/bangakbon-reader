# 방각본 판독기 대량 판독 (Colab에서 돌림)
#
# 쓰는 법
#   1) 드라이브 '방각본_판독기/판독할_PDF' 폴더에 PDF를 넣는다.
#      위키미디어 공용 파일은 같은 폴더의 commons.txt에 'File:…pdf' 이름을 한 줄에 하나씩 적어도 된다.
#   2) Colab에서 이 파일을 실행한다(런타임 유형은 GPU 권장, CPU도 된다).
#   3) '방각본_판독기/판독결과'에 책마다 .json(판독기에서 불러오기용)과 .txt(판독문)가 생긴다.
#   4) 판독기에 같은 PDF와 .json을 함께 끌어다 놓고 '판독 시작'을 누르면, 다시 읽지 않고 바로 검수 화면이 뜬다.
#
# 행 나누기는 판독기와 같은 seg.py를 GitHub에서 받아 쓴다. 판독은 학습 때와 같은 전처리(prep)를 쓴다.
import os, io, re, sys, json, glob, time, unicodedata, subprocess
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'regex', 'pymupdf'], check=False)
import numpy as np, cv2, fitz, regex, requests
from PIL import Image, ImageOps
from scipy.signal import find_peaks

ROOT = os.environ.get('BANGAKBON_ROOT', '/content/drive/MyDrive/방각본_판독기')
IN, OUT = ROOT + '/판독할_PDF', ROOT + '/판독결과'
REPO = 'https://raw.githubusercontent.com/kangeunkim9/bangakbon-reader/main/'
UA = {'User-Agent': 'KoreanWoodblockOCR/0.1 (https://kangeunkim9.github.io; academic research) python-requests'}
H, MAXW = 64, 1600
nfc = lambda s: unicodedata.normalize('NFC', s)
G = lambda s: regex.findall(r'\X', nfc(s))

# ---------- 행 나누기: 판독기(engine.js)와 같은 규칙 ----------
def binar(im):
    g = np.asarray(im.convert('L'))
    return (cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 12) > 0).astype(np.uint8)
_seg_src = os.environ.get('BANGAKBON_SEG') and open(os.environ['BANGAKBON_SEG'], encoding='utf-8').read() \
           or requests.get(REPO + 'web/src/seg.py', timeout=60).text
exec(_seg_src)          # vbounds(), segment()

def split_spread(im):
    """두 면이 나란히 찍힌 사진이면 가운데 골에서 나눠 [(오른쪽), (왼쪽)]. 아니면 [(전체)]. (KP.splitSpread와 같은 규칙)"""
    W, Hh = im.size
    if W < Hh * 1.15: return [(im, '', None)]
    b = binar(im); vl = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((Hh // 4, 1), np.uint8))
    col = (b & (1 - vl))[int(Hh * 0.1):int(Hh * 0.9)].sum(0).astype(float)
    k = max(5, round(W / 60) | 1); sm = np.convolve(col, np.ones(k) / k, 'same')
    lo, hi = int(W * 0.38), int(np.ceil(W * 0.62)); x = lo + int(np.argmin(sm[lo:hi]))
    med = float(np.median(sm[int(W * 0.1):int(np.ceil(W * 0.9))]))
    if sm[x] > med * 0.6: return [(im, '', None)]
    return [(im.crop((x, 0, W, Hh)), '오른쪽', [x, W]), (im.crop((0, 0, x, Hh)), '왼쪽', [0, x])]

def norm_width(im):
    W, Hh = im.size
    if W > 1700 or W < 1000: return im.resize((1500, max(1, round(Hh * 1500 / W))), Image.BILINEAR)
    return im

def page_gray(doc, i):
    """판독기와 같게: 그림이 하나면 그 그림, 아니면 200dpi로 그린 면."""
    pg = doc[i]; imgs = pg.get_images(full=True)
    if len(imgs) == 1:
        d = doc.extract_image(imgs[0][0])
        im = Image.open(io.BytesIO(d['image'])).convert('L')
        if im.width > 300: return im
    pix = pg.get_pixmap(dpi=200, colorspace=fitz.csGRAY)
    return Image.frombytes('L', (pix.width, pix.height), pix.samples)

# ---------- 판독 모델 ----------
def prep_arr(im):
    im = ImageOps.autocontrast(im.convert('L').rotate(90, expand=True), cutoff=1)
    w, h = im.size; nw = min(MAXW, max(32, int(w * H / h)))
    return 1.0 - np.asarray(im.resize((nw, H), Image.BILINEAR), dtype=np.float32) / 255.0

class Reader:
    def __init__(self, path):
        self.label = os.path.basename(path)
        if path.endswith('.pt'):
            import torch, torch.nn as nn
            ck = torch.load(path, map_location='cpu'); self.chars = ck['chars']
            class CRNN(nn.Module):
                def __init__(s, n):
                    super().__init__()
                    def blk(i, o, pool): return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True)] + ([nn.MaxPool2d(pool)] if pool else [])
                    s.cnn = nn.Sequential(*blk(1, 64, (2, 2)), *blk(64, 128, (2, 2)), *blk(128, 256, None),
                                          *blk(256, 256, (2, 1)), *blk(256, 384, None), *blk(384, 384, (2, 1)), nn.Dropout2d(0.1))
                    s.rnn = nn.LSTM(384 * (H // 16), 256, num_layers=2, bidirectional=True, batch_first=True, dropout=0.25)
                    s.fc = nn.Linear(512, n)
                def forward(s, x):
                    f = s.cnn(x); b, c, h, w = f.shape
                    y, _ = s.rnn(f.permute(0, 3, 1, 2).reshape(b, w, c * h)); return s.fc(y)
            self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.m = CRNN(len(self.chars) + 1).to(self.dev); self.m.load_state_dict(ck['state']); self.m.eval(); self.torch = torch
        else:   # 웹용 ONNX(한 번에 한 행)
            import onnxruntime as ort
            self.s = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
            self.chars = json.load(open(os.environ['BANGAKBON_CHARSET'], encoding='utf-8'))['chars']; self.torch = None
    def decode(self, lp):
        k = lp.argmax(-1); out, prev = [], 0
        for c in k:
            if c != prev and c: out.append(self.chars[c - 1])
            prev = c
        m = lp.max(-1); keep = k > 0
        conf = float(np.exp(m[keep]).mean()) if keep.any() else 0.0
        return nfc(''.join(out)), conf
    def read(self, arrs, bs=32):
        res = []
        if self.torch is None:
            for a in arrs:
                y = self.s.run(None, {'x': a[None, None]})[0][0]
                y = y - y.max(-1, keepdims=True); y = y - np.log(np.exp(y).sum(-1, keepdims=True)); res.append(self.decode(y))
            return res
        torch = self.torch
        with torch.no_grad():
            for i in range(0, len(arrs), bs):
                ch = arrs[i:i + bs]; W = max(a.shape[1] for a in ch); x = np.zeros((len(ch), 1, H, W), np.float32)
                for j, a in enumerate(ch): x[j, 0, :, :a.shape[1]] = a
                y = self.m(torch.from_numpy(x).to(self.dev)).log_softmax(-1).cpu().numpy()
                res += [self.decode(r) for r in y]
        return res

def newest_model():
    """드라이브에 학습한 모델(.pt)이 있으면 그것, 없으면 판독기에 올린 공개 모델(ONNX)을 GitHub에서 받는다."""
    for v in (9, 8, 7):
        p = f'{ROOT}/gyeongpan_model_v{v}.pt'
        if os.path.exists(p): return p
    html = requests.get(REPO + 'web/index.html', timeout=60).text
    m = re.search(r"model: \[([^\]]+)\]", html); cs = re.search(r"charset: '([^']+)'", html)
    parts = re.findall(r"'([^']+\.wasm)'", m.group(1)); os.makedirs('/content/bangakbon_model', exist_ok=True)
    dst = '/content/bangakbon_model/' + parts[0].rsplit('_', 1)[0] + '.onnx'
    if not os.path.exists(dst):
        print('드라이브에 학습 모델이 없어 판독기의 공개 모델을 받습니다:', ', '.join(parts))
        data = b''.join(requests.get(REPO + 'web/' + p, timeout=300).content for p in parts); open(dst, 'wb').write(data)
    cpath = '/content/bangakbon_model/' + cs.group(1)
    open(cpath, 'wb').write(requests.get(REPO + 'web/' + cs.group(1), timeout=60).content)
    os.environ['BANGAKBON_CHARSET'] = cpath
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'onnxruntime'], check=False)
    return dst

# ---------- 한 면 / 한 책 ----------
def read_face(R, im, no, side, split, srcW, srcH):
    im = norm_width(im)
    q = {'no': no, 'side': side, 'split': split, 'srcW': srcW, 'srcH': srcH, 'W': im.width, 'H': im.height}
    try: t, bt, cols = segment(im)[:3]
    except Exception: t, bt, cols = 0, im.height, []
    cols = [[int(a), int(c)] for a, c in cols]; q.update(t=int(t), bt=int(bt), cols=cols)
    if len(cols) < 6: q.update(cols=[], lines=[], conf=[], status='empty'); return q
    arrs = [prep_arr(im.crop((max(0, a - 3), t, c + 3, bt))) for a, c in cols]
    out = R.read(arrs); q['lines'] = [s for s, _ in out]; q['conf'] = [round(c, 3) for _, c in out]
    for k in (-1, 0):   # 판 가장자리에 붙은 짧고 흐린 행(테두리·얼룩)은 뺀다
        if len(q['lines']) > 6 and len(G(q['lines'][k])) < 3 and q['conf'][k] < 0.6:
            q['lines'].pop(k); q['conf'].pop(k); q['cols'].pop(k)
    lens = [len(G(s)) for s in q['lines']]
    cover = all(n < 3 for n in lens[:6]); avg = sum(q['conf']) / len(q['conf'])
    q['status'] = 'nomatch' if (cover or float(np.median(lens)) < 8 or avg < 0.55) else 'ok'
    return q

def read_book(R, pdf, name):
    doc = fitz.open(pdf); pages = []; t0 = time.time()
    for i in range(len(doc)):
        g = page_gray(doc, i)
        for im, side, split in split_spread(g):
            pages.append(read_face(R, im, i + 1, side, split, g.width, g.height))
        if (i + 1) % 10 == 0: print(f'  {name}: {i+1}/{len(doc)}쪽 ({(time.time()-t0)/(i+1):.1f}초/쪽)', flush=True)
    return {'format': 'bangakbon-ocr', 'version': 1, 'src': name, 'model': R.label,
            'made': time.strftime('%Y-%m-%d %H:%M'), 'pages': pages}

def to_text(d):
    out = []
    for q in d['pages']:
        lab = f"{q['no']}쪽" + (' ' + q['side'] if q['side'] else '')
        if q['status'] == 'empty': out.append(f'## {lab}\n(본문 없음)\n')
        elif q['status'] == 'nomatch': out.append(f'## {lab}\n(표지·빈 면으로 보임)\n')
        else: out.append(f'## {lab}\n' + '\n'.join(q['lines']) + '\n')
    return '\n'.join(out)

def get_commons(title, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 100000: return dst
    r = requests.get('https://commons.wikimedia.org/w/api.php', params={'action': 'query', 'prop': 'imageinfo', 'iiprop': 'url',
                     'format': 'json', 'titles': title}, headers=UA, timeout=60).json()
    url = list(r['query']['pages'].values())[0]['imageinfo'][0]['url'].split('?')[0]
    for _ in range(5):
        x = requests.get(url, headers=UA, timeout=300)
        if x.status_code == 200 and x.content[:4] == b'%PDF': open(dst, 'wb').write(x.content); return dst
        time.sleep(10)
    raise RuntimeError('내려받지 못함: ' + title)

def main(model_path=None):
    os.makedirs(IN, exist_ok=True); os.makedirs(OUT, exist_ok=True)
    R = Reader(model_path or newest_model()); print('판독 모델:', R.label, '/', getattr(R, 'dev', 'onnx'))
    jobs = [(p, os.path.splitext(os.path.basename(p))[0]) for p in sorted(glob.glob(IN + '/*.pdf'))]
    ct = IN + '/commons.txt'
    if os.path.exists(ct):
        os.makedirs('/content/commons_pdf', exist_ok=True)
        for line in open(ct, encoding='utf-8'):
            t = line.strip()
            if not t or t.startswith('#'): continue
            if not t.startswith('File:'): t = 'File:' + t
            nm = re.sub(r'\.pdf$', '', t[5:], flags=re.I)
            try: jobs.append((get_commons(t, f'/content/commons_pdf/{nm}.pdf'), nm))
            except Exception as e: print('건너뜀', t, e)
    if not jobs: print('판독할 PDF가 없습니다. 드라이브', IN, '폴더에 PDF를 넣거나 commons.txt를 만들어 주세요.'); return
    for pdf, nm in jobs:
        js = f'{OUT}/{nm}.json'
        if os.path.exists(js) and json.load(open(js, encoding='utf-8')).get('model') == R.label:
            print('이미 판독함:', nm); continue
        t0 = time.time(); d = read_book(R, pdf, nm)
        json.dump(d, open(js, 'w', encoding='utf-8'), ensure_ascii=False)
        open(f'{OUT}/{nm}.txt', 'w', encoding='utf-8').write(to_text(d))
        ok = sum(q['status'] == 'ok' for q in d['pages'])
        print(f'{nm}: {len(d["pages"])}면 중 본문 {ok}면, {time.time()-t0:.0f}초 → {js}', flush=True)

if __name__ == '__main__' and not os.environ.get('BANGAKBON_NO_MAIN'):
    main()
