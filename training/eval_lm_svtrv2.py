# 언어 모델 보정까지 붙여 10차와 SVTRv2를 금향졍긔로 비교하는 셀(Colab).
#@title ㉞ 언어 모델 보정을 붙여 10차 대 SVTRv2 비교 (금향졍긔 권하) + SVTRv2 브라우저용 파일 만들기
# 사이트와 똑같은 언어 모델(lm7.gz.wasm)과 빔 탐색(engine.js KP.beam을 그대로 옮김)을 쓴다.
# 가중치(alpha, beta)는 사이트 기본값(0.3, 1.0)으로 한 번, 금향 앞쪽 절반(검증)에서 고른 값으로 한 번 잰다. 숫자는 뒤쪽 절반(시험).
import subprocess, sys, os, re, io, gzip, json, math, time, collections, unicodedata, requests
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'rapidfuzz', 'pymupdf', 'onnx'], check=False)
if not os.path.isdir('/content/openocr'):
    subprocess.run(['git', 'clone', '-q', 'https://github.com/Topdu/OpenOCR', '/content/openocr'], check=True)
sys.path.insert(0, '/content/openocr')
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image, ImageOps
from rapidfuzz.distance import Levenshtein as Lv
from google.colab import drive; drive.mount('/content/drive')
ROOT = '/content/drive/MyDrive/방각본_판독기'; DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
RAWG = 'https://raw.githubusercontent.com/kangeunkim9/bangakbon-reader/main/'
SV = globals().get('SV_MODEL', ROOT + '/1_모델/svtrv2_model_v2.pt')

# ---------------- 1) 금향졍긔 권하를 사이트와 같은 방식으로 행 나누기 ----------------
os.environ['BANGAKBON_NO_MAIN'] = '1'
B = {'__name__': 'bulk'}; exec(requests.get(RAWG + 'training/bulk_ocr.py', timeout=60).text, B)
pdf = B['get_commons']('File:CNTS-00109637790 금향졍긔 이종. 권하.pdf', '/content/geumhyang2.pdf'); doc = B['fitz'].open(pdf)
FACES = []
for i in range(len(doc)):
    no = i + 1
    if not (3 <= no <= 62 or no == 65): continue
    im = B['norm_width'](B['page_gray'](doc, i)); t, bt, cols = B['segment'](im)[:3]
    FACES.append((no, [im.crop((max(0, a - 3), t, c + 3, bt)) for a, c in cols]))
print('면', len(FACES), '행', sum(len(c) for _, c in FACES), flush=True)

# ---------------- 2) 행마다 글자 확률표(log) 뽑기: 10차, SVTRv2 ----------------
R10 = B['Reader'](ROOT + '/1_모델/gyeongpan_model_v10.pt'); CH = list(R10.chars)
LP = {'10차': [], 'SVTRv2': []}
with torch.no_grad():
    for _, crops in FACES:
        for c in crops:
            a = B['prep_arr'](c)
            LP['10차'].append(R10.m(torch.from_numpy(a[None, None]).to(R10.dev)).log_softmax(-1)[0].float().cpu().numpy())
import openrec.modeling.encoders.svtrv2_lnconv_two33 as _sv2
from openrec.modeling import build_model
def _attn_fwd(self, x):
    Bn, N, _ = x.shape
    q, k, v = self.qkv(x).reshape(Bn, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
    x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
    return self.proj_drop(self.proj(x.transpose(1, 2).reshape(Bn, N, self.dim)))
_sv2.Attention.forward = _attn_fwd
ck = torch.load(SV, map_location='cpu', weights_only=False)
assert list(ck['chars']) == CH, '글자표가 10차와 다릅니다'
svm = build_model(ck['arch']); svm.load_state_dict(ck['state']); svm = svm.to(DEV).eval()
SH, SMAXW = ck.get('H', 48), 1280
def prep_s(im):
    im = ImageOps.autocontrast(im.convert('L').rotate(90, expand=True), cutoff=1); w, h = im.size
    nw = min(SMAXW, max(64, int(w * SH / h)))
    a = np.asarray(im.resize((nw, SH), Image.BILINEAR), dtype=np.float32) / 127.5 - 1.0
    W = (nw + 31) // 32 * 32; x = np.ones((3, SH, W), np.float32); x[:, :, :nw] = a
    return x
with torch.no_grad():
    for _, crops in FACES:
        for c in crops:
            pr = svm(torch.from_numpy(prep_s(c)[None]).to(DEV)).float()[0]
            LP['SVTRv2'].append(torch.log(pr.clamp_min(1e-9)).cpu().numpy())
OWN = [no for no, crops in FACES for _ in crops]

# ---------------- 3) 사이트와 같은 언어 모델·빔 탐색 (engine.js에서 그대로 옮김) ----------------
LMJ = json.loads(gzip.decompress(requests.get(RAWG + 'web/lm7.gz.wasm', timeout=120).content))
CTX, V, N = LMJ['ctx'], LMJ['V'], LMJ['N']; MEMO = {}
def lmp(h, c):
    key = (h, c)
    if key in MEMO: return MEMO[key]
    e = CTX.get(h)
    if h == '': v = (e[2].get(c, 0) + e[1] / V) / (e[0] + e[1])
    else:
        lo = lmp(h[1:], c); v = lo if (not e or e[0] == 0) else (e[2].get(c, 0) + e[1] * lo) / (e[0] + e[1])
    if len(MEMO) > 3000000: MEMO.clear()
    MEMO[key] = v; return v
NEG = -1e30
def lse(a, b):
    if a < -1e29: return b
    if b < -1e29: return a
    m = max(a, b); return m + math.log(math.exp(a - m) + math.exp(b - m))
def greedy(lp):
    out, prev = [], 0
    for k in lp.argmax(-1):
        if k != prev and k != 0: out.append(CH[k - 1])
        prev = k
    return unicodedata.normalize('NFC', ''.join(out))
THR = math.log(1e-3); PAD = '^' * (N - 1)
def beam(lp, alpha=0.3, beta=1.0, Bw=10):
    beams = {'': [0.0, NEG]}
    for t in range(lp.shape[0]):
        row = lp[t]; cand = np.nonzero(row[1:] > THR)[0] + 1; nb = {}
        def get(k):
            e = nb.get(k)
            if e is None: e = nb[k] = [NEG, NEG]
            return e
        for pre, (pb, pn) in beams.items():
            tot = lse(pb, pn); e = get(pre); e[0] = lse(e[0], tot + float(row[0])); last = pre[-1] if pre else None
            ctx = (PAD + pre)[-(N - 1):]
            for i in cand:
                ch = CH[i - 1]; lc = float(row[i])
                if ch == last:
                    e = get(pre); e[1] = lse(e[1], pn + lc)
                    if pb > NEG:
                        f = get(pre + ch); f[1] = lse(f[1], pb + lc + alpha * math.log(lmp(ctx, ch)) + beta)
                else:
                    f = get(pre + ch); f[1] = lse(f[1], tot + lc + alpha * math.log(lmp(ctx, ch)) + beta)
        beams = dict(sorted(nb.items(), key=lambda kv: -lse(*kv[1]))[:Bw])
    return unicodedata.normalize('NFC', max(beams.items(), key=lambda kv: lse(*kv[1]))[0])

# ---------------- 4) 채점: 금향 KOHICO 전사와 통째로 맞춰 앞 절반(검증)·뒤 절반(시험) 오류를 따로 ----------------
src10 = requests.get(RAWG + 'training/train_v10.py', timeout=60).text
_fr = src10[src10.index('def fix_rep'):src10.index('def fix_group')]; exec(_fr)   # 반복 부호 정답 바로잡기(10차와 같게)
KPG = json.load(gzip.open(ROOT + '/4_말뭉치_비공개/kohico_pages.json.gz', 'rt', encoding='utf-8'))
CL = re.compile(r'[ᄀ-ᅟꥠ-꥿]+[ᅠ-ᆧힰ-ퟆ]+[ᆨ-ᇿퟋ-ퟻ]*|.', re.S)
toks = lambda s: CL.findall(unicodedata.normalize('NFC', s))
GP = [v for k, v in KPG.items() if k.startswith('금향정기_2')][0]
GT = [g for g, _ in fix_rep([(t, 0) for k in sorted(GP) if k not in ('31ᄀ', '31ᄂ') for t in toks(re.sub(r'\s+', '', GP[k]))])[0]]
def score(texts):
    tk = [toks(s) for s in texts]; flat = [t for x in tk for t in x]; own = [no for no, x in zip(OWN, tk) for _ in x]
    err = collections.Counter(); n = collections.Counter(no >= 36 for no in own)
    for op in Lv.editops(flat, GT):
        p = min(op.src_pos, len(own) - 1); err[own[p] >= 36] += 1
    return err[False] / n[False] * 100, err[True] / n[True] * 100      # (검증, 시험)
RES = {}
for name in ('10차', 'SVTRv2'):
    t0 = time.time(); L = LP[name]
    g = [greedy(lp) for lp in L]; RES[name + ' 보정 없음'] = score(g)
    b0 = [beam(lp) for lp in L]; RES[name + ' 보정(사이트 기본 0.3, 1.0)'] = score(b0)
    best = None
    for a in (0.2, 0.3, 0.5, 0.8):
        for bt in (0.0, 1.0, 2.0):
            if (a, bt) == (0.3, 1.0): v = RES[name + ' 보정(사이트 기본 0.3, 1.0)'][0]
            else: v = score([beam(lp, a, bt) if no < 36 else g[k] for k, (no, lp) in enumerate(zip(OWN, L))])[0]
            if best is None or v < best[0]: best = (v, a, bt)
    RES[name + f' 보정(검증으로 고른 {best[1]}, {best[2]})'] = score([beam(lp, best[1], best[2]) for lp in L])
    print(name, '끝', int(time.time() - t0), '초', flush=True)
print('\n[금향졍긔 권하]  앞 절반(검증)  뒤 절반(시험)')
for k, (v, t) in RES.items(): print(f'  {k:32s} {v:6.2f}%  {t:6.2f}%')
json.dump(RES, open(ROOT + '/기록/result_lm_svtrv2.json', 'w'), ensure_ascii=False, indent=1)

# ---------------- 5) SVTRv2 브라우저용 파일 (ONNX, 출력은 log 확률) ----------------
class _Wrap(nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x): return torch.log(s.m(x).clamp_min(1e-9))
OUTD = ROOT + '/1_모델/web_svtrv2'; os.makedirs(OUTD, exist_ok=True)
svc = svm.cpu().float().eval()
torch.onnx.export(_Wrap(svc), torch.randn(1, 3, SH, 640), OUTD + '/svtrv2.onnx', input_names=['x'], output_names=['y'],
                  dynamic_axes={'x': {0: 'b', 3: 'w'}, 'y': {0: 'b', 1: 't'}}, opset_version=17, dynamo=False)
print('브라우저용 파일:', OUTD + '/svtrv2.onnx', f'{os.path.getsize(OUTD + "/svtrv2.onnx")/1e6:.1f}MB')
