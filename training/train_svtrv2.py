# SVTRv2 파인튜닝 시험 셀(Colab).
#@title ㉛ SVTRv2 시험: OpenOCR의 중국어 SVTRv2(server, CTC)를 옛한글 자모로 다시 가르쳐 10차와 비교
# 10차와 똑같은 학습 자료를 쓰고 같은 시험(금향졍긔 권하 뒤쪽 절반 등)으로 비교한다. 사이트에는 아직 올리지 않는다.
# SVTRv2: Du et al., ICCV 2025. 코드와 가중치 OpenOCR(Apache-2.0), https://github.com/Topdu/OpenOCR
# CTC 모델이라 지금 사이트의 언어 모델 보정(자모 7-gram 빔 탐색)을 그대로 붙일 수 있다.
import subprocess, sys, os
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'imgaug', 'lmdb', 'pyclipper', 'rapidfuzz'], check=False)
if not os.path.isdir('/content/openocr'):
    subprocess.run(['git', 'clone', '-q', 'https://github.com/Topdu/OpenOCR', '/content/openocr'], check=True)
    subprocess.run(['git', '-C', '/content/openocr', 'checkout', '-q', '1ccfc6ee6161f7133b192e16af3a9265273997ba'], check=True)
sys.path.insert(0, '/content/openocr')
import requests
RAW = 'https://raw.githubusercontent.com/kangeunkim9/bangakbon-reader/main/training/'
src10 = requests.get(RAW + 'train_v10.py', timeout=60).text
exec(src10[:src10.index('# ---------------- 4)')], globals())     # 10차와 같은 학습 자료
exec(src10[src10.index('# ---------------- 4)'):src10.index('def prep9')], globals())   # lowres

import math, warnings, yaml, gc
for _k in ('sysm', 'm10', 'R10', 'opt', 'svm'): globals().pop(_k, None)   # 앞 셀(PARSeq)이 남긴 GPU 메모리 비우기
gc.collect(); torch.cuda.empty_cache()
warnings.filterwarnings('ignore')
from openrec.modeling import build_model
AMP = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
scaler = torch.cuda.amp.GradScaler(enabled=(AMP == torch.float16))

# ---------------- 1) 중국어 SVTRv2 가중치 받기 ----------------
CKURL = 'https://github.com/Topdu/OpenOCR/releases/download/develop0.0.1/openocr_svtrv2_ch.pth'
SVCK = ROOT + '/svtrv2/openocr_svtrv2_ch.pth'
if not os.path.exists(SVCK):
    os.makedirs(ROOT + '/svtrv2', exist_ok=True)
    with requests.get(CKURL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open('/content/sv.pth', 'wb') as f:
            for b in r.iter_content(1 << 20): f.write(b)
    shutil.copy('/content/sv.pth', SVCK)

# ---------------- 2) 글자표를 옛한글 자모로 바꾼 모델 ----------------
CH = list(load_model(ROOT + '/gyeongpan_model_v10.pt', 'cpu')[1].chars)   # 10차와 같은 자모 222종, 0번은 공백 기호
IDX = {c: i + 1 for i, c in enumerate(CH)}
arch = yaml.safe_load(open('/content/openocr/configs/rec/svtrv2/svtrv2_ch.yml'))['Architecture']
arch['Decoder']['out_channels'] = len(CH) + 1
arch['Decoder']['svtr_encoder']['use_guide'] = False     # 원래 설정은 CTC 쪽 기울기를 인코더로 안 보냄. 여기서는 인코더까지 같이 배움
svm = build_model(arch)
sd = torch.load(SVCK, map_location='cpu', weights_only=False)['state_dict']
own = svm.state_dict(); keep = 0
for k, v in sd.items():
    if k in own and own[k].shape == v.shape: own[k] = v; keep += 1
svm.load_state_dict(own); svm = svm.to(DEV)
NEW = {'decoder.fc.weight', 'decoder.fc.bias'}
print('중국어 SVTRv2 가중치', keep, '개 가져옴 / 새로 배울 것: 출력층', '/ 파라미터', round(sum(p.numel() for p in svm.parameters()) / 1e6, 2), 'M', flush=True)

# ---------------- 3) 전처리: 10차와 같은 덧붙임, 높이 48, 너비는 비율대로(32의 배수), -1~1 ----------------
SH = 48; SMAXW = 1280
def prep_s(src, aug=False):
    im = (Image.open(src) if isinstance(src, str) else src).convert('L').rotate(90, expand=True)
    im = ImageOps.autocontrast(im, cutoff=1)
    if aug:
        im = lowres(im); im = invert_span(im, p=0.12)
        r = random.random()
        if r < 0.25: im = im.filter(ImageFilter.MinFilter(3))
        elif r < 0.4: im = im.filter(ImageFilter.MaxFilter(3))
        if random.random() < 0.5: im = im.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 1.2)))
        if random.random() < 0.5:
            w, h = im.size; im = im.transform((w, h), Image.AFFINE, (1, random.uniform(-0.04, 0.04), 0, 0, 1, random.randint(-5, 5)), fillcolor=255)
        if random.random() < 0.3: im = im.rotate(random.uniform(-1.2, 1.2), fillcolor=255)
        if random.random() < 0.4:
            g = random.uniform(0.6, 1.6); im = im.point(lambda v: int(255 * ((v / 255) ** g)))
    w, h = im.size
    nw = min(SMAXW, max(64, int(w * SH / h * (random.uniform(0.9, 1.1) if aug else 1))))
    a = np.asarray(im.resize((nw, SH), Image.BILINEAR), dtype=np.float32) / 127.5 - 1.0
    if aug: a = np.clip(a + np.random.normal(0, 0.06, a.shape), -1, 1)
    return a
def batch_s(arrs):
    W = max(a.shape[1] for a in arrs); W = (W + 31) // 32 * 32
    x = np.ones((len(arrs), 3, SH, W), np.float32)            # 남는 자리는 흰 바탕(+1)
    for i, a in enumerate(arrs): x[i, :, :, :a.shape[1]] = a
    return torch.from_numpy(x)
enc = lambda t: [IDX[c] for c in nfd(t) if c in IDX]
class DSS(torch.utils.data.Dataset):
    def __init__(s, pairs, aug): s.p = pairs; s.aug = aug
    def __len__(s): return len(s.p)
    def __getitem__(s, i): p, t = s.p[i]; return prep_s(p, s.aug), enc(t)
cols = lambda b: (batch_s([a for a, _ in b]), [t for _, t in b])
def greedy(lp):
    out = []
    for r in lp.argmax(-1).cpu().numpy():
        s, prev = [], 0
        for k in r:
            if k != prev and k != 0: s.append(CH[k - 1])
            prev = k
        out.append(nfc(''.join(s)))
    return out
@torch.no_grad()
def read_s(m, srcs, bs=32):
    m.eval(); out = []
    for i in range(0, len(srcs), bs):
        x = batch_s([prep_s(s) for s in srcs[i:i + bs]]).to(DEV)
        with torch.autocast('cuda', dtype=AMP):
            pr = m(x).float()
        out += greedy(pr)
    return out
PSV = random.Random(2).sample(PS_te, min(600, len(PS_te)))
def val_s(m):
    return cer(read_s(m, [p for p, _ in KVA]), [t for _, t in KVA]), cer(read_s(m, [p for p, _ in PSV]), [t for _, t in PSV])

# ---------------- 4) 학습 (에폭마다 드라이브에 저장, 끊기면 이어서) ----------------
EP = 8; NAH = min(len(AHtr), 25000); BS = 16; REP_IN = 3
per = len(KTR) + REP_IN * len(INTR) + NAH + len(PS_tr); steps = EP * (per // BS)
pg = [{'params': [p for n, p in svm.named_parameters() if n in NEW], 'lr': 1e-3},
      {'params': [p for n, p in svm.named_parameters() if n not in NEW], 'lr': 2e-4}]
opt = torch.optim.AdamW(pg, weight_decay=0.05)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-3, 2e-4], total_steps=steps, pct_start=0.05)
ctc = nn.CTCLoss(blank=0, zero_infinity=True)
CKP = ROOT + '/ckpt_svtrv2.pt'; SAVES = ROOT + '/svtrv2_model_v1.pt'; ep0 = 1; best = None; logs = []
if os.path.exists(CKP):
    c0 = torch.load(CKP, map_location=DEV, weights_only=False)
    if c0.get('per') == per and c0.get('chars') == CH:
        svm.load_state_dict(c0['state']); opt.load_state_dict(c0['opt']); sch.load_state_dict(c0['sch'])
        ep0 = c0['ep'] + 1; best = c0['best']; logs = c0['log']; print('이어서 학습:', ep0, '에폭부터', flush=True)
print(f'한 에폭 {per}행, {per // BS}걸음 x {EP}에폭', flush=True)
for ep in range(ep0, EP + 1):
    tr = KTR + INTR * REP_IN + random.sample(AHtr, NAH) + PS_tr
    dl = torch.utils.data.DataLoader(DSS(tr, True), batch_size=BS, shuffle=True, num_workers=2, collate_fn=cols, drop_last=True)
    svm.train(); t0 = time.time(); tot = 0; n = 0
    for x, tg in dl:
        x = x.to(DEV)
        with torch.autocast('cuda', dtype=AMP):
            y = svm(x)
        lp = y.float().log_softmax(-1).permute(1, 0, 2)          # T,B,C
        tl = torch.tensor([len(t) for t in tg])
        loss = ctc(lp, torch.tensor(sum(tg, [])), torch.full((len(tg),), lp.shape[0], dtype=torch.long), tl)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(svm.parameters(), 5); scaler.step(opt); scaler.update()
        if sch.last_epoch + 1 < steps: sch.step()
        tot += loss.item(); n += 1
        if n % 500 == 0: print(f'  {ep}에폭 {n}걸음 loss {tot/n:.3f} {time.time()-t0:.0f}s', flush=True)
    k, q = val_s(svm); sc = k + q; logs.append((ep, tot / n, k, q))
    if best is None or sc < best:
        best = sc; torch.save({'state': svm.state_dict(), 'arch': arch, 'chars': CH, 'H': SH}, SAVES)
    torch.save({'state': svm.state_dict(), 'opt': opt.state_dict(), 'sch': sch.state_dict(), 'ep': ep, 'best': best, 'log': logs, 'chars': CH, 'per': per}, CKP)
    print(f'epoch {ep}  loss {tot/n:.3f}  경판 검증 {k*100:.2f}%  필사본 {q*100:.2f}%  {time.time()-t0:.0f}s', flush=True)

# ---------------- 5) 시험: 10차와 나란히 ----------------
svm.load_state_dict(torch.load(SAVES, map_location=DEV, weights_only=False)['state'])
m10, cs10 = load_model(ROOT + '/gyeongpan_model_v10.pt', DEV)
R = {}
R['val_v10'] = cer(predict(m10, cs10, [p for p, _ in KVA], DEV), [t for _, t in KVA]); R['pilsa_v10'] = cer(predict(m10, cs10, [p for p, _ in PSV], DEV), [t for _, t in PSV])
R['val_s'], R['pilsa_s'] = val_s(svm)
sm = random.Random(1).sample(AHte, min(800, len(AHte)))
R['simcheong_v10'] = cer(predict(m10, cs10, [p for p, _ in sm], DEV), [t for _, t in sm]); R['simcheong_s'] = cer(read_s(svm, [p for p, _ in sm]), [t for _, t in sm])
from rapidfuzz.distance import Levenshtein as Lv
os.environ['BANGAKBON_NO_MAIN'] = '1'
B = {'__name__': 'bulk'}; exec(requests.get(RAW + 'bulk_ocr.py', timeout=60).text, B)
KPG = json.load(gzip.open(ROOT + '/private/kohico_pages.json.gz', 'rt', encoding='utf-8'))
CL = re.compile(r'[ᄀ-ᅟꥠ-꥿]+[ᅠ-ᆧힰ-ퟆ]+[ᆨ-ᇿퟋ-ퟻ]*|.', re.S)
toks = lambda s: CL.findall(unicodedata.normalize('NFC', s))
GP = [v for k, v in KPG.items() if k.startswith('금향정기_2')][0]
GT = [g for g, _ in fix_rep([(t, 0) for k in sorted(GP) if k not in ('31ᄀ', '31ᄂ') for t in toks(re.sub(r'\s+', '', GP[k]))])[0]]
pdf = B['get_commons']('File:CNTS-00109637790 금향졍긔 이종. 권하.pdf', '/content/geumhyang2.pdf'); doc = B['fitz'].open(pdf)
def geum(read_face):
    own_ = []; flat = []
    for i in range(len(doc)):
        no = i + 1
        if not (3 <= no <= 62 or no == 65): continue
        im = B['norm_width'](B['page_gray'](doc, i)); t, bt, cols_ = B['segment'](im)[:3]
        crops = [im.crop((max(0, a - 3), t, c + 3, bt)) for a, c in cols_]
        for s in read_face(crops):
            tk = toks(s); flat += tk; own_ += [no] * len(tk)
    err = collections.Counter(); nn_ = collections.Counter(no >= 36 for no in own_)
    for op in Lv.editops(flat, GT):
        p = min(op.src_pos, len(own_) - 1); err[own_[p] >= 36] += 1
    return err[True] / nn_[True]
R10 = B['Reader'](ROOT + '/gyeongpan_model_v10.pt')
R['geum_v10'] = geum(lambda crops: [s for s, _ in R10.read([B['prep_arr'](c) for c in crops], bs=1)])
R['geum_s'] = geum(lambda crops: read_s(svm, crops))
print(f"\n[금향졍긔 뒤쪽 절반]  10차 {R['geum_v10']*100:.2f}%  SVTRv2 {R['geum_s']*100:.2f}%  (언어 모델 없이)")
print(f"[경판 검증]          10차 {R['val_v10']*100:.2f}%  SVTRv2 {R['val_s']*100:.2f}%")
print(f"[필사본 시험 행]      10차 {R['pilsa_v10']*100:.2f}%  SVTRv2 {R['pilsa_s']*100:.2f}%")
print(f"[AI허브 심청전_04]   10차 {R['simcheong_v10']*100:.2f}%  SVTRv2 {R['simcheong_s']*100:.2f}%")
json.dump({'result': R, 'log': logs}, open(ROOT + '/result_svtrv2_v1.json', 'w'), ensure_ascii=False)
