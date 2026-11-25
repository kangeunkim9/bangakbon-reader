# PARSeq 파인튜닝 시험 셀(Colab).
#@title ㉕ PARSeq 시험: 일본 국립국회도서관 NDLOCR-Lite의 PARSeq(tiny, 24x768)를 옛한글 자모로 다시 가르쳐 10차와 비교
# 10차와 똑같은 학습 자료(경판 행, 고쳐 보낸 면, AI허브 목판본, AI허브 필사본)를 쓰고,
# 금향졍긔 권하 뒤쪽 절반(학습에 안 씀)으로 오류율을 비교한다. 사이트에는 아직 올리지 않는다.
# NDLOCR-Lite 모델: CC BY 4.0, https://github.com/ndl-lab/ndlocr-lite
import subprocess, sys, os
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'timm', 'pytorch-lightning', 'nltk'], check=False)
if not os.path.isdir('/content/parseq'):
    subprocess.run(['git', 'clone', '-q', 'https://github.com/baudm/parseq', '/content/parseq'], check=True)
    subprocess.run(['git', '-C', '/content/parseq', 'checkout', '-q', '1902db043c029a7e03a3818c616c06600af574be'], check=True)
sys.path.insert(0, '/content/parseq')
import requests
RAW = 'https://raw.githubusercontent.com/kangeunkim9/bangakbon-reader/main/training/'
src10 = requests.get(RAW + 'train_v10.py', timeout=60).text
exec(src10[:src10.index('# ---------------- 4)')], globals())     # 10차와 같은 학습 자료: KTR, KVA, INTR, AHtr, AHte, PS_tr, PS_te

exec(src10[src10.index('# ---------------- 4)'):src10.index('def prep9')], globals())   # lowres
import math, warnings
warnings.filterwarnings('ignore')
from strhub.models.parseq.system import PARSeq

# ---------------- 1) NDL 체크포인트 받기 ----------------
CKURL = 'https://lab.ndl.go.jp/dataset/ndlocr-lite/ver1.2_models/parseq-ndl-24x768-100-tiny-153epoch-tegaki3-r8data-202604.ckpt'
NDLCK = ROOT + '/ndl/' + CKURL.rsplit('/', 1)[1]
if not os.path.exists(NDLCK):
    os.makedirs(ROOT + '/ndl', exist_ok=True)
    with requests.get(CKURL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open('/content/ndl.ckpt', 'wb') as f:
            for b in r.iter_content(1 << 20): f.write(b)
    shutil.copy('/content/ndl.ckpt', NDLCK)
ck = torch.load(NDLCK, map_location='cpu', weights_only=False)
hp = dict(ck['hyper_parameters'])
print('NDL 모델 설정:', {k: hp[k] for k in ('img_size', 'patch_size', 'embed_dim', 'enc_depth', 'dec_depth', 'max_label_length') if k in hp}, flush=True)

# ---------------- 2) 글자표를 옛한글 자모로 바꾼 모델 ----------------
CH = list(load_model(ROOT + '/gyeongpan_model_v10.pt', 'cpu')[1].chars)   # 10차와 같은 자모 222종
CHS = set(CH); CHARSET = ''.join(CH)
IMG = list(hp['img_size']); MAXL = int(hp['max_label_length'])
lab = lambda t: ''.join(c for c in nfd(t) if c in CHS)
hp.update(charset_train=CHARSET, charset_test=CHARSET, batch_size=48, lr=5e-4, warmup_pct=0.03, weight_decay=0.01)
sysm = PARSeq(**hp)
own = sysm.state_dict(); keep = 0; skip = []
for k, v in ck['state_dict'].items():
    if k in own and own[k].shape == v.shape: own[k] = v; keep += 1
    else: skip.append(k)
sysm.load_state_dict(own)
print('NDL 가중치 가져옴', keep, '개 / 새로 배울 것(글자표 관련)', skip, flush=True)
sysm = sysm.to(DEV)
NEW = {n for n, _ in sysm.named_parameters() if any(n == s for s in skip)}

def fit(pairs):
    out = [(p, t) for p, t in pairs if 1 <= len(lab(t)) <= MAXL]
    return out, len(pairs) - len(out)
KTRp, a = fit(KTR); INTRp, b = fit(INTR); AHtrp, c = fit(AHtr); PS_trp, d = fit(PS_tr)
print(f'자모 {MAXL}개를 넘어 뺀 행: 경판 {a}, 보낸 면 {b}, 목판본 {c}, 필사본 {d}', flush=True)

# ---------------- 3) 전처리: 10차와 같은 덧붙임, 크기만 24x768(3채널, -1~1) ----------------
def prep_p(src, aug=False):
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
    a = np.asarray(im.resize((IMG[1], IMG[0]), Image.BICUBIC), dtype=np.float32) / 127.5 - 1.0
    if aug: a = np.clip(a + np.random.normal(0, 0.06, a.shape), -1, 1)
    return torch.from_numpy(np.repeat(a[None], 3, 0).astype(np.float32))
class DSP(torch.utils.data.Dataset):
    def __init__(s, pairs, aug): s.p = pairs; s.aug = aug
    def __len__(s): return len(s.p)
    def __getitem__(s, i): p, t = s.p[i]; return prep_p(p, s.aug), lab(t)
colp = lambda b: (torch.stack([x for x, _ in b]), [t for _, t in b])

@torch.no_grad()
def read_p(m, srcs, bs=64):
    m.eval(); out = []
    for i in range(0, len(srcs), bs):
        x = torch.stack([prep_p(s) for s in srcs[i:i + bs]]).to(DEV)
        with torch.autocast('cuda', dtype=torch.float16):
            pr = m(x).float().softmax(-1)
        out += [nfc(s) for s in m.tokenizer.decode(pr)[0]]
    return out
PSV = random.Random(2).sample(PS_te, min(600, len(PS_te)))
def val_p(m):
    return cer(read_p(m, [p for p, _ in KVA]), [t for _, t in KVA]), cer(read_p(m, [p for p, _ in PSV]), [t for _, t in PSV])

# ---------------- 4) 학습 (에폭마다 드라이브에 저장, 끊기면 이어서) ----------------
EP = 8; NAH = min(len(AHtrp), 25000); BS = hp['batch_size']; REP_IN = 3
per = len(KTRp) + REP_IN * len(INTRp) + NAH + len(PS_trp); steps = EP * (per // BS)
pg = [{'params': [p for n, p in sysm.named_parameters() if n in NEW], 'lr': 1e-3},
      {'params': [p for n, p in sysm.named_parameters() if n not in NEW], 'lr': 3e-4}]
opt = torch.optim.AdamW(pg, weight_decay=0.01)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-3, 3e-4], total_steps=steps, pct_start=0.05)
scaler = torch.cuda.amp.GradScaler()
CKP = ROOT + '/ckpt_parseq.pt'; SAVEP = ROOT + '/parseq_model_v1.pt'; ep0 = 1; best = None; logp = []
if os.path.exists(CKP):
    c0 = torch.load(CKP, map_location=DEV, weights_only=False)
    if c0.get('per') == per and c0.get('chars') == CHARSET:
        sysm.load_state_dict(c0['state']); opt.load_state_dict(c0['opt']); sch.load_state_dict(c0['sch'])
        ep0 = c0['ep'] + 1; best = c0['best']; logp = c0['log']; print('이어서 학습:', ep0, '에폭부터', flush=True)
print(f'한 에폭 {per}행, {per // BS}걸음 x {EP}에폭', flush=True)
for ep in range(ep0, EP + 1):
    tr = KTRp + INTRp * REP_IN + random.sample(AHtrp, NAH) + PS_trp
    dl = torch.utils.data.DataLoader(DSP(tr, True), batch_size=BS, shuffle=True, num_workers=2, collate_fn=colp, drop_last=True)
    sysm.train(); t0 = time.time(); tot = 0; n = 0
    for x, ys in dl:
        x = x.to(DEV)
        with torch.autocast('cuda', dtype=torch.float16):
            loss = sysm.training_step((x, ys), n)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(sysm.parameters(), 20); scaler.step(opt); scaler.update()
        if sch.last_epoch + 1 < steps: sch.step()
        tot += loss.item(); n += 1
        if n % 500 == 0: print(f'  {ep}에폭 {n}걸음 loss {tot/n:.3f} {time.time()-t0:.0f}s', flush=True)
    k, q = val_p(sysm); sc = k + q; logp.append((ep, tot / n, k, q))
    if best is None or sc < best:
        best = sc; torch.save({'state': sysm.state_dict(), 'hp': hp}, SAVEP)
    torch.save({'state': sysm.state_dict(), 'opt': opt.state_dict(), 'sch': sch.state_dict(), 'ep': ep, 'best': best, 'log': logp, 'chars': CHARSET, 'per': per}, CKP)
    print(f'epoch {ep}  loss {tot/n:.3f}  경판 검증 {k*100:.2f}%  필사본 {q*100:.2f}%  {time.time()-t0:.0f}s', flush=True)

# ---------------- 5) 시험: 10차와 나란히 ----------------
sysm.load_state_dict(torch.load(SAVEP, map_location=DEV, weights_only=False)['state'])
m10, cs10 = load_model(ROOT + '/gyeongpan_model_v10.pt', DEV)
R = {}
R['val_v10'] = cer(predict(m10, cs10, [p for p, _ in KVA], DEV), [t for _, t in KVA]); R['pilsa_v10'] = cer(predict(m10, cs10, [p for p, _ in PSV], DEV), [t for _, t in PSV])
R['val_p'], R['pilsa_p'] = val_p(sysm)
sm = random.Random(1).sample(AHte, min(800, len(AHte)))
R['simcheong_v10'] = cer(predict(m10, cs10, [p for p, _ in sm], DEV), [t for _, t in sm]); R['simcheong_p'] = cer(read_p(sysm, [p for p, _ in sm]), [t for _, t in sm])
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
        im = B['norm_width'](B['page_gray'](doc, i)); t, bt, cols = B['segment'](im)[:3]
        crops = [im.crop((max(0, a - 3), t, c + 3, bt)) for a, c in cols]
        for s in read_face(crops):
            tk = toks(s); flat += tk; own_ += [no] * len(tk)
    err = collections.Counter(); nn_ = collections.Counter(no >= 36 for no in own_)
    for op in Lv.editops(flat, GT):
        p = min(op.src_pos, len(own_) - 1); err[own_[p] >= 36] += 1
    return err[True] / nn_[True]
R10 = B['Reader'](ROOT + '/gyeongpan_model_v10.pt')
R['geum_v10'] = geum(lambda crops: [s for s, _ in R10.read([B['prep_arr'](c) for c in crops], bs=1)])
R['geum_p'] = geum(lambda crops: read_p(sysm, crops))
print(f"\n[금향졍긔 뒤쪽 절반]  10차 {R['geum_v10']*100:.2f}%  PARSeq {R['geum_p']*100:.2f}%  (언어 모델 없이)")
print(f"[경판 검증]          10차 {R['val_v10']*100:.2f}%  PARSeq {R['val_p']*100:.2f}%")
print(f"[필사본 시험 행]      10차 {R['pilsa_v10']*100:.2f}%  PARSeq {R['pilsa_p']*100:.2f}%")
print(f"[AI허브 심청전_04]   10차 {R['simcheong_v10']*100:.2f}%  PARSeq {R['simcheong_p']*100:.2f}%")
json.dump({'result': R, 'log': logp, 'hp': {k: v for k, v in hp.items() if k not in ('charset_train', 'charset_test')}}, open(ROOT + '/result_parseq_v1.json', 'w'), ensure_ascii=False)
