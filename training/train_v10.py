# 10차 학습 셀(Colab). 9차 가중치에서 이어서, 반복 부호(〃) 정답 표기를 원문대로 바로잡아 다시 학습한다.
#@title ㉔ 10차 학습: 반복 부호 정답 바로잡기 (KOHICO '당당ᄒᆞᆫ〃' → 원문 '당〃ᄒᆞᆫ')
# KOHICO 전사는 반복 부호를 '반복 글자를 풀어 쓰고 〃를 어절 끝에 표시'하는 방식이라(예: 당당ᄒᆞᆫ〃장뷔라),
# 이걸로 만든 행 정답을 판독 규칙(원문 표기, 반복 부호는 〃)에 맞게 '당〃ᄒᆞᆫ장뷔라'로 고친다.
# 한 면의 행들을 이어 붙인 상태에서 고쳐서, 어절이 행을 넘어가도 맞게 처리한다.
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'regex', 'pymupdf', 'rapidfuzz', 'onnx', 'onnxruntime'], check=False)
import os, io, re, glob, json, gzip, zipfile, shutil, random, time, collections, unicodedata, requests
import numpy as np, torch, torch.nn as nn
from PIL import Image, ImageOps, ImageFilter
from google.colab import drive; drive.mount('/content/drive')
ROOT = '/content/drive/MyDrive/방각본_판독기'
RAW = 'https://raw.githubusercontent.com/kangeunkim9/bangakbon-reader/main/training/'
get = lambda f: requests.get(RAW + f, timeout=60).text

# ---------------- 0) 7차의 공통 코드(모델·행 나누기)와 월봉긔 검수 정답 ----------------
src7 = get('train_v7.py')
exec(src7[:src7.index('# ---------------- 1)')], globals())   # 공통 코드(모델·행 나누기)
if os.path.exists(ROOT + '/private/wolbong_gt.json'):              # 월봉긔 사람 검수 정답이 드라이브에 있으면 행 자료를 새로 만든다
    exec(src7[src7.index('# ---------------- 1)'):src7.index('# ---------------- 2)')], globals())
elif not os.path.isdir('/content/data/lines'):
    print('월봉긔 검수 정답이 없어 이번에는 빼고 학습한다', flush=True); os.makedirs('/content/data/lines', exist_ok=True)
exec(get('aug_invert.py'), globals())                                                       # invert_span

# ---------------- 1) 경판 행 자료(7차 때 만든 것)와 검증 면 나누기(7차와 똑같이) ----------------
with zipfile.ZipFile(ROOT + '/gyeongpan_lines_v5align.zip') as z: z.extractall('/content/data5')
OUTL = '/content/data5/lines'
P5 = {'wb': load_pairs('/content/data/lines')}
for p, t in load_pairs(OUTL): P5.setdefault(os.path.basename(p).split('_')[0], []).append((p, t))
face5 = lambda p: '_'.join(os.path.basename(p).split('_')[:2])
facew = lambda p: os.path.basename(p).rsplit('_', 1)[0]
REPCL = re.compile(r'[ᄀ-ᅟꥠ-꥿]+[ᅠ-ᆧힰ-ퟆ]*[ᆨ-ᇿퟋ-ퟻ]*|.', re.S)
def fix_rep(items):
    """items: [(글자, 행번호)…]. 〃마다 바로 앞(6글자 안)의 겹친 글자 'XX'를 'X〃'로, 겹친 두 글자 'XYXY'를 'XY〃〃'로 바꾸고 그 〃는 지운다."""
    t = [list(x) for x in items]; i = 0; nfix = nleft = 0
    while i < len(t):
        if t[i][0] != '〃': i += 1; continue
        done = False
        for d in range(1, 7):
            j = i - d
            if j - 1 >= 0 and t[j][0] == t[j - 1][0] and t[j][0] != '〃':
                t[j][0] = '〃'; del t[i]; done = True; break
            if j - 3 >= 0 and t[j][0] == t[j - 2][0] and t[j - 1][0] == t[j - 3][0] and '〃' not in (t[j][0], t[j - 1][0]):
                t[j][0] = t[j - 1][0] = '〃'; del t[i]; done = True; break
        if done: nfix += 1
        else: nleft += 1; i += 1
    return t, nfix, nleft
def fix_group(pairs):
    """한 면(같은 책·같은 쪽)의 행 정답을 행 순서대로 이어서 고치고 다시 행으로 나눈다."""
    nat = lambda p: [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', os.path.basename(p))]
    pairs = sorted(pairs, key=lambda x: nat(x[0]))
    items = [(g, k) for k, (_, txt) in enumerate(pairs) for g in REPCL.findall(unicodedata.normalize('NFC', txt))]
    t, a, b = fix_rep(items); out = [''] * len(pairs)
    for g, k in t: out[k] += g
    return [(p, out[k]) for k, (p, _) in enumerate(pairs) if out[k]], a, b
NF = NL = 0
for k in P5:
    grp = collections.defaultdict(list)
    for p, t in P5[k]: grp[facew(p) if k == 'wb' else face5(p)].append((p, t))
    new = []
    for g in grp.values():
        r, a, b = fix_group(g); new += r; NF += a; NL += b
    P5[k] = new
print('반복 부호 정답 바로잡음', NF, '곳 / 못 고친 〃', NL, '곳', flush=True)
VF5 = {'wb': {'01b', '06b', '11b', '32b'}}
if not P5['wb']: del P5['wb']
for k in P5:
    if k != 'wb': fs = sorted({face5(p) for p, _ in P5[k]}); random.seed(len(k) * 7 + len(fs)); VF5[k] = set(random.sample(fs, min(4, max(1, len(fs) // 6))))
fw = lambda k, p: facew(p) if k == 'wb' else face5(p)
KTR = [x for k in P5 for x in P5[k] if fw(k, x[0]) not in VF5[k]]; KVA = [x for k in P5 for x in P5[k] if fw(k, x[0]) in VF5[k]]
print('경판 학습', len(KTR), '행 / 검증', len(KVA), '행', flush=True)

# ---------------- 2) 판독기에서 고쳐 보낸 면 → 행 자료 ----------------
INB = '/content/inbox_lines'; os.makedirs(INB, exist_ok=True); INTR = []; nf = 0
for js in sorted(glob.glob(ROOT + '/학습자료/*/*.json')):
    try:
        d = json.load(open(js, encoding='utf-8'))
        if '금향' in unicodedata.normalize('NFC', str(d.get('src', '')) + str(d.get('commons', ''))): continue   # 시험 책
        im = Image.open(os.path.join(os.path.dirname(js), d['imageFile'])).convert('L')
        cols = d['cols']; t, bt = int(d['t']), int(d['bt']); k = 0
        for li in d['lines']:
            if li.get('added'): continue               # 새로 넣은 행은 자리(좌표)가 없다
            if k >= len(cols): break
            a, c = cols[k]['a'], cols[k]['c']; k += 1
            txt = re.sub(r'\s+', '', li.get('text', ''))
            if li.get('removed') or len(txt) < 2: continue
            fn = f"{INB}/{os.path.basename(js)[:-5]}_{k:02d}"
            im.crop((max(0, int(a) - 3), t, min(im.width, int(c) + 3), bt)).save(fn + '.png')
            INTR.append((fn + '.png', txt))
        nf += 1
    except Exception as e: print('건너뜀', os.path.basename(js), e)
print('고쳐 보낸 면', nf, '면 →', len(INTR), '행', flush=True)

# ---------------- 3) AI허브 목판본: 춘향전은 학습으로, 심청전_04만 시험 ----------------
os.makedirs('/content/aihub_lines', exist_ok=True)
AHZ = ROOT + '/aihub/aihub_mokpan_lines_v9.zip' if os.path.exists(ROOT + '/aihub/aihub_mokpan_lines_v9.zip') else ROOT + '/aihub/aihub_mokpan_lines.zip'
with zipfile.ZipFile(AHZ) as z: z.extractall('/content/aihub_lines')
AH = [json.loads(l) for l in open('/content/aihub_lines/lines.jsonl', encoding='utf-8')]
AH = [(f"/content/aihub_lines/{r['img']}", r['text'], r['book']) for r in AH if len(G(r['text'])) >= 3 and os.path.exists(f"/content/aihub_lines/{r['img']}")]
hold9 = lambda b: '심청전_04' in b
AHtr = [(p, t) for p, t, b in AH if not hold9(b)]; AHte = [(p, t) for p, t, b in AH if hold9(b)]
print('AI허브 정답 속 〃 개수', sum(t.count('〃') for _, t in AHtr), '/ 행', len(AHtr)); print('AI허브 학습', len(AHtr), '행 / 시험(심청전_04)', len(AHte), '행', flush=True)
# AI허브 필사본(행 자료는 ㉑ 셀이 만든 aihub_pilsa_lines.zip). 유이양문록 전부와 명행정의록 권8은 시험용으로 뺀다.
PS_tr, PS_te = [], []
PZ = ROOT + '/aihub_pilsa/aihub_pilsa_lines.zip'
if os.path.exists(PZ):
    os.makedirs('/content/pilsa_lines', exist_ok=True)
    with zipfile.ZipFile(PZ) as z: z.extractall('/content/pilsa_lines')
    for l in open('/content/pilsa_lines/lines.jsonl', encoding='utf-8'):
        r = json.loads(l); pth = '/content/pilsa_lines/' + r['img']
        if len(G(r['text'])) < 3 or not os.path.exists(pth): continue
        (PS_te if re.search(r'유이양문록|명행정의록8_', r['book']) else PS_tr).append((pth, r['text']))
print('AI허브 필사본 학습', len(PS_tr), '행 / 시험(유이양문록·명행정의록8)', len(PS_te), '행', flush=True)

# ---------------- 4) 덧붙임을 늘린 전처리 ----------------
def lowres(im):
    """작은 사진·낮은 화질 흉내: 줄였다가 다시 키우고, 가끔 JPEG로 뭉갠다."""
    if random.random() < 0.25:
        w, h = im.size; f = random.uniform(0.3, 0.65)
        im = im.resize((max(8, int(w * f)), max(8, int(h * f))), random.choice([Image.BILINEAR, Image.BOX])).resize((w, h), Image.BILINEAR)
    if random.random() < 0.2:
        b = io.BytesIO(); im.save(b, 'JPEG', quality=random.randint(15, 55)); im = Image.open(io.BytesIO(b.getvalue())).convert('L')
    return im
def prep9(path, aug=False):
    im = Image.open(path).convert('L').rotate(90, expand=True)
    im = ImageOps.autocontrast(im, cutoff=1)
    if aug:
        im = lowres(im)
        im = invert_span(im, p=0.12)
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
    nw = min(MAXW, max(32, int(w * H / h * (random.uniform(0.85, 1.15) if aug else 1))))
    a = 1.0 - np.asarray(im.resize((nw, H), Image.BILINEAR), dtype=np.float32) / 255.0
    if aug: a = np.clip(a * random.uniform(0.75, 1.25) + np.random.normal(0, 0.04, a.shape), 0, 1)
    return a
class LineDS9(torch.utils.data.Dataset):
    def __init__(s, pairs, cs, aug): s.p = pairs; s.cs = cs; s.aug = aug
    def __len__(s): return len(s.p)
    def __getitem__(s, i): p, t = s.p[i]; return prep9(p, s.aug), s.cs.encode(t)

# ---------------- 5) 9차 가중치에서 시작 ----------------
m8, cs8 = load_model(ROOT + '/gyeongpan_model_v9.pt', DEV)   # 이름은 m8이지만 9차(직전 모델)
cs9 = Charset(list(cs8.chars) + [t for _, t in INTR + PS_tr + PS_te])
print('자모', len(cs8.chars), '→', len(cs9.chars), '종', flush=True)
random.seed(0); torch.manual_seed(0)
m9 = CRNN(len(cs9)).to(DEV); sd = m8.state_dict(); own = m9.state_dict()
for k, v in sd.items():
    if not k.startswith('fc.'): own[k] = v
for c, i in cs8.idx.items():
    if c in cs9.idx: own['fc.weight'][cs9.idx[c]] = sd['fc.weight'][i]; own['fc.bias'][cs9.idx[c]] = sd['fc.bias'][i]
own['fc.weight'][0] = sd['fc.weight'][0]; own['fc.bias'][0] = sd['fc.bias'][0]; m9.load_state_dict(own)
c8 = cer(predict(m8, cs8, [p for p, _ in KVA], DEV), [t for _, t in KVA]); print(f'9차 경판 검증 {c8*100:.2f}%', flush=True)

# ---------------- 6) 학습 (에폭마다 드라이브에 저장하고, 끊기면 이어서 한다) ----------------
EP = 8; NAH = min(len(AHtr), 25000); BS = 16; LR = 1.5e-4; REP_IN = 3
per = len(KTR) + REP_IN * len(INTR) + NAH + len(PS_tr); steps = EP * (per // BS)
opt = torch.optim.AdamW(m9.parameters(), lr=LR, weight_decay=1e-4); sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.05)
ctc = nn.CTCLoss(blank=0, zero_infinity=True)
PSV = random.Random(2).sample(PS_te, min(600, len(PS_te)))
def val_score(m, cs):
    k = cer(predict(m, cs, [p for p, _ in KVA], DEV), [t for _, t in KVA])
    q = cer(predict(m, cs, [p for p, _ in PSV], DEV), [t for _, t in PSV]) if PSV else 0.0
    return k, q
CK = ROOT + '/ckpt_v10.pt'; SAVE9 = '/content/gyeongpan_model_v10.pt'; ep0 = 1; best9 = None; log9 = []
if os.path.exists(CK):
    ck = torch.load(CK, map_location=DEV)
    if ck.get('chars') == cs9.chars and ck.get('per') == per:
        m9.load_state_dict(ck['state']); opt.load_state_dict(ck['opt']); sch.load_state_dict(ck['sch'])
        ep0 = ck['ep'] + 1; best9 = ck['best9']; log9 = ck['log']; print('이어서 학습:', ep0, '에폭부터', flush=True)
        if os.path.exists(ROOT + '/gyeongpan_model_v10_진행중.pt'): shutil.copy(ROOT + '/gyeongpan_model_v10_진행중.pt', SAVE9)
q8 = val_score(m8, cs8)[1]; print(f'9차 필사본 시험 행 {q8*100:.2f}%', flush=True)
for ep in range(ep0, EP + 1):
    tr = KTR + INTR * REP_IN + random.sample(AHtr, NAH) + PS_tr
    dl = torch.utils.data.DataLoader(LineDS9(tr, cs9, True), batch_size=BS, shuffle=True, num_workers=2, collate_fn=collate, drop_last=True)
    m9.train(); t0 = time.time(); tot = 0; n = 0
    for x, tg in dl:
        x = x.to(DEV); y = m9(x).log_softmax(-1).permute(1, 0, 2)
        loss = ctc(y, torch.tensor(sum(tg, [])), torch.full((len(tg),), y.shape[0], dtype=torch.long), torch.tensor([len(t) for t in tg]))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m9.parameters(), 5); opt.step()
        if sch.last_epoch + 1 < steps: sch.step()
        tot += loss.item(); n += 1
    k, q = val_score(m9, cs9); sc = k + q; log9.append((ep, tot / n, k, q))
    if best9 is None or sc < best9:
        best9 = sc; torch.save({'state': m9.state_dict(), 'chars': cs9.chars, 'H': H}, SAVE9); shutil.copy(SAVE9, ROOT + '/gyeongpan_model_v10_진행중.pt')
    torch.save({'state': m9.state_dict(), 'opt': opt.state_dict(), 'sch': sch.state_dict(), 'ep': ep, 'best9': best9, 'log': log9, 'chars': cs9.chars, 'per': per}, CK)
    print(f'epoch {ep:2d}  loss {tot/n:.3f}  경판 검증 {k*100:.2f}% (9차 {c8*100:.2f}%)  필사본 {q*100:.2f}% (9차 {q8*100:.2f}%)  {time.time()-t0:.0f}s', flush=True)
m9b, cs9b = load_model(SAVE9, DEV)
torch.save({'state': m9b.state_dict(), 'chars': cs9b.chars, 'H': H, 'arch': 'CRNN'}, ROOT + '/gyeongpan_model_v10.pt')

# ---------------- 7) 시험 ----------------
from rapidfuzz.distance import Levenshtein as Lv
R9 = {'val_v8': c8, 'pilsa_v8': q8}; R9['val_v9'], R9['pilsa_v9'] = val_score(m9b, cs9b)
sm = random.Random(1).sample(AHte, min(800, len(AHte)))
R9['simcheong_v8'] = cer(predict(m8, cs8, [p for p, _ in sm], DEV), [t for _, t in sm])
R9['simcheong_v9'] = cer(predict(m9b, cs9b, [p for p, _ in sm], DEV), [t for _, t in sm])
# 금향졍긔 권하: 면을 통째로 읽어 KOHICO 전사와 맞춤 (대량 판독과 같은 행 나누기)
os.environ['BANGAKBON_NO_MAIN'] = '1'
B = {'__name__': 'bulk'}; exec(get('bulk_ocr.py'), B)
KP = json.load(gzip.open(ROOT + '/private/kohico_pages.json.gz', 'rt', encoding='utf-8'))
CL = re.compile(r'[ᄀ-ᅟꥠ-꥿]+[ᅠ-ᆧힰ-ퟆ]+[ᆨ-ᇿퟋ-ퟻ]*|.', re.S)
toks = lambda s: CL.findall(unicodedata.normalize('NFC', s))
GP = [v for k, v in KP.items() if k.startswith('금향정기_2')][0]
GT = [g for g, _ in fix_rep([(t, 0) for k in sorted(GP) if k not in ('31ᄀ', '31ᄂ') for t in toks(re.sub(r'\s+', '', GP[k]))])[0]]
pdf = B['get_commons']('File:CNTS-00109637790 금향졍긔 이종. 권하.pdf', '/content/geumhyang2.pdf'); doc = B['fitz'].open(pdf)
def geum(path):
    R = B['Reader'](path); own = []; flat = []
    for i in range(len(doc)):
        no = i + 1
        if not (3 <= no <= 62 or no == 65): continue
        im = B['norm_width'](B['page_gray'](doc, i)); t, bt, cols = B['segment'](im)[:3]
        arrs = [B['prep_arr'](im.crop((max(0, a - 3), t, c + 3, bt))) for a, c in cols]
        for s, _ in R.read(arrs, bs=1):              # 한 행씩(⑰과 같은 조건)
            tk = toks(s); flat += tk; own += [no] * len(tk)
    err = collections.Counter(); n = collections.Counter(no >= 36 for no in own)
    for op in Lv.editops(flat, GT):
        p = min(op.src_pos, len(own) - 1); err[own[p] >= 36] += 1
    return err[True] / n[True]
R9['geum_v8'] = geum(ROOT + '/gyeongpan_model_v9.pt'); R9['geum_v9'] = geum(ROOT + '/gyeongpan_model_v10.pt')
print(f"\n[경판 검증]          9차 {R9['val_v8']*100:.2f}% → 10차 {R9['val_v9']*100:.2f}%")
print(f"[금향졍긔 뒤쪽 절반]  9차 {R9['geum_v8']*100:.2f}% → 10차 {R9['geum_v9']*100:.2f}%  (언어 모델 없이)")
print(f"[필사본 시험 행]      9차 {R9['pilsa_v8']*100:.2f}% → 10차 {R9['pilsa_v9']*100:.2f}%")
print(f"[AI허브 심청전_04]   9차 {R9['simcheong_v8']*100:.2f}% → 10차 {R9['simcheong_v9']*100:.2f}%")
json.dump({'result': R9, 'log': log9, 'n_kyeong': len(KTR), 'n_inbox': len(INTR), 'n_aihub': len(AHtr), 'n_pilsa': len(PS_tr)}, open(ROOT + '/result_v10.json', 'w'), ensure_ascii=False)

# ---------------- 8) 웹용 내보내기 (배치 축도 가변, 9MB 조각) ----------------
import onnxruntime as ort
OUTW = ROOT + '/web_v10'; os.makedirs(OUTW, exist_ok=True)
mm = m9b.cpu().eval(); kw = dict(input_names=['x'], output_names=['y'], dynamic_axes={'x': {0: 'b', 3: 'w'}, 'y': {0: 'b', 1: 't'}}, opset_version=17)
try: torch.onnx.export(mm, torch.zeros(2, 1, 64, 800), '/content/m10.onnx', dynamo=False, **kw)
except TypeError: torch.onnx.export(mm, torch.zeros(2, 1, 64, 800), '/content/m10.onnx', **kw)
se = ort.InferenceSession('/content/m10.onnx'); xt = np.random.rand(3, 1, 64, 517).astype(np.float32)
d1 = np.abs(se.run(None, {'x': xt})[0] - mm(torch.from_numpy(xt)).detach().numpy()).max(); print('ONNX 점검(배치 3) 차이', f'{d1:.2e}')
d = open('/content/m10.onnx', 'rb').read(); P = 9_000_000; n = 0
for f in glob.glob(OUTW + '/model_v10_*.wasm'): os.remove(f)
for i in range(0, len(d), P): open(f'{OUTW}/model_v10_{n}.wasm', 'wb').write(d[i:i + P]); n += 1
json.dump({'chars': cs9b.chars, 'H': 64, 'MAXW': 1600}, open(OUTW + '/charset.json', 'w'), ensure_ascii=False)
print('9차 웹용 조각', n, '개 저장 끝. 사이트에 올릴지는 위 시험 숫자를 보고 정한다.')
