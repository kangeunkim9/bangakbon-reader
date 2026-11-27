# SVTRv2에 AI허브 필사본 전체(약 32만 행)를 더 가르치는 셀(Colab).
#@title ㊱ SVTRv2 + AI허브 필사본 전체: 정답 점검으로 걸러 낸 행을 넣어 이어 학습 (사이트 모델 svtrv2_model_v2에서 출발)
# 1) 10차·SVTRv2와 같은 학습 자료를 부른다(경판, 학습용 보내기, AI허브 목판본, AI허브 필사본 작은 묶음)
# 2) 필사본 전체 묶음(aihub_pilsa_lines_all.zip)을 풀고, 정답 표기를 맞춘다(옛이응 ᅌ → ᄋ: 일부 책만 ᅌ로 적어 둠)
# 3) 지금 SVTRv2로 모든 행을 읽어, 정답과 너무 다른 행(정답 잘못·행 자르기 잘못일 가능성)을 뺀다
# 4) 에폭마다 필사본을 6만 행씩 새로 뽑아 경판 자료와 섞어 배운다
# 5) 시험: 경판 검증, 필사본(유이양문록·명행정의록8), 언간(경주최씨댁한글편지, 흘림체), 심청전, 금향졍긔
# 시험용 책은 학습에서 뺀다. 결과는 드라이브 svtrv2_model_p1.pt, result_svtrv2_p1.json
import os, re, json, random, zipfile, time
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')   # 메모리 조각남 줄이기(런타임 새로 시작했을 때만 먹힘)
SV_TAG = 'p1'; SV_EP = globals().get('SV_EP', 6)
SV_INIT = '/content/drive/MyDrive/방각본_판독기/1_모델/svtrv2_model_v2.pt'
import requests
RAW = 'https://raw.githubusercontent.com/kangeunkim9/bangakbon-reader/main/training/'
_src = requests.get(RAW + 'train_svtrv2.py', timeout=60).text
exec(_src[:_src.index('\n# ---------------- 4) 학습')], globals())        # 1)~3): 자료, 모델(v2에서 출발), 전처리

# ---------------- 2) 필사본 전체 묶음 ----------------
PD = '/content/pilsa_lines_all'
if not os.path.exists(PD + '/lines.jsonl'):
    os.makedirs(PD, exist_ok=True); t0 = time.time()
    with zipfile.ZipFile(ROOT + '/3_AI허브_원본/aihub_pilsa/aihub_pilsa_lines_all.zip') as z: z.extractall(PD)
    print('필사본 전체 풀기', int(time.time() - t0), '초', flush=True)
fixt = lambda s: nfc(nfd(s).replace('ᅌ', 'ᄋ'))      # 옛이응 → 이응
HOLD = re.compile(r'유이양문록|명행정의록8_|경주최씨댁한글편지')
ALL, EG_te, drop_ch = [], [], 0
for l in open(PD + '/lines.jsonl', encoding='utf-8'):
    r = json.loads(l); pth = PD + '/' + r['img']; t = fixt(r['text'])
    if len(G(t)) < 2 or not os.path.exists(pth): continue
    if any(c not in IDX for c in nfd(t)): drop_ch += 1; continue
    if '경주최씨댁한글편지' in r['book']: EG_te.append((pth, t)); continue
    if HOLD.search(r['book']): continue
    ALL.append((pth, t, r['book']))
print('필사본 전체 학습 후보', len(ALL), '행 / 글자표 밖 글자로 뺀 행', drop_ch, '/ 언간 시험(경주최씨댁)', len(EG_te), '행', flush=True)
PS_te = [(p, fixt(t)) for p, t in PS_te]; PSV = random.Random(2).sample(PS_te, min(600, len(PS_te)))
EGV = random.Random(3).sample(EG_te, min(600, len(EG_te)))

# ---------------- 3) 정답 점검: 지금 모델과 너무 다른 행 빼기 ----------------
FILT = ROOT + '/2_학습용_행자료/pilsa_all_filter_v2.json'
if os.path.exists(FILT):
    keep = set(json.load(open(FILT)))
else:
    # 중간에 끊겨도 이어서 하도록 4천 행쯤마다 드라이브에 적어 둔다
    PART = FILT + '.part'; st = json.load(open(PART)) if os.path.exists(PART) else {'n': len(ALL), 'done': 0, 'keep': []}
    if st['n'] != len(ALL): st = {'n': len(ALL), 'done': 0, 'keep': []}
    todo = list(range(st['done'], len(ALL))); keep = st['keep']
    if st['done']: print('정답 점검 이어서:', st['done'], '행부터', flush=True)
    class _R(torch.utils.data.Dataset):
        def __len__(s): return len(todo)
        def __getitem__(s, j): return prep_s(ALL[todo[j]][0]), todo[j]
    FBS = 64 if torch.cuda.get_device_properties(0).total_memory > 20e9 else 16; gc.collect(); torch.cuda.empty_cache()
    dl = torch.utils.data.DataLoader(_R(), batch_size=FBS, num_workers=4, collate_fn=lambda b: (batch_s([a for a, _ in b]), [i for _, i in b]))
    svm.eval(); t0 = time.time(); bad = collections.Counter(); seen = collections.Counter()
    with torch.no_grad():
        for k, (x, ii) in enumerate(dl):
            with torch.autocast('cuda', dtype=AMP):
                pr = greedy(svm(x.to(DEV)).float())
            for p, i in zip(pr, ii):
                g = graphemes(ALL[i][1]); e = lev(graphemes(p), g) / max(1, len(g)); seen[ALL[i][2]] += 1
                if e <= 0.35: keep.append(i)
                else: bad[ALL[i][2]] += 1
            if (k + 1) % max(1, 4096 // FBS) == 0:
                json.dump({'n': len(ALL), 'done': ii[-1] + 1, 'keep': keep}, open(PART, 'w'))
            if k % 2000 == 0: print(f'  점검 {ii[-1]+1}/{len(ALL)}행 {time.time()-t0:.0f}s', flush=True)
    json.dump(keep, open(FILT, 'w'))
    if os.path.exists(PART): os.remove(PART)
    print('정답 점검으로 뺀 비율이 높은 책:', sorted(((bad[b] / seen[b], seen[b], b) for b in seen if seen[b] >= 50), reverse=True)[:10], flush=True)
# 8_검증용_작품에 들어 있는 작품은 학습에서 뺀다. 폴더 이름이 '작품명_소장처'든 '작품명_판종_소장처'든 맨 앞을 작품명으로 읽는다.
# (정답 점검 결과 keep은 ALL의 순서에 묶여 있으므로 ALL은 그대로 두고 여기서 거른다)
def _val_works():
    vd = [os.path.join(ROOT, d) for d in os.listdir(ROOT) if nfc(d).startswith('8_')]
    w = set()
    for dp, dn, fn in (os.walk(vd[0]) if vd else []):
        if nfc(os.path.basename(dp)) == '원문': continue
        for d in dn:
            if '_' in d: w.add(re.sub(r'\(.*?\)', '', nfc(d).split('_')[0]).strip())
    return w
def _book_work(b):
    m = re.match(r'(?:필사본|목판본)_\d+_[^_]+_(.+?)(?:_\d+)?$', nfc(b))
    return re.sub(r'\d+$', '', m.group(1) if m else nfc(b))
VWORKS = _val_works()
EXV = {i for i in keep if _book_work(ALL[i][2]) in VWORKS}
print('검증용 작품과 같은 작품이라 학습에서 뺀 행', len(EXV), sorted({_book_work(ALL[i][2]) for i in EXV}), flush=True)
PALL = [(ALL[i][0], ALL[i][1]) for i in sorted(keep) if i not in EXV]
print('정답 점검 통과', len(PALL), '/', len(ALL), '행', flush=True)

def val_s(m):
    return (cer(read_s(m, [p for p, _ in KVA]), [t for _, t in KVA]), cer(read_s(m, [p for p, _ in PSV]), [t for _, t in PSV]),
            cer(read_s(m, [p for p, _ in EGV]), [t for _, t in EGV]))
v0 = val_s(svm); print(f'출발점(v2)  경판 검증 {v0[0]*100:.2f}%  필사본 {v0[1]*100:.2f}%  언간 {v0[2]*100:.2f}%', flush=True)

# ---------------- 4) 학습 (에폭마다 드라이브에 저장, 끊기면 이어서) ----------------
EP = SV_EP; NAH = min(len(AHtr), 25000); NPS = min(len(PALL), 60000); REP_IN = 3
BS = 24 if torch.cuda.get_device_properties(0).total_memory > 20e9 else 12   # T4(16GB)면 12
gc.collect(); torch.cuda.empty_cache(); n_oom = 0
per = len(KTR) + REP_IN * len(INTR) + NAH + NPS; steps = EP * (per // BS)
opt = torch.optim.AdamW(svm.parameters(), lr=1.5e-4, weight_decay=0.05)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1.5e-4, total_steps=steps, pct_start=0.05)
ctc = nn.CTCLoss(blank=0, zero_infinity=True)
CKP = ROOT + f'/1_모델/ckpt_svtrv2_{SV_TAG}.pt'; SAVES = ROOT + f'/1_모델/svtrv2_model_{SV_TAG}.pt'; ep0 = 1; best = None; logs = []
if os.path.exists(CKP):
    c0 = torch.load(CKP, map_location=DEV, weights_only=False)
    if c0.get('per') == per and c0.get('chars') == CH:
        svm.load_state_dict(c0['state']); opt.load_state_dict(c0['opt']); sch.load_state_dict(c0['sch'])
        ep0 = c0['ep'] + 1; best = c0['best']; logs = c0['log']; print('이어서 학습:', ep0, '에폭부터', flush=True)
print(f'한 에폭 {per}행(필사본 {NPS}), {per // BS}걸음 x {EP}에폭', flush=True)
for ep in range(ep0, EP + 1):
    tr = KTR + INTR * REP_IN + random.sample(AHtr, NAH) + random.sample(PALL, NPS)
    dl = torch.utils.data.DataLoader(DSS(tr, True), batch_size=BS, shuffle=True, num_workers=4, collate_fn=cols, drop_last=True)
    svm.train(); t0 = time.time(); tot = 0; n = 0
    for x, tg in dl:
        x = x.to(DEV)
        try:
            with torch.autocast('cuda', dtype=AMP):
                y = svm(x)
            lp = y.float().log_softmax(-1).permute(1, 0, 2)
            tl = torch.tensor([len(t) for t in tg])
            loss = ctc(lp, torch.tensor(sum(tg, [])), torch.full((len(tg),), lp.shape[0], dtype=torch.long), tl)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(svm.parameters(), 5); scaler.step(opt); scaler.update()
        except torch.OutOfMemoryError:   # 아주 긴 행이 몰린 묶음은 건너뛴다
            del x; opt.zero_grad(set_to_none=True); y = lp = loss = None; gc.collect(); torch.cuda.empty_cache(); n_oom += 1
            if n_oom % 10 == 1: print('  메모리 부족으로 건너뛴 묶음', n_oom, flush=True)
            continue
        if sch.last_epoch + 1 < steps: sch.step()
        tot += loss.item(); n += 1
        if n % 1000 == 0: print(f'  {ep}에폭 {n}걸음 loss {tot/n:.3f} {time.time()-t0:.0f}s', flush=True)
    k, q, e = val_s(svm); sc = k + q + e; logs.append((ep, tot / n, k, q, e))
    if best is None or sc < best:
        best = sc; torch.save({'state': svm.state_dict(), 'arch': arch, 'chars': CH, 'H': SH}, SAVES)
    torch.save({'state': svm.state_dict(), 'opt': opt.state_dict(), 'sch': sch.state_dict(), 'ep': ep, 'best': best, 'log': logs, 'chars': CH, 'per': per}, CKP)
    print(f'epoch {ep}  loss {tot/n:.3f}  경판 검증 {k*100:.2f}%  필사본 {q*100:.2f}%  언간 {e*100:.2f}%  {time.time()-t0:.0f}s', flush=True)

# ---------------- 5) 시험: v2(지금 사이트)와 나란히 ----------------
svm.load_state_dict(torch.load(SAVES, map_location=DEV, weights_only=False)['state'])
k, q, e = val_s(svm)
from rapidfuzz.distance import Levenshtein as Lv
os.environ['BANGAKBON_NO_MAIN'] = '1'
B = {'__name__': 'bulk'}; exec(requests.get(RAW + 'bulk_ocr.py', timeout=60).text, B)
KPG = json.load(gzip.open(ROOT + '/4_말뭉치_비공개/kohico_pages.json.gz', 'rt', encoding='utf-8'))
CL = re.compile(r'[ᄀ-ᅟꥠ-꥿]+[ᅠ-ᆧힰ-ퟆ]+[ᆨ-ᇿퟋ-ퟻ]*|.', re.S)
toks = lambda s: CL.findall(unicodedata.normalize('NFC', s))
GP = [v for kk, v in KPG.items() if kk.startswith('금향정기_2')][0]
GT = [g for g, _ in fix_rep([(t, 0) for kk in sorted(GP) if kk not in ('31ᄀ', '31ᄂ') for t in toks(re.sub(r'\s+', '', GP[kk]))])[0]]
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
sm = random.Random(1).sample(AHte, min(800, len(AHte)))
R = {'v2': {'경판 검증': v0[0], '필사본': v0[1], '언간': v0[2]}, 'p1': {'경판 검증': k, '필사본': q, '언간': e}}
R['p1']['심청전'] = cer(read_s(svm, [p for p, _ in sm]), [t for _, t in sm]); R['p1']['금향'] = geum(lambda crops: read_s(svm, crops))
svm.load_state_dict(torch.load(SV_INIT, map_location=DEV, weights_only=False)['state'])
R['v2']['심청전'] = cer(read_s(svm, [p for p, _ in sm]), [t for _, t in sm]); R['v2']['금향'] = geum(lambda crops: read_s(svm, crops))
print('\n[언어 모델 없이 글자 오류율]   지금 사이트(v2)   필사본 더 배움(p1)')
for kk in ('경판 검증', '금향', '심청전', '필사본', '언간'): print(f'  {kk:8s} {R["v2"][kk]*100:8.2f}%  {R["p1"][kk]*100:8.2f}%')
json.dump({'result': R, 'log': logs, 'n_pilsa': len(PALL), 'n_pilsa_all': len(ALL)}, open(ROOT + f'/기록/result_svtrv2_{SV_TAG}.json', 'w'), ensure_ascii=False)
