# 검증용 작품 한 권을 지금 사이트 모델(p1)로 통째로 읽어 정답과 맞춰 보는 스크립트(Colab 터미널에서 python3로). 학습은 하지 않는다.
# 쓰는 법: python3 eval_book.py <원문.pdf> <정답.txt> <이름>
# 사이트와 같은 행 나누기(seg.py), 같은 판심·광곽 줄 빼기, 같은 언어 모델 보정(빔 탐색)을 쓴다.
# 채점은 한글 글자(옛한글 포함)만 남겨 책 전체를 한 줄로 이어 맞춘다(면 나눔·띄어쓰기·부호·한자·반복 부호는 빼고 셈).
import os, sys, re, json, time, collections, unicodedata, requests
PDF, GTF, NAME = sys.argv[1], sys.argv[2], sys.argv[3]
RAWG = 'https://raw.githubusercontent.com/kangeunkim9/bangakbon-reader/main/'
ROOT = '/content/drive/MyDrive/방각본_판독기'
src = requests.get(RAWG + 'training/eval_lm_svtrv2.py', timeout=60).text
G = {'__name__': 'evb', 'SV_MODEL': ROOT + '/1_모델/svtrv2_model_p1.pt'}
exec(src[:src.index('# ---------------- 1)')].replace("from google.colab import drive; drive.mount('/content/drive')", 'pass'), G)
torch, np, F, Image, ImageOps, Lv = G['torch'], G['np'], G['F'], G['Image'], G['ImageOps'], G['Lv']
DEV = G['DEV']
os.environ['BANGAKBON_NO_MAIN'] = '1'
B = {'__name__': 'bulk'}; exec(requests.get(RAWG + 'training/bulk_ocr.py', timeout=60).text, B)

# ---- 모델(p1) ----
s2 = src[src.index('import openrec.modeling.encoders'):src.index('with torch.no_grad():\n    for _, crops in FACES:\n        for c in crops:\n            pr = svm')]
G['CH'] = None; exec(s2.replace("assert list(ck['chars']) == CH, '글자표가 10차와 다릅니다'\n", ''), G)
svm, prep_s = G['svm'], G['prep_s']; CH = list(G['ck']['chars']); G['CH'] = CH
exec(src[src.index('# ---------------- 3)'):src.index('# ---------------- 4)')], G)
greedy, beam = G['greedy'], G['beam']

# ---- 읽기: 사이트처럼 펼친 면 나누고, 행 나누고, 판심·광곽 줄 빼기 ----
doc = B['fitz'].open(PDF); LPS, FACE, t0 = [], [], time.time()
junk = lambda s, c: c < 0.6 or len(B['G'](s)) <= 2 or (len(B['G'](s)) <= 3 and c < 0.9)
for i in range(len(doc)):
    g = B['page_gray'](doc, i)
    for im, side, _ in B['split_spread'](g):
        im = B['norm_width'](im)
        try: t, bt, cols = B['segment'](im)[:3]
        except Exception: cols = []
        if len(cols) < 6: continue
        lps = []
        with torch.no_grad():
            for a, c in cols:
                pr = svm(torch.from_numpy(prep_s(im.crop((max(0, a - 3), t, c + 3, bt)))[None]).to(DEV)).float()[0]
                lps.append(torch.log(pr.clamp_min(1e-9)).cpu().numpy())
        txt = [greedy(lp) for lp in lps]
        conf = []
        for lp in lps:
            k = lp.argmax(-1); m = lp.max(-1); conf.append(float(np.exp(m[k > 0]).mean()) if (k > 0).any() else 0.0)
        n = len(txt); rm = set()
        for k in range(min(2, n)):
            if not junk(txt[k], conf[k]): break
            rm.add(k)
        for k in range(n - 1, max(n - 3, -1), -1):
            if k in rm or not junk(txt[k], conf[k]): break
            rm.add(k)
        keep = [k for k in range(n) if k not in rm]
        lens = [len(B['G'](txt[k])) for k in keep]
        if not keep or all(x < 3 for x in lens[:6]) or float(np.median(lens)) < 8 or np.mean(conf) < 0.55: continue   # 표지·빈 면
        lab = f'{i + 1}쪽' + (' ' + side if side else '')
        for k in keep: LPS.append(lps[k]); FACE.append(lab)
    if (i + 1) % 10 == 0: print(f'  {i + 1}/{len(doc)}쪽 읽음 ({time.time() - t0:.0f}초)', flush=True)
print('읽은 면', len(set(FACE)), '행', len(LPS), flush=True)
TG = [greedy(lp) for lp in LPS]
TB = [beam(lp) for lp in LPS]

# ---- 정답 ----
b = open(GTF, 'rb').read()
if b[:2] in (b'\xff\xfe', b'\xfe\xff'): gt = b.decode('utf-16', errors='ignore')
elif b[:3] == b'\xef\xbb\xbf': gt = b[3:].decode('utf-8', errors='ignore')
else:
    try: gt = b.decode('utf-8')
    except UnicodeDecodeError: gt = b.decode('cp949', errors='ignore')
if re.search('[-]', gt):
    subprocess = __import__('subprocess'); subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'hypua2jamo'], check=False)
    import hypua2jamo; gt = hypua2jamo.translate(gt, composed=True)
gt = re.sub(r'<(title|name|edition|indo|note|date|extent|head|author|publisher)\b[^>]*>.*?</\1>', ' ', gt, flags=re.S)
gt = re.sub(r'<[^>]*>', ' ', gt)
HG = re.compile(r'[ᄀ-ᅟꥠ-꥿]+[ᅠ-ᆧힰ-ퟆ]+[ᆨ-ᇿퟋ-ퟻ]*|[가-힣]')
toks = lambda s: HG.findall(unicodedata.normalize('NFC', s))
GT = toks(gt)
print('정답 글자', len(GT), '/ 정답에 반복 부호(〃) 수', gt.count('〃'), flush=True)

def score(texts):
    P = [x for s in texts for x in toks(s)]
    ops = Lv.editops(P, GT); c = collections.Counter()
    for op in ops:
        if op.tag == 'replace': c[(P[op.src_pos], GT[op.dest_pos])] += 1
        elif op.tag == 'delete': c[(P[op.src_pos], '∅')] += 1
        else: c[('∅', GT[op.dest_pos])] += 1
    kinds = collections.Counter(op.tag for op in ops)
    return len(ops) / len(GT) * 100, len(P), dict(kinds), c.most_common(25)
RES = {}
for nm, T in (('보정 없음', TG), ('언어 모델 보정(사이트 기본)', TB)):
    e, n, kinds, top = score(T)
    RES[nm] = {'CER': round(e, 2), '판독 글자': n, '정답 글자': len(GT), '종류': kinds}
    print(f'\n[{NAME}] {nm}: 글자 오류율 {e:.2f}%  (판독 {n}자 / 정답 {len(GT)}자, 바꿈 {kinds.get("replace", 0)} 뺌 {kinds.get("delete", 0)} 빠뜨림 {kinds.get("insert", 0)})')
    print('  자주 틀린 것(판독→정답):', ', '.join(f'{a}→{b} {k}' for (a, b), k in top))
    RES[nm]['자주 틀린 것'] = [[a, b, k] for (a, b), k in top]
os.makedirs(ROOT + '/기록', exist_ok=True)
json.dump(RES, open(ROOT + f'/기록/result_book_{NAME}.json', 'w'), ensure_ascii=False, indent=1)
with open(f'/content/read_{NAME}.txt', 'w', encoding='utf-8') as f:
    last = None
    for lab, s in zip(FACE, TB):
        if lab != last: f.write(f'\n## {lab}\n'); last = lab
        f.write(s + '\n')
print('결과:', ROOT + f'/기록/result_book_{NAME}.json', '/ 판독문:', f'/content/read_{NAME}.txt')
