# 검증용 작품을 지금 사이트 모델(p1)로 통째로 읽어 정답과 맞춰 보는 스크립트(Colab에서). 학습은 하지 않는다.
# 쓰는 법
#   한 작품: python3 eval_book.py <원문> <정답.txt> <이름>
#            <원문>은 PDF 하나, 이미지 하나, 또는 폴더(안의 PDF·이미지를 이름 순서대로 모두 읽음)
#   한꺼번에: python3 eval_book.py --all [--redo] [--only 2,3,15]
#            드라이브 '8_검증용_작품/완성본' 안의 작품 폴더를 번호 순서대로 모두 읽는다.
#            작품 폴더 안의 PDF·이미지가 원문, 맨 위의 .txt(출처.txt 등 빼고)가 정답이다.
#            이미 결과가 있으면 건너뛴다(--redo면 다시 함).
# 사이트와 같은 행 나누기(seg.py), 같은 판심·광곽 줄 빼기, 같은 언어 모델 보정(빔 탐색)을 쓴다.
# 필사본은 펼친 면을 필사본 방식(split_spread_paper)으로 나누지만, 사이트의 필사본 행 찾기 모델은 쓰지 않으므로 참고용 수치다.
# 채점은 한글 글자(옛한글 포함)만 남겨 책 전체를 한 줄로 이어 맞춘다(면 나눔·띄어쓰기·부호·한자·반복 부호는 빼고 셈).
import os, sys, re, json, time, glob, collections, unicodedata, traceback, requests
RAWG = 'https://raw.githubusercontent.com/kangeunkim9/bangakbon-reader/main/'
ROOT = '/content/drive/MyDrive/방각본_판독기'
nfc = lambda s: unicodedata.normalize('NFC', s)
natkey = lambda s: [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', nfc(s))]
IMG = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')

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


# ---- 원문 면 목록: PDF 여러 개, 이미지 여러 장, 섞인 폴더 모두 ----
def pages(path):
    """(면 이름, 회색 이미지를 여는 함수) 목록. 폴더면 안의 파일을 이름 순서(숫자는 크기 순)대로."""
    if os.path.isdir(path):
        fs = sorted((os.path.join(d, f) for d, _, fn in os.walk(path) for f in fn), key=lambda p: natkey(os.path.relpath(p, path)))
        base = path
    else:
        fs, base = [path], os.path.dirname(path)
    out = []
    for f in fs:
        name, low = nfc(os.path.relpath(f, base)), f.lower()
        if low.endswith('.pdf'):
            doc = B['fitz'].open(f)
            out += [(f'{name} {i + 1}쪽', lambda doc=doc, i=i: B['page_gray'](doc, i)) for i in range(len(doc))]
        elif low.endswith(IMG):
            out.append((name, lambda f=f: Image.open(f).convert('L')))
    return out


# ---- 정답 읽기 ----
HG = re.compile(r'[ᄀ-ᅟꥠ-꥿]+[ᅠ-ᆧힰ-ퟆ]+[ᆨ-ᇿퟋ-ퟻ]*|[가-힣]')
toks = lambda s: HG.findall(nfc(s))
def read_gt(files):
    out = []
    for gf in files:
        b = open(gf, 'rb').read()
        if b[:2] in (b'\xff\xfe', b'\xfe\xff'): gt = b.decode('utf-16', errors='ignore')
        elif b[:3] == b'\xef\xbb\xbf': gt = b[3:].decode('utf-8', errors='ignore')
        else:
            try: gt = b.decode('utf-8')
            except UnicodeDecodeError: gt = b.decode('cp949', errors='ignore')
        if re.search('[-]', gt):
            try: import hypua2jamo
            except ImportError:
                import subprocess; subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'hypua2jamo'], check=False); import hypua2jamo
            gt = hypua2jamo.translate(gt, composed=True)
        gt = re.sub(r'<(title|name|edition|indo|note|date|extent|head|author|publisher)\b[^>]*>.*?</\1>', ' ', gt, flags=re.S)
        gt = re.sub(r'<teiHeader>.*?</teiHeader>', ' ', gt, flags=re.S)
        out.append(re.sub(r'<[^>]*>', ' ', gt))
    return '\n'.join(out)


# ---- 한 권 읽기 ----
junk = lambda s, c: c < 0.6 or len(B['G'](s)) <= 2 or (len(B['G'](s)) <= 3 and c < 0.9)
def read_book(path, pilsa=False):
    split = B['split_spread_paper'] if pilsa else B['split_spread']
    P = pages(path); LPS, FACE, FC, t0 = [], [], [], time.time()   # FC: 면마다 (이름, 상태, 나눈 행) - line_check.py가 읽는다
    for n, (pname, opener) in enumerate(P):
        try: g = opener()
        except Exception as e:
            print('  못 읽은 면:', pname, e, flush=True); continue
        for im, side, _ in split(g):
            im = B['norm_width'](im)
            try: t, bt, cols = B['segment'](im)[:3]
            except Exception: cols = []
            lab = pname + (' ' + side if side else '')
            if len(cols) < 6:
                FC.append({'면': lab, '상태': 'empty', '행': []}); continue
            lps = []
            with torch.no_grad():
                for a, c in cols:
                    pr = svm(torch.from_numpy(prep_s(im.crop((max(0, a - 3), t, c + 3, bt))))[None].to(DEV)).float()[0]
                    lps.append(torch.log(pr.clamp_min(1e-9)).cpu().numpy())
            txt = [greedy(lp) for lp in lps]
            conf = []
            for lp in lps:
                k = lp.argmax(-1); m = lp.max(-1); conf.append(float(np.exp(m[k > 0]).mean()) if (k > 0).any() else 0.0)
            nn, rm = len(txt), set()
            for k in range(min(2, nn)):
                if not junk(txt[k], conf[k]): break
                rm.add(k)
            for k in range(nn - 1, max(nn - 3, -1), -1):
                if k in rm or not junk(txt[k], conf[k]): break
                rm.add(k)
            keep = [k for k in range(nn) if k not in rm]
            lens = [len(B['G'](txt[k])) for k in keep]
            if not keep or all(x < 3 for x in lens[:6]) or float(np.median(lens)) < 8 or np.mean(conf) < 0.55:   # 표지·빈 면
                FC.append({'면': lab, '상태': 'nomatch', '행': [list(map(int, c)) for c in cols]}); continue
            FC.append({'면': lab, '상태': 'ok', '행': [list(map(int, cols[k])) for k in keep]})
            for k in keep: LPS.append(lps[k]); FACE.append(lab)
        if (n + 1) % 10 == 0: print(f'  {n + 1}/{len(P)}면 읽음 ({time.time() - t0:.0f}초)', flush=True)
    return P, LPS, FACE, FC


def score(texts, GT):
    Pt = [x for s in texts for x in toks(s)]
    ops = Lv.editops(Pt, GT); c = collections.Counter()
    for op in ops:
        if op.tag == 'replace': c[(Pt[op.src_pos], GT[op.dest_pos])] += 1
        elif op.tag == 'delete': c[(Pt[op.src_pos], '∅')] += 1
        else: c[('∅', GT[op.dest_pos])] += 1
    kinds = collections.Counter(op.tag for op in ops)
    return len(ops) / max(1, len(GT)) * 100, len(Pt), dict(kinds), c.most_common(25)


def eval_one(path, gt_files, name, pilsa=False, outdir=None):
    outdir = outdir or ROOT + '/기록'
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    P, LPS, FACE, FC = read_book(path, pilsa)
    print(f'[{name}] 원문 {len(P)}면 중 글자 있는 면 {len(set(FACE))}, 행 {len(LPS)}', flush=True)
    gt = read_gt(gt_files); GT = toks(gt)
    print(f'[{name}] 정답 글자 {len(GT)} / 정답에 반복 부호(〃) {gt.count("〃")}', flush=True)
    TG = [greedy(lp) for lp in LPS]; TB = [beam(lp) for lp in LPS]
    RES = {'이름': name, '원문': nfc(path), '정답 파일': [nfc(os.path.basename(f)) for f in gt_files], '필사본 방식': pilsa,
           '원문 면': len(P), '읽은 면': len(set(FACE)), '행': len(LPS), '걸린 초': 0}
    for nm, T in (('보정 없음', TG), ('언어 모델 보정', TB)):
        e, n, kinds, top = score(T, GT)
        RES[nm] = {'CER': round(e, 2), '판독 글자': n, '정답 글자': len(GT), '종류': kinds, '자주 틀린 것': [[a, b, k] for (a, b), k in top]}
        print(f'[{name}] {nm}: 글자 오류율 {e:.2f}%  (판독 {n}자 / 정답 {len(GT)}자, 바꿈 {kinds.get("replace", 0)} 뺌 {kinds.get("delete", 0)} 빠뜨림 {kinds.get("insert", 0)})')
        print('  자주 틀린 것(판독→정답):', ', '.join(f'{a}→{b} {k}' for (a, b), k in top[:15]), flush=True)
    RES['걸린 초'] = round(time.time() - t0); RES['면별 행'] = FC
    safe = re.sub(r'[\\/:*?"<>|]', '_', name)
    json.dump(RES, open(f'{outdir}/{safe}.json', 'w'), ensure_ascii=False, indent=1)
    with open(f'{outdir}/{safe}_판독문.txt', 'w', encoding='utf-8') as f:
        last = None
        for lab, s in zip(FACE, TB):
            if lab != last: f.write(f'\n## {lab}\n'); last = lab
            f.write(s + '\n')
    return RES


# ---- 완성본 한꺼번에 ----
def kind_of(folder):
    """폴더 이름 '작품명(장수)_판종_소장처'에서 판본 갈래를 뽑는다: 경판·완판·안성판·방각본(그 밖)·필사본·활자본·판종미상."""
    nm = nfc(folder)
    parts = nm.split('_')
    pz = parts[1] if len(parts) > 1 else ''
    for k in ('경판', '완판', '안성판'):
        if k in nm: return k
    for k in ('필사본', '활자본', '방각본', '판종미상'):
        if k in pz: return k
    return pz or '모름'

def run_all(redo=False, only=None):
    V8 = [p for p in glob.glob(ROOT + '/*') if nfc(os.path.basename(p)).startswith('8_')][0]
    done = [p for p in glob.glob(V8 + '/*') if nfc(os.path.basename(p)) == '완성본'][0]
    works = sorted((p for p in glob.glob(done + '/*') if os.path.isdir(p) and re.match(r'\d+\.', nfc(os.path.basename(p)))),
                   key=lambda p: int(nfc(os.path.basename(p)).split('.')[0]))
    outdir = ROOT + '/기록/검증결과'; os.makedirs(outdir, exist_ok=True)
    rows = []
    for w in works:
        fname = nfc(os.path.basename(w)); num = int(fname.split('.')[0])
        if only and num not in only: continue
        name = re.sub(r'[\\/:*?"<>|]', '_', fname)
        if not redo and os.path.exists(f'{outdir}/{name}.json'):
            print('이미 있음, 건너뜀:', fname); rows.append(json.load(open(f'{outdir}/{name}.json'))); continue
        gts = sorted((os.path.join(w, f) for f in os.listdir(w) if f.lower().endswith('.txt') and not re.search('출처|넣는|메모|^_', nfc(f))), key=natkey)
        srcs = [f for d, _, fn in os.walk(w) for f in fn if f.lower().endswith(IMG + ('.pdf',))]
        if not gts or not srcs:
            print('건너뜀(정답 또는 원문 없음):', fname, len(gts), len(srcs)); continue
        print('\n=====', fname, f'(정답 {len(gts)}개, 원문 파일 {len(srcs)}개)', flush=True)
        try:
            r = eval_one(w, gts, name, pilsa=kind_of(fname.split('. ', 1)[-1]) == '필사본', outdir=outdir)
            rows.append(r)
        except Exception:
            traceback.print_exc()
    # 작품별 한 줄 요약
    with open(ROOT + '/기록/검증_성적표.tsv', 'w', encoding='utf-8') as f:
        f.write('번호\t작품 폴더\t판본 갈래\t원문 면\t읽은 면\t행\t정답 글자\t판독 글자\t오류율(보정 없음)\t오류율(언어 모델)\n')
        for r in sorted(rows, key=lambda r: int(r['이름'].split('.')[0])):
            a, b = r['보정 없음'], r['언어 모델 보정']
            f.write('\t'.join(map(str, [r['이름'].split('.')[0], r['이름'], kind_of(r['이름'].split('. ', 1)[-1]), r['원문 면'], r['읽은 면'], r['행'],
                                        a['정답 글자'], a['판독 글자'], a['CER'], b['CER']])) + '\n')
    print('\n작품별 결과:', outdir, '/ 요약:', ROOT + '/기록/검증_성적표.tsv')
    return rows


if __name__ == '__main__':
    A = sys.argv[1:]
    if A and A[0] == '--all':
        only = None
        if '--only' in A: only = {int(x) for x in A[A.index('--only') + 1].split(',')}
        run_all(redo='--redo' in A, only=only)
    else:
        PDF, GTF, NAME = A[0], A[1], A[2]
        eval_one(PDF, [GTF], NAME, pilsa='--pilsa' in A)
