#@title ⑥ AI허브 목판본 → 행 단위 학습 자료 (드라이브 방각본_판독기/aihub 의 tar 파일들을 씀)
import os, re, io, glob, json, tarfile, zipfile, unicodedata, collections, shutil, time
import numpy as np
from PIL import Image
SRC = '/content/drive/MyDrive/방각본_판독기/aihub'
WORK = '/content/aihub_work'; OUT = '/content/aihub_lines'
os.makedirs(WORK, exist_ok=True); os.makedirs(OUT, exist_ok=True)
dec = lambda s: re.sub(r'#U([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), s)

def unpack_tars():
    # 작게 나눠 올린 조각(이름.tar.part00, part01 …)은 먼저 합친다
    groups = collections.defaultdict(list)
    for p in glob.glob(SRC + '/*.tar.part*'):
        base, n = p.rsplit('.part', 1); groups[base].append((int(n), p))
    for base, ps in groups.items():
        dst = WORK + '/' + os.path.basename(base)
        if os.path.exists(dst + '.done'): continue
        with open(dst, 'wb') as w:
            for _, p in sorted(ps):
                with open(p, 'rb') as r: shutil.copyfileobj(r, w)
        print('조각 합침:', os.path.basename(base), len(ps), '조각', flush=True)
        with tarfile.open(dst) as tf: tf.extractall(WORK)
        open(dst + '.done', 'w').close(); os.remove(dst)
    for t in sorted(glob.glob(SRC + '/*.tar')):
        mark = WORK + '/' + os.path.basename(t) + '.done'
        if os.path.exists(mark): continue
        print('tar 풀기:', os.path.basename(t), f'{os.path.getsize(t)/1e9:.2f}GB', flush=True)
        with tarfile.open(t) as tf: tf.extractall(WORK)
        open(mark, 'w').close()
def merged_zips():
    parts = collections.defaultdict(list)
    for p in glob.glob(WORK + '/**/*.zip.part*', recursive=True):
        base, n = p.rsplit('.part', 1); parts[base].append((int(n), p))
    out = []
    for base, ps in parts.items():
        if not os.path.exists(base):
            print('합치기:', os.path.basename(base), len(ps), '조각', flush=True)
            with open(base, 'wb') as w:
                for _, p in sorted(ps):
                    with open(p, 'rb') as r: shutil.copyfileobj(r, w, 64 << 20)
                    os.remove(p)
        out.append(base)
    return out + [z for z in glob.glob(WORK + '/**/*.zip', recursive=True) if z not in out]

FILL = {'ᅟ', 'ᅠ'}
def clean(s):
    s = ''.join(ch for ch in unicodedata.normalize('NFD', s) if ch not in FILL)
    return unicodedata.normalize('NFC', s)
def runs_of(d):
    """한 면의 글자 상자를 행(열 번호)별로 모으고, 한자처럼 상자가 빠진 틈에서 끊어 연속 구간으로 나눈다."""
    cols = collections.defaultdict(list)
    for t in d['Text_Coord']:
        x, y, w, h, c, k = t['bbox'][:6]
        if w > 0 and h > 0: cols[c].append((k, x, y, w, h, t['annotate']))
    res = []
    for c, ts in cols.items():
        ts.sort()
        mh = float(np.median([t[4] for t in ts])); mw = float(np.median([t[3] for t in ts]))
        cur = [ts[0]]
        for a, b in zip(ts, ts[1:]):
            gap = b[2] - (a[2] + a[4]); shift = abs((b[1] + b[3] / 2) - (a[1] + a[3] / 2))
            if gap > 0.6 * mh or shift > 0.6 * mw or b[0] != a[0] + 1: res.append((c, cur)); cur = [b]
            else: cur.append(b)
        res.append((c, cur))
    return [(c, r) for c, r in res if len(r) >= 3]

def build():
    zs = merged_zips(); print('zip:', [os.path.basename(z) for z in zs])
    lab = {}
    for z in zs:
        if os.path.basename(z).startswith(('TL', 'VL')):
            with zipfile.ZipFile(z) as zf:
                for n in zf.namelist():
                    if n.endswith('.json'): lab[os.path.basename(dec(n))[:-5]] = (z, n)
    print('정답 파일', len(lab), flush=True)
    t0 = time.time(); nimg = 0
    labzips = {}
    for z in zs:
        b = os.path.basename(z)
        if not b.startswith(('TS', 'VS')): continue
        split = 'val' if b.startswith('VS') else 'train'
        part = f'{OUT}/lines_{b[:4]}.jsonl'
        if os.path.exists(part): print('이미 변환함:', b); continue
        rows = []
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                if not n.lower().endswith(('.png', '.jpg', '.jpeg')): continue
                key = os.path.splitext(os.path.basename(dec(n)))[0]
                if key not in lab: continue
                lz, ln = lab[key]
                if lz not in labzips: labzips[lz] = zipfile.ZipFile(lz)
                d = json.loads(labzips[lz].read(ln).decode('utf-8-sig'))
                im = Image.open(io.BytesIO(zf.read(n))).convert('L'); W, H = im.size
                sx = W / d['Image_Width'] if d.get('Image_Width') else 1.0; sy = H / d['Image_Height'] if d.get('Image_Height') else 1.0
                parts = dec(n).split('/'); cat = next((p for p in parts if re.match(r'\d\d\.', p)), ''); book = parts[-2]
                os.makedirs(f'{OUT}/{book}', exist_ok=True)
                for ri, (c, r) in enumerate(runs_of(d)):
                    mw = float(np.median([t[3] for t in r])); pad = 0.12 * mw
                    x0 = min(t[1] for t in r) - pad; x1 = max(t[1] + t[3] for t in r) + pad
                    y0 = min(t[2] for t in r) - pad; y1 = max(t[2] + t[4] for t in r) + pad
                    box = (max(0, int(x0 * sx)), max(0, int(y0 * sy)), min(W, int(x1 * sx)), min(H, int(y1 * sy)))
                    if box[2] - box[0] < 8 or box[3] - box[1] < 16: continue
                    cr = im.crop(box); s = min(1.0, 128 / cr.size[0])
                    if s < 1: cr = cr.resize((max(8, round(cr.size[0] * s)), max(8, round(cr.size[1] * s))), Image.LANCZOS)
                    fn = f'{book}/{key}_{c:02d}_{ri:02d}.png'; cr.save(f'{OUT}/{fn}')
                    rows.append({'img': fn, 'text': clean(''.join(t[5] for t in r)), 'book': book, 'cat': cat, 'split': split})
                nimg += 1
                if nimg % 500 == 0: print(f'면 {nimg}  행 {len(rows)}  {time.time()-t0:.0f}초', flush=True)
        with open(part, 'w', encoding='utf-8') as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(b, '끝: 행', len(rows), flush=True)
    allr = []
    for part in sorted(glob.glob(f'{OUT}/lines_*.jsonl')):
        allr += [json.loads(l) for l in open(part, encoding='utf-8')]
    with open(f'{OUT}/lines.jsonl', 'w', encoding='utf-8') as f:
        for r in allr: f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print('전체: 면', nimg, '(이번 실행)  행', len(allr))
    return allr

if __name__ == '__main__' and os.path.isdir(SRC):
    unpack_tars(); rows = build()
    shutil.make_archive('/content/aihub_lines', 'zip', OUT)
    shutil.copy('/content/aihub_lines.zip', SRC + '/aihub_mokpan_lines.zip')
    print('드라이브 저장:', SRC + '/aihub_mokpan_lines.zip', f'{os.path.getsize("/content/aihub_lines.zip")/1e9:.2f}GB')
