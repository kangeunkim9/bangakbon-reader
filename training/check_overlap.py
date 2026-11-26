# 검증용 작품(8_검증용_작품)의 정답 전사문이 학습 자료와 겹치는지 문장 단위로 확인한다(Colab 터미널에서 python3로).
# 학습 코드와는 따로 도는 확인용이다. 검증 폴더의 글은 여기서만 읽고 학습에는 쓰지 않는다.
# 방법: 띄어쓰기·문장부호를 뺀 글자열을 12글자씩 잘라(12-gram), 검증 작품의 12-gram 가운데
#       학습 자료의 한 책에도 들어 있는 비율을 책마다 센다. 같은 판본이면 대개 50%를 넘고, 다른 판본이면 몇 % 안팎이다.
# 결과: 화면 출력과 드라이브 방각본_판독기/검증_학습_겹침.tsv
import os, re, json, glob, zipfile, unicodedata, collections, zlib, sys
ROOT = '/content/drive/MyDrive/방각본_판독기'
N = 12
nfc = lambda s: unicodedata.normalize('NFC', s)
KEEP = re.compile(r'[^\w〃□]')
def norm(s): return KEEP.sub('', re.sub(r'\s+', '', nfc(s)))
def grams(s):
    s = norm(s)
    return {zlib.crc32(s[i:i + N].encode('utf-8')) for i in range(len(s) - N + 1)}

# ---------------- 1) 검증용 작품 정답 ----------------
VD = [os.path.join(ROOT, d) for d in os.listdir(ROOT) if nfc(d).startswith('8_')][0]
VAL = {}
for dp, dn, fn in os.walk(VD):
    for f in fn:
        if not f.lower().endswith('.txt') or nfc(f).startswith('_'): continue
        p = os.path.join(dp, f)
        try: t = open(p, encoding='utf-8').read()
        except UnicodeDecodeError: t = open(p, encoding='cp949', errors='ignore').read()
        work = nfc(os.path.basename(dp)) if nfc(os.path.basename(dp)) != '원문' else nfc(os.path.basename(os.path.dirname(dp)))
        key = work + ' / ' + nfc(f)
        g = grams(t)
        if g: VAL[key] = g
print('검증 정답 파일', len(VAL), '개', flush=True)
IDX = collections.defaultdict(list)
for k, g in VAL.items():
    for h in g: IDX[h].append(k)
HIT = collections.defaultdict(set)          # (검증 파일, 학습 책) -> 겹친 12-gram

def feed(book, text):
    for h in grams(text):
        for k in IDX.get(h, ()): HIT[(k, book)].add(h)

# ---------------- 2) 학습 자료 ----------------
SRC = collections.Counter()
def zlines(zp, tag):                         # AI허브 행 자료(lines.jsonl, 책 이름 포함)
    if not os.path.exists(zp): print('없음', zp); return
    with zipfile.ZipFile(zp) as z:
        name = [n for n in z.namelist() if n.endswith('lines.jsonl')][0]
        buf = collections.defaultdict(list)
        for l in z.open(name):
            r = json.loads(l); buf[tag + ':' + nfc(r.get('book', '?'))].append(r['text'])
    for b, ts in buf.items(): feed(b, ''.join(ts)); SRC[tag] += len(ts)
    print(tag, '책', len(buf), flush=True)
# 경판 행 자료(KOHICO를 행에 맞춘 것): 파일 이름 앞부분이 작품
with zipfile.ZipFile(ROOT + '/gyeongpan_lines_v5align.zip') as z:
    buf = collections.defaultdict(list)
    for n in sorted(z.namelist()):
        if n.endswith('.gt.txt'): buf['경판:' + nfc(os.path.basename(n).split('_')[0])].append(z.read(n).decode('utf-8').strip())
for b, ts in buf.items(): feed(b, ''.join(ts)); SRC['경판'] += len(ts)
print('경판 책', len(buf), flush=True)
# 월봉긔 검수 정답(있으면)
if os.path.exists(ROOT + '/private/wolbong_gt.json'):
    feed('경판:월봉긔(검수)', json.dumps(json.load(open(ROOT + '/private/wolbong_gt.json', encoding='utf-8')), ensure_ascii=False))
# 판독기에서 고쳐 보낸 면
for js in sorted(glob.glob(ROOT + '/학습자료/*/*.json')):
    try:
        d = json.load(open(js, encoding='utf-8'))
        feed('보내기:' + nfc(str(d.get('src') or d.get('commons') or os.path.basename(js)))[:40], ''.join(l.get('text', '') for l in d.get('lines', [])))
        SRC['보내기'] += 1
    except Exception as e: pass
zlines(ROOT + '/aihub/aihub_mokpan_lines_v9.zip' if os.path.exists(ROOT + '/aihub/aihub_mokpan_lines_v9.zip') else ROOT + '/aihub/aihub_mokpan_lines.zip', 'AI허브목판')
zlines(ROOT + '/aihub_pilsa/aihub_pilsa_lines.zip', 'AI허브필사')
zlines(ROOT + '/aihub_pilsa/aihub_pilsa_lines_all.zip', 'AI허브필사전체')
print('학습 자료 행 수', dict(SRC), flush=True)

# ---------------- 3) 결과 ----------------
rows = []
for (k, b), s in HIT.items():
    r = len(s) / len(VAL[k])
    if r >= 0.02: rows.append((k, b, r, len(s), len(VAL[k])))
best = {}
for k, b, r, a, n in rows:
    if k not in best or r > best[k][1]: best[k] = (b, r)
out = ROOT + '/검증_학습_겹침.tsv'
with open(out, 'w', encoding='utf-8') as f:
    f.write('검증 파일\t학습 책\t겹친 비율\t겹친 12-gram\t전체 12-gram\n')
    for k, b, r, a, n in sorted(rows, key=lambda x: -x[2]): f.write(f'{k}\t{b}\t{r:.3f}\t{a}\t{n}\n')
print('\n[겹침이 큰 검증 파일] 50% 이상: 같은 판본으로 보임 / 10~50%: 확인 필요')
for k, (b, r) in sorted(best.items(), key=lambda x: -x[1][1]):
    if r >= 0.10: print(f'  {r*100:5.1f}%  {k[:60]}  <=  {b[:50]}')
print('검증 파일 중 10% 넘게 겹친 것', sum(r >= 0.10 for _, r in best.values()), '/', len(VAL))
print('결과 파일:', out)
