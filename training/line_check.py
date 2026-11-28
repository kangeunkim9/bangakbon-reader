# 행 나누기가 잘못됐을 가능성이 큰 면을 골라 목록으로 만든다.
# 쓰는 법: python3 line_check.py [결과 json이 든 폴더 ...] [-o 목록.tsv]
#   폴더를 주지 않으면 사이트에 올린 미리 판독 결과(web/pre)를 본다(학습에 쓴 경판 책들이 대부분 여기 있다).
#   Colab에서 eval_book.py --all 을 돌린 뒤 드라이브 '기록/검증결과'를 주면 검증 작품도 같이 본다.
# 판단 기준(책 하나 안에서 서로 견준다)
#   행 수: 그 책에서 가장 흔한 행 수(보통 행 수)보다 2행 이상 많거나 적은 면
#   넓은 행: 그 면 행 너비 중앙값의 1.6배가 넘는 행이 있는 면(두 행이 한 행으로 붙었을 수 있음)
#   좁은 행: 중앙값의 0.45배가 안 되는 행이 있는 면(한 행이 둘로 갈렸거나 판심·광곽 조각)
#   빈 면: 책 중간인데 행을 못 찾았거나(empty) 표지처럼 보여 숨긴(nomatch) 면
import os, sys, json, glob, collections, statistics, unicodedata
nfc = lambda s: unicodedata.normalize('NFC', s)
args = [a for a in sys.argv[1:]]
out = 'line_check.tsv'
if '-o' in args: i = args.index('-o'); out = args[i + 1]; del args[i:i + 2]
dirs = args or [os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'web', 'pre')]


def faces_of(d):
    """결과 json 하나에서 (면 이름, 상태, 행 목록[(a, c)]) 목록을 뽑는다. 사이트 미리 판독 꼴과 eval_book 꼴을 모두 읽는다."""
    if 'pages' in d:   # 사이트 미리 판독(web/pre)
        for p in d['pages']:
            yield f"{p['no']}쪽" + (' ' + p['side'] if p.get('side') else ''), p.get('status', 'ok'), [tuple(c) for c in p.get('cols', [])]
    elif '면별 행' in d:   # eval_book.py 결과
        for f in d['면별 행']:
            yield f['면'], f['상태'], [tuple(c) for c in f['행']]


rows = []
for dr in dirs:
    for fp in sorted(glob.glob(os.path.join(dr, '*.json'))):
        if os.path.basename(fp) == 'index.json': continue
        try: d = json.load(open(fp, encoding='utf-8'))
        except Exception: continue
        book = nfc(d.get('이름') or d.get('src') or os.path.basename(fp)[:-5])
        F = list(faces_of(d))
        if not F: continue
        ok = [i for i, (_, st, cols) in enumerate(F) if st == 'ok' and cols]
        if len(ok) < 3: continue
        mode = collections.Counter(len(F[i][2]) for i in ok).most_common(1)[0][0]
        first, last = ok[0], ok[-1]
        for i, (name, st, cols) in enumerate(F):
            why = []
            if st != 'ok' or not cols:
                if first < i < last: why.append('빈 면(책 중간)' if st == 'empty' or not cols else '표지처럼 보여 숨김(책 중간)')
            else:
                n = len(cols)
                if abs(n - mode) >= 2: why.append(f'행 수 {n} (보통 {mode})')
                w = [c - a for a, c in cols]; med = statistics.median(w)
                wide = [k + 1 for k, x in enumerate(w) if x > med * 1.6]
                thin = [k + 1 for k, x in enumerate(w) if x < med * 0.45]
                if wide: why.append('넓은 행 ' + ','.join(map(str, wide)) + '번')
                if thin: why.append('좁은 행 ' + ','.join(map(str, thin)) + '번')
            if why: rows.append((book, name, len(cols), mode, ' / '.join(why)))

with open(out, 'w', encoding='utf-8') as f:
    f.write('책\t면\t나눈 행 수\t그 책 보통 행 수\t의심되는 까닭\n')
    for r in rows: f.write('\t'.join(map(str, r)) + '\n')
books = collections.Counter(r[0] for r in rows)
print(f'의심 면 {len(rows)}개 / 책 {len(books)}권 →', out)
for b, k in books.most_common(15): print(f'  {k:4d}  {b}')
