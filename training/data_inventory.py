# 지금 사이트 모델(p1)까지 학습에 쓴 자료 목록을 만든다(Colab 터미널에서 python3로).
# p1 학습 셀(train_svtrv2_pilsa.py)과 똑같이 자료를 불러, 책·작품마다 학습에 들어간 행 수와 시험용으로 뺀 행 수를 센다.
# 글은 옮기지 않고 이름과 행 수만 적는다. 결과: 드라이브 방각본_판독기/학습자료_목록.tsv
import sys, types, os, json, re, zipfile, collections, unicodedata, requests
_g = types.ModuleType('google'); _c = types.ModuleType('google.colab'); _d = types.ModuleType('google.colab.drive')
_d.mount = lambda *a, **k: None; _c.drive = _d; _g.colab = _c
sys.modules.update({'google': _g, 'google.colab': _c, 'google.colab.drive': _d})
RAW = 'https://raw.githubusercontent.com/kangeunkim9/bangakbon-reader/main/training/'
src = requests.get(RAW + 'train_svtrv2_pilsa.py', timeout=60).text
G = {'__name__': 'inv', 'SV_EP': 6}
exec(src[:src.index('\n# ---------------- 4) 학습 (')], G)
nfc = lambda s: unicodedata.normalize('NFC', s)
ROWS = []      # (자료, 묶음, 책·작품, 학습 행, 시험 행, 비고)
# 1) 경판(KOHICO 2023을 행에 맞춘 것): 작품마다 학습 면 / 검증으로 떼어 둔 면
tr = collections.Counter(nfc(os.path.basename(p).split('_')[0]) for p, _ in G['KTR'] if '/data5/' in p)
va = collections.Counter(nfc(os.path.basename(p).split('_')[0]) for p, _ in G['KVA'] if '/data5/' in p)
for w in sorted(set(tr) | set(va)): ROWS.append(('국어 역사 자료 말뭉치(KOHICO) 2023', '경판', w, tr[w], va[w], '같은 작품의 일부 면을 검증으로 뗌'))
wb_tr = sum('/data5/' not in p for p, _ in G['KTR']); wb_va = sum('/data5/' not in p for p, _ in G['KVA'])
ROWS.append(('사람 검수 정답', '경판', '월봉긔', wb_tr, wb_va, '이번 학습에는 정답 파일이 없어 빠짐' if wb_tr + wb_va == 0 else ''))
# 2) 판독기에서 고쳐 보낸 면
c = collections.Counter(re.sub(r'_\d+$', '', os.path.basename(p)[:-4]) for p, _ in G['INTR'])
ROWS.append(('판독기 사용자가 고쳐 보낸 면', '보내기', f'{len(c)}면', len(G['INTR']), 0, '학습 때 3번씩 반복'))
# 3) AI허브 목판본
c = collections.Counter(nfc(b) for _, _, b in G['AH'])
for b in sorted(c): ROWS.append(('AI허브 옛한글 OCR(목판본)', '목판본', b, 0 if '심청전_04' in b else c[b], c[b] if '심청전_04' in b else 0, '시험용(완판으로 보아 뗌)' if '심청전_04' in b else '에폭마다 25,000행씩 뽑아 씀'))
# 4) AI허브 필사본 작은 묶음(v2 때부터)과 전체 묶음(p1에서 더함)
small = collections.Counter(); smallte = collections.Counter()
PD = '/content/pilsa_lines'
if os.path.exists(PD + '/lines.jsonl'):
    for l in open(PD + '/lines.jsonl', encoding='utf-8'):
        r = json.loads(l); b = nfc(r['book'])
        (smallte if re.search(r'유이양문록|명행정의록8_', b) else small)[b] += 1
for b in sorted(set(small) | set(smallte)): ROWS.append(('AI허브 옛한글 OCR(필사본, 작은 묶음)', '필사본', b, small[b], smallte[b], 'v2부터' + (' / 시험용' if smallte[b] else '')))
ALL = G['ALL']
keepi = set(json.load(open(G['FILT'])))
allc = collections.Counter(); keptc = collections.Counter()
for i, (_, _, b) in enumerate(ALL):
    allc[nfc(b)] += 1
    if i in keepi: keptc[nfc(b)] += 1
for b in sorted(allc): ROWS.append(('AI허브 옛한글 OCR(필사본, 전체)', '필사본', b, keptc[b], 0, f'p1에서 더함 / 정답 점검으로 {allc[b]-keptc[b]}행 뺌 / 에폭마다 6만 행씩 뽑아 씀'))
ROWS.append(('AI허브 옛한글 OCR(필사본, 전체)', '필사본', '유이양문록·명행정의록 권8', 0, len(G['PS_te']), '시험용'))
ROWS.append(('AI허브 옛한글 OCR(필사본, 전체)', '언간', '경주최씨댁한글편지', 0, len(G['EG_te']), '시험용(언간, 보고에서는 뺌)'))
ROWS.append(('위키미디어 공용(국립중앙도서관 원문)', '경판', '금향졍긔 권하', 0, 0, '시험용(행 단위가 아니라 면 통째로 채점)'))
out = G['ROOT'] + '/기록/학습자료_목록.tsv'
with open(out, 'w', encoding='utf-8') as f:
    f.write('자료\t묶음\t책·작품\t학습 행\t시험 행\t비고\n')
    for r in ROWS: f.write('\t'.join(map(str, r)) + '\n')
print('줄', len(ROWS), '학습 행 합계', sum(r[3] for r in ROWS), '시험 행 합계', sum(r[4] for r in ROWS))
print('결과 파일:', out)
