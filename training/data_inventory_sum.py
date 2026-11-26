# 학습자료_목록.tsv(책·권마다 한 줄)를 작품 단위로 묶어 학습자료_목록_작품별.csv로 만든다(Colab 터미널에서 python3로).
import csv, re, collections
ROOT = '/content/drive/MyDrive/방각본_판독기'
KP = {'ss1': '사씨남정기 상(경판32장본, 1851)', 'ss2': '사씨남정기 하(경판34장본, 1851)', 'sy1': '서유기 상(경판31장본, 1856)', 'sy2': '서유기 하(경판28장본, 1856)',
      'w00': '곽분양전 상(경판)', 'w01': '곽분양전 중(경판)', 'w02': '곽분양전 하(경판)', 'w03': '삼국지 1(경판)', 'w04': '삼국지 2(경판)', 'w05': '삼국지 3(경판)',
      'w06': '심청전(경판24장본)', 'w07': '쌍주기연(경판16장본)', 'w08': '옥주호연(경판29장본)', 'w09': '장한절효기(경판29장본)', 'w10': '홍길동전(경판30장본, 1860)',
      'w11': '황운전(경판29장본)', 'w12': '황운전(경판30장본)'}
def work(src, b):
    if src.startswith('국어 역사'): return KP.get(b, b)
    m = re.match(r'(?:목판본|필사본)_\d+_[^_]+_(.+?)(\d+)?(?:_\d+)?$', b)
    return m.group(1) if m else b
agg = collections.OrderedDict()
for r in csv.DictReader(open(ROOT + '/학습자료_목록.tsv', encoding='utf-8'), delimiter='\t'):
    k = (r['자료'], r['묶음'], work(r['자료'], r['책·작품']))
    a = agg.setdefault(k, {'권': 0, '학습': 0, '시험': 0, '비고': set()})
    a['권'] += 1; a['학습'] += int(r['학습 행']); a['시험'] += int(r['시험 행'])
    note = re.sub(r'정답 점검으로 \d+행 뺌 / ', '', r['비고']); a['비고'].add(note) if note else None
with open(ROOT + '/학습자료_목록_작품별.csv', 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f); w.writerow(['자료', '묶음', '작품', '권·책 수', '학습 행', '시험 행', '비고'])
    for (s, g, n), a in agg.items(): w.writerow([s, g, n, a['권'], a['학습'], a['시험'], ' / '.join(sorted(a['비고']))])
print('작품 줄', len(agg), '->', ROOT + '/학습자료_목록_작품별.csv')
