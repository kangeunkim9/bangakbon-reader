# 지금 있는 시험 자료를 '방각본'과 '필사본' 두 묶음으로 다시 재는 스크립트(Colab 터미널에서 python3로).
# 방각본 = 경판 검증(KOHICO 경판, 학습 작품의 떼어 둔 면) + 금향졍긔 권하 뒤쪽 절반(경판) + AI허브 심청전_04(완판으로 보아 떼어 둔 책)
# 필사본 = AI허브 유이양문록 전부 + 명행정의록 권8
# 묶음 오류율은 시험마다 평균을 내지 않고 틀린 글자 수 / 전체 글자 수로 합친다. 언어 모델 보정 없이 잰다.
# 이 숫자는 공식 수치가 아니다(경판 검증은 학습 작품의 다른 면이라 후하게 나옴). 공식 수치는 8_검증용_작품으로 낸다.
import sys, types, os, json, random, collections, re, gzip, unicodedata, requests
_g = types.ModuleType('google'); _c = types.ModuleType('google.colab'); _d = types.ModuleType('google.colab.drive')
_d.mount = lambda *a, **k: None; _c.drive = _d; _g.colab = _c
sys.modules.update({'google': _g, 'google.colab': _c, 'google.colab.drive': _d})   # 터미널에서는 드라이브가 이미 연결돼 있다
RAW = 'https://raw.githubusercontent.com/kangeunkim9/bangakbon-reader/main/training/'
src = requests.get(RAW + 'train_svtrv2_pilsa.py', timeout=60).text
G = {'__name__': 'evalg', 'SV_EP': 6}
exec(src[:src.index('# ---------------- 4) 학습')], G)        # 자료와 v2 모델, 정답 점검 결과(저장된 것)까지
torch, read_s, lev, graphemes, DEV = G['torch'], G['read_s'], G['lev'], G['graphemes'], G['DEV']
ROOT, svm = G['ROOT'], G['svm']

def counts(preds, gts):
    e = sum(lev(graphemes(p), graphemes(g)) for p, g in zip(preds, gts)); n = sum(len(graphemes(g)) for g in gts)
    return e, n
# 금향졍긔 권하 뒤쪽 절반(36면부터): 사이트와 같은 행 나누기로 읽어 KOHICO 전사와 통째로 맞춘다
from rapidfuzz.distance import Levenshtein as Lv
os.environ['BANGAKBON_NO_MAIN'] = '1'
B = {'__name__': 'bulk'}; exec(requests.get(RAW + 'bulk_ocr.py', timeout=60).text, B)
KPG = json.load(gzip.open(ROOT + '/private/kohico_pages.json.gz', 'rt', encoding='utf-8'))
CL = re.compile(r'[ᄀ-ᅟꥠ-꥿]+[ᅠ-ᆧힰ-ퟆ]+[ᆨ-ᇿퟋ-ퟻ]*|.', re.S)
toks = lambda s: CL.findall(unicodedata.normalize('NFC', s))
GP = [v for kk, v in KPG.items() if kk.startswith('금향정기_2')][0]
GT = [g for g, _ in G['fix_rep']([(t, 0) for kk in sorted(GP) if kk not in ('31ᄀ', '31ᄂ') for t in toks(re.sub(r'\s+', '', GP[kk]))])[0]]
pdf = B['get_commons']('File:CNTS-00109637790 금향졍긔 이종. 권하.pdf', '/content/geumhyang2.pdf'); doc = B['fitz'].open(pdf)
FACES = []
for i in range(len(doc)):
    no = i + 1
    if not (3 <= no <= 62 or no == 65): continue
    im = B['norm_width'](B['page_gray'](doc, i)); t, bt, cols_ = B['segment'](im)[:3]
    FACES.append((no, [im.crop((max(0, a - 3), t, c + 3, bt)) for a, c in cols_]))
def geum(m):
    own, flat = [], []
    for no, crops in FACES:
        for s in read_s(m, crops):
            tk = toks(s); flat += tk; own += [no] * len(tk)
    err = collections.Counter(); n = collections.Counter(no >= 36 for no in own)
    for op in Lv.editops(flat, GT):
        p = min(op.src_pos, len(own) - 1); err[own[p] >= 36] += 1
    return err[True], n[True]

TESTS = {'방각본': {'경판 검증': G['KVA'], '심청전_04(완판)': G['AHte']}, '필사본': {'유이양문록·명행정의록8': G['PS_te']}}
def run(m):
    R = {}
    for grp, d in TESTS.items():
        for name, pairs in d.items(): R[(grp, name)] = counts(read_s(m, [p for p, _ in pairs]), [t for _, t in pairs])
    R[('방각본', '금향졍긔 권하(경판)')] = geum(m)
    return R
OUT = {}
for tag, path in (('v2', ROOT + '/svtrv2_model_v2.pt'), ('p1', ROOT + '/svtrv2_model_p1.pt')):
    svm.load_state_dict(torch.load(path, map_location=DEV, weights_only=False)['state']); OUT[tag] = run(svm)
    print(tag, '끝', flush=True)
print('\n[언어 모델 보정 없이 글자 오류율]          v2(이전)     p1(지금 사이트)   글자 수')
res = {}
for grp in ('방각본', '필사본'):
    keys = [k for k in OUT['v2'] if k[0] == grp]
    for k in keys:
        e2, n = OUT['v2'][k]; e1, _ = OUT['p1'][k]
        print(f'   {k[1]:24s} {e2/n*100:7.2f}%   {e1/n*100:7.2f}%   {n:8d}')
    E2 = sum(OUT['v2'][k][0] for k in keys); E1 = sum(OUT['p1'][k][0] for k in keys); N = sum(OUT['v2'][k][1] for k in keys)
    print(f' ▶ {grp:24s} {E2/N*100:7.2f}%   {E1/N*100:7.2f}%   {N:8d}')
    res[grp] = {'v2': E2 / N, 'p1': E1 / N, 'n': N, 'parts': {k[1]: {'v2': OUT['v2'][k][0] / OUT['v2'][k][1], 'p1': OUT['p1'][k][0] / OUT['p1'][k][1], 'n': OUT['v2'][k][1]} for k in keys}}
json.dump(res, open(ROOT + '/result_groups_v2_p1.json', 'w'), ensure_ascii=False, indent=1)
print('결과 파일:', ROOT + '/result_groups_v2_p1.json')
