#@title ㉘ 필사본 학습 자료 점검: 10차 모델로 학습용(TS_2)·검증용(VS_2) 행을 읽어 오류율을 비교하고, 크게 틀린 행을 저장
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'regex', 'rapidfuzz'], check=False)
import os, re, json, random, shutil, collections, requests, zipfile
import numpy as np, torch
from PIL import Image
from google.colab import drive; drive.mount('/content/drive')
ROOT = '/content/drive/MyDrive/방각본_판독기'
get = lambda f: requests.get(f'https://raw.githubusercontent.com/kangeunkim9/bangakbon-reader/main/training/{f}', timeout=60).text
src7 = get('train_v7.py'); exec(src7[:src7.index('# ---------------- 1)')], globals())
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PD = '/content/pilsa_lines_all'
if not os.path.exists(PD + '/lines.jsonl'):
    os.makedirs(PD, exist_ok=True)
    with zipfile.ZipFile(ROOT + '/aihub_pilsa/aihub_pilsa_lines_all.zip') as z: z.extractall(PD)
R = [json.loads(l) for l in open(PD + '/lines.jsonl', encoding='utf-8')]
N_TS = 321809   # ㉕ 셀 출력: TS_2 묶음이 먼저(321,809행), 그 뒤가 VS_2
for i, r in enumerate(R): r['src'] = 'TS' if i < N_TS else 'VS'
print('전체', len(R), '/ 앞 행 예:', {k: R[0][k] for k in R[0] if k != 'text'}, '/ 뒤 행 예:', {k: R[-1][k] for k in R[-1] if k != 'text'}, flush=True)
m, cs = load_model(ROOT + '/gyeongpan_model_v10.pt', DEV)
rnd = random.Random(7); out = {}
from rapidfuzz.distance import Levenshtein as Lv
def run(rows):
    pr = predict(m, cs, [PD + '/' + r['img'] for r in rows], DEV)
    return [(Lv.distance(p, r['text']) / max(1, len(r['text'])), p, r) for p, r in zip(pr, rows)]
for src in ('TS', 'VS'):
    rows = [r for r in R if r['src'] == src and os.path.exists(PD + '/' + r['img'])]
    smp = rnd.sample(rows, 800); res = run(smp); out[src] = res
    e = [x[0] for x in res]
    print(f'{src}: 행 {len(rows)}  10차 오류율 평균 {np.mean(e)*100:.1f}%  중앙값 {np.median(e)*100:.1f}%  50% 넘게 틀린 행 {np.mean(np.array(e)>0.5)*100:.1f}%', flush=True)
# 책별로 (TS) 크게 틀린 비율
bk = collections.defaultdict(list)
for e, p, r in out['TS']: bk[r['book']].append(e)
worst = sorted(((np.mean(v), len(v), b) for b, v in bk.items() if len(v) >= 4), reverse=True)[:15]
print('TS 책별 오류율(나쁜 순):', [(b, f'{a*100:.0f}%', n) for a, n, b in worst], flush=True)
ED = ROOT + '/pilsa_check'; os.makedirs(ED, exist_ok=True)
with open(ED + '/check.tsv', 'w', encoding='utf-8') as f:
    f.write('no\t묶음\t오류율\t책\t사진\t정답\t10차판독\n')
    k = 0
    for src in ('TS', 'VS'):
        for e, p, r in sorted(out[src], key=lambda x: -x[0])[:40]:
            nm = f'{k:03d}_{src}.png'; Image.open(PD + '/' + r['img']).save(ED + '/' + nm)
            f.write(f"{k}\t{src}\t{e:.2f}\t{r['book']}\t{nm}\t{r['text']}\t{p}\n"); k += 1
print('크게 틀린 행 80개 저장:', ED, flush=True)
