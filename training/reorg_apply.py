# 드라이브 방각본_판독기 폴더를 새 구조로 옮긴다(Colab 터미널에서 python3로). 'go'를 붙이지 않으면 옮길 계획만 보여 준다.
import os, re, sys, unicodedata
N = lambda s: unicodedata.normalize('NFC', s)
R = '/content/drive/MyDrive'
B = [os.path.join(R, a) for a in os.listdir(R) if N(a) == '방각본_판독기'][0]
RENAME = {'private': '4_말뭉치_비공개', '판독할_PDF': '5_판독할_PDF', '판독결과': '6_판독결과', '학습자료': '7_보내준_면',
          '경판_판독기_학습_v2.ipynb': '판독기_학습.ipynb', '방각본_대량판독.ipynb': '판독기_대량판독.ipynb'}
RULES = [
    (r'^(aihub|aihub_pilsa|aihub_pilsa_ts)$', '3_AI허브_원본'),
    (r'^pre_v\d+\.zip$', '6_판독결과'),
    (r'(lines|wb_part\d+).*\.zip$|^pilsa_all_filter', '2_학습용_행자료'),
    (r'^(gyeongpan_model|svtrv2_model|ckpt_|parseq_model)|\.pt$|^svtrv2$|^web_|^ndl$|^ppocrv5_out$', '1_모델'),
    (r'\.(json|tsv|csv|txt|gsheet)$|^pilsa_check$|^pilsa_errors', '기록'),
    (r'\.(ipynb|py)$', '기록/옛_코드'),
]
KEEP = re.compile(r'^(_|[1-8]_|기록$|판독기_학습\.ipynb$|판독기_대량판독\.ipynb$)')
plan = []
for a in sorted(os.listdir(B)):
    n = N(a)
    if n in RENAME: plan.append((a, RENAME[n])); continue
    if KEEP.match(n): continue
    for pat, d in RULES:
        if re.search(pat, n): plan.append((a, d + '/' + a)); break
    else: plan.append((a, None))
for a, d in plan: print(f'{N(a)[:45]:45s} -> {d}')
if len(sys.argv) > 1 and sys.argv[1] == 'go':
    for a, d in plan:
        if d is None: print('그대로 둠', N(a)); continue
        dst = os.path.join(B, d); os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst): print('이미 있음, 건너뜀', d); continue
        os.rename(os.path.join(B, a), dst)
    print('끝. 맨 위:', sorted(N(x) for x in os.listdir(B)))
