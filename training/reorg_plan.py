# 드라이브 방각본_판독기 폴더 정리 계획(옮기지 않고 목록만)을 만든다(Colab 터미널에서 python3로).
# 결과: 방각본_판독기/_폴더정리_계획.tsv (지금 이름, 종류, 크기, 옮길 곳)
import os, re, unicodedata
N = lambda s: unicodedata.normalize('NFC', s)
R = '/content/drive/MyDrive'
B = [os.path.join(R, a) for a in os.listdir(R) if N(a) == '방각본_판독기'][0]
def size(p):
    if os.path.isfile(p): return os.path.getsize(p)
    t = 0
    for dp, dn, fn in os.walk(p):
        for f in fn:
            try: t += os.path.getsize(os.path.join(dp, f))
            except OSError: pass
    return t
RULES = [
    (r'^8_', '(그대로) 8_검증용_작품'),
    (r'^경판_판독기_학습_v2\.ipynb$', '(맨 위, 이름 바꿈) 판독기_학습.ipynb'),
    (r'^방각본_대량판독\.ipynb$', '(맨 위, 이름 바꿈) 판독기_대량판독.ipynb'),
    (r'^private$', '4_말뭉치_비공개'),
    (r'^(aihub|aihub_pilsa|aihub_pilsa_ts)$', '3_AI허브_원본'),
    (r'^판독할_PDF$', '5_판독할_PDF'),
    (r'^(판독결과|pre_v\d+\.zip)$', '6_판독결과'),
    (r'^학습자료$', '7_보내준_면'),
    (r'(lines|wb_part\d+).*\.zip$|^pilsa_all_filter', '2_학습용_행자료'),
    (r'^(gyeongpan_model|svtrv2_model|ckpt_)|\.pt$|^svtrv2$|^web_|^gyeongpan_model_v5\.zip$', '1_모델'),
    (r'\.(json|tsv|csv)$|^학습자료_목록|^검증_학습_겹침|^_원문받기', '기록'),
    (r'\.(ipynb|py)$', '기록/옛_코드'),
]
rows = []
for a in sorted(os.listdir(B)):
    n = N(a); p = os.path.join(B, a); dest = '(확인 필요)'
    for pat, d in RULES:
        if re.search(pat, n): dest = d; break
    rows.append((n, '폴더' if os.path.isdir(p) else '파일', size(p), dest))
out = os.path.join(B, '_폴더정리_계획.tsv')
with open(out, 'w', encoding='utf-8') as f:
    f.write('지금 이름\t종류\t크기(MB)\t옮길 곳\n')
    for n, k, s, d in sorted(rows, key=lambda r: (r[3], r[0])): f.write(f'{n}\t{k}\t{s/1e6:.1f}\t{d}\n')
print(len(rows), '개 ->', out)
for n, k, s, d in sorted(rows, key=lambda r: (r[3], r[0])): print(f'{d:28s} {k} {s/1e6:9.1f}MB  {n[:50]}')
