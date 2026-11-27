# 8_검증용_작품/보완 필요 안의 '소장처별 폴더'를 풀어, 작품 폴더를 시트 순서대로 '1. 이름', '2. 이름' ...으로 바꾼다(Colab 터미널에서 python3로).
# 시트의 폴더 이름 목록은 /content/v8names.json(시트 순서)에 미리 넣어 둔다. 'go'를 붙이지 않으면 계획만 보여 준다.
# 완성본 안의 작품은 완성본에 그대로 두고 번호만 붙인다. 비게 된 소장처 폴더만 지운다.
import os, re, sys, json, unicodedata
N = lambda s: unicodedata.normalize('NFC', s)
R = '/content/drive/MyDrive'
B = [os.path.join(R, a) for a in os.listdir(R) if N(a) == '방각본_판독기'][0]
V = [os.path.join(B, a) for a in os.listdir(B) if N(a).startswith('8_')][0]
sub = {N(a): os.path.join(V, a) for a in os.listdir(V) if os.path.isdir(os.path.join(V, a))}
TODO, DONE = sub['보완 필요'], sub['완성본']
names = [N(x) for x in json.load(open('/content/v8names.json', encoding='utf-8'))]
num = {n: i + 1 for i, n in enumerate(names)}
strip = lambda s: re.sub(r'^\d+\.\s*', '', s)
found, extra = {}, []
for top in (TODO, DONE):
    for dp, dn, fn in os.walk(top):
        if N(os.path.basename(dp)) == '원문': dn[:] = []; continue
        for d in list(dn):
            k = strip(N(d))
            if k in num:
                found.setdefault(k, []).append((os.path.join(dp, d), top)); dn.remove(d)   # 작품 폴더 안으로는 안 들어간다
            elif dp != top or '_' in d: extra.append(os.path.relpath(os.path.join(dp, d), V))
miss = [n for n in names if n not in found]; dup = {k: v for k, v in found.items() if len(v) > 1}
print('시트', len(names), '/ 찾은 작품 폴더', len(found), '/ 시트에 있는데 폴더 없음', len(miss), '/ 같은 이름 폴더 둘 이상', len(dup))
for n in miss: print('  폴더 없음:', n)
for k, v in dup.items(): print('  중복:', k, [os.path.relpath(p, V) for p, _ in v])
for e in extra: print('  시트에 없는 폴더:', e)
if len(sys.argv) > 1 and sys.argv[1] == 'go':
    if dup: sys.exit('중복이 있어 멈춤')
    moved = 0
    for k, [(p, top)] in found.items():
        dst = os.path.join(top, f'{num[k]}. {k}')
        if N(p) == N(dst): continue
        if os.path.exists(dst): print('이미 있음, 건너뜀', dst); continue
        os.rename(p, dst); moved += 1
    gone = 0
    for a in os.listdir(TODO):
        p = os.path.join(TODO, a)
        if os.path.isdir(p) and not re.match(r'^\d+\.', N(a)):
            try: os.rmdir(p); gone += 1
            except OSError: print('비어 있지 않아 남김:', N(a))
    print('옮김', moved, '/ 지운 빈 소장처 폴더', gone)
    print('보완 필요 안 폴더 수', len(os.listdir(TODO)), '/ 완성본', sorted(N(x) for x in os.listdir(DONE)))
