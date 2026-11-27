#@title ㉙ 국어 역사 자료 말뭉치 2023·2024·2025 받기 → 드라이브 private/kohico_raw (저장소에 올리지 않음) + 작품 목록
# 인증키는 Colab 보안 비밀(KOHICO_KEY_2023/2024/2025)에서 읽고, 화면에 찍지 않는다.
# 이용 약정: 원문 복제·배포 금지, 결과물 공개 전 국립국어원 승인, 이용 기간 뒤 원자료 파기.
import os, re, io, json, glob, zipfile, collections, requests
from google.colab import userdata, drive
drive.mount('/content/drive')
ROOT = '/content/drive/MyDrive/방각본_판독기'; RAW = ROOT + '/4_말뭉치_비공개/kohico_raw'; os.makedirs(RAW, exist_ok=True)
API = 'https://kli.korean.go.kr/restapi/v1/corpus/download'
for yr in ('2023', '2024', '2025'):
    dst = f'{RAW}/{yr}'; os.makedirs(dst, exist_ok=True)
    if glob.glob(dst + '/*.zip'): print(yr, '이미 받음'); continue
    try: key = userdata.get(f'KOHICO_KEY_{yr}')
    except Exception as e: print(yr, '보안 비밀을 못 읽음:', type(e).__name__); continue
    r = requests.get(API, params={'keyVal': key}, timeout=60)
    print(yr, 'API 응답', r.status_code, r.headers.get('content-type'), len(r.content), '바이트', flush=True)
    url = r.text.strip().strip('"')
    if not url.startswith('http'):
        print('  응답 앞부분:', re.sub(r'[0-9a-fA-F]{32}', '(키)', r.text[:300])); continue
    d = requests.get(url, timeout=600, stream=True)
    ct = d.headers.get('content-type', ''); cd = d.headers.get('content-disposition', '')
    print('  내려받기', d.status_code, ct, cd[:120], flush=True)
    name = (re.findall(r"filename\*?=(?:UTF-8'')?\"?([^\";]+)", cd) or [f'kohico_{yr}.zip'])[0]
    name = requests.utils.unquote(name)
    with open(f'{dst}/{name}', 'wb') as f:
        for ch in d.iter_content(1 << 20): f.write(ch)
    print('  저장', name, os.path.getsize(f'{dst}/{name}') // 1024, 'KB', flush=True)

# ---- 작품 목록: 압축 안의 문서마다 서명·판본·간행연도·유형만 뽑는다(본문은 보지 않음) ----
def fields(t):
    g = lambda p: (re.search(p, t) or [None, ''])[1].strip()
    return {'서명': g(r'한글서명\s*[:：]\s*(.*)') or g(r'서명\s*[:：]\s*(.*)'), '판본': g(r'판본\s*[:：]\s*(.*)'),
            '간행': g(r'간행연도\s*[:：]\s*(.*)'), '유형': g(r'(?:자료\s*유형|문헌\s*유형|분야)\s*[:：]\s*(.*)')}
rows = []
for yr in ('2023', '2024', '2025'):
    for z in glob.glob(f'{RAW}/{yr}/*.zip'):
        with zipfile.ZipFile(z) as zf:
            names = [n for n in zf.namelist() if not n.endswith('/')]
            print(yr, os.path.basename(z), '파일', len(names), '예:', names[:3], flush=True)
            for n in names:
                if not re.search(r'\.(txt|json|xml)$', n, re.I): continue
                raw = zf.read(n)
                for enc in ('utf-8-sig', 'utf-16', 'cp949'):
                    try: t = raw.decode(enc); break
                    except Exception: t = None
                if not t: continue
                f = fields(t[:5000]); f.update(년판=yr, 파일=n, 글자수=len(t)); rows.append(f)
json.dump(rows, open(RAW + '/목록.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
c = collections.Counter((r['년판'], r['판본'] or '(판본 없음)') for r in rows)
print('문서', len(rows), '개 / 년판·판본별:'); [print(' ', k, v) for k, v in sorted(c.items())]
print('방각본(경판·완판·안성판) 문서:')
for r in rows:
    if re.search(r'경판|완판|안성|방각', r['판본'] + r['서명']): print(' ', r['년판'], r['서명'], '|', r['판본'], '|', r['간행'])
