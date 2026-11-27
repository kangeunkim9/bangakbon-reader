# AI허브 옛한글(504) 필사본 묶음(VS_2·VL_2·TL_2)의 생김새를 살펴본다. 드라이브 aihub_pilsa/download.tar를 풀어 조각을 합친다.
import os, glob, json, tarfile, zipfile, subprocess, re, collections
from google.colab import drive; drive.mount('/content/drive')
ROOT = '/content/drive/MyDrive/방각본_판독기'; SRC = ROOT + '/3_AI허브_원본/aihub_pilsa'; W = '/content/pilsa'; os.makedirs(W, exist_ok=True)
if os.path.exists(SRC + '/download.tar') and not glob.glob(W + '/**/*.zip*', recursive=True):
    with tarfile.open(SRC + '/download.tar') as t: t.extractall(W)
parts = collections.defaultdict(list)
for p in glob.glob(W + '/**/*.part*', recursive=True): parts[re.sub(r'\.part\d+$', '', p)].append(p)
for base, ps in parts.items():
    if not os.path.exists(base):
        ps.sort(key=lambda s: int(re.search(r'part(\d+)$', s).group(1)))
        with open(base, 'wb') as o:
            for p in ps: o.write(open(p, 'rb').read()); os.remove(p)
for z in glob.glob(W + '/**/*.zip', recursive=True):
    print(os.path.relpath(z, W), f'{os.path.getsize(z)/1e6:.0f}MB')
    with zipfile.ZipFile(z) as f:
        n = f.namelist(); print('  파일', len(n), n[:3])
        js = [x for x in n if x.endswith('.json')]
        if js and len(n) < 200000:
            d = json.loads(f.read(js[0]).decode('utf-8-sig')); s = json.dumps(d, ensure_ascii=False)
            print('  예시 라벨:', s[:1500])
        books = collections.Counter(x.split('/')[-2] if '/' in x else '' for x in n)
        print('  책(폴더)별 파일 수 앞부분:', books.most_common(8), '... 총', len(books))
