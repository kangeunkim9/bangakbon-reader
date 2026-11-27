#@title ㉕ AI허브 필사본 전체(학습 TS_2 21GB + 검증 VS_2) → 행 단위 학습 자료 aihub_pilsa_lines_all.zip
# 드라이브의 tar를 /content로 복사하지 않고 바로 풀어 디스크를 아낀다. 조각(.part)은 합치면서 지운다.
import os, glob, tarfile, shutil, collections, requests
src = requests.get('https://raw.githubusercontent.com/kangeunkim9/bangakbon-reader/main/training/prep_aihub.py', timeout=60).text
G_ = {'__name__': 'prep'}; exec(src, G_)
from google.colab import drive; drive.mount('/content/drive')
R = '/content/drive/MyDrive/방각본_판독기'
G_['WORK'] = W = '/content/pilsa_all'; G_['OUT'] = '/content/pilsa_lines_all'; G_['SRC'] = R + '/3_AI허브_원본/aihub_pilsa'
os.makedirs(W, exist_ok=True); os.makedirs(G_['OUT'], exist_ok=True)
for tf in (R + '/3_AI허브_원본/aihub_pilsa/download.tar', R + '/3_AI허브_원본/aihub_pilsa_ts/TS_2.tar'):
    tag = W + '/.' + os.path.basename(tf) + '.done'
    if os.path.exists(tag): continue
    print('푸는 중:', tf, flush=True)
    with tarfile.open(tf) as t: t.extractall(W)
    open(tag, 'w').close()
    print('남은 디스크', shutil.disk_usage('/content').free // 2**30, 'GB', flush=True)
rows = G_['build']()
c = collections.Counter(r['book'].split('_')[3] if r['book'].count('_') >= 3 else r['book'] for r in rows)
print('작품 수', len(c), '/ 행', len(rows)); print('작품별 행 수(앞부분):', c.most_common(30))
shutil.make_archive('/content/pilsa_lines_all', 'zip', G_['OUT'])
shutil.copy('/content/pilsa_lines_all.zip', R + '/3_AI허브_원본/aihub_pilsa/aihub_pilsa_lines_all.zip')
print('드라이브 저장 끝:', f"{os.path.getsize('/content/pilsa_lines_all.zip')/1e9:.1f}GB")
