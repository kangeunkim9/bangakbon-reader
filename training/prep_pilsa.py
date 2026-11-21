#@title ㉑ AI허브 필사본 → 행 단위 학습 자료 (드라이브 방각본_판독기/aihub_pilsa/download.tar)
# 목판본 때 쓴 prep_aihub.py의 행 자르기를 그대로 쓰고, 읽을 곳과 저장할 곳만 바꾼다.
import os, requests
src = requests.get('https://raw.githubusercontent.com/kangeunkim9/bangakbon-reader/main/training/prep_aihub.py', timeout=60).text
G_ = {'__name__': 'prep'}; exec(src, G_)
from google.colab import drive; drive.mount('/content/drive')
G_['SRC'] = '/content/drive/MyDrive/방각본_판독기/aihub_pilsa'
G_['WORK'] = '/content/pilsa'; G_['OUT'] = '/content/pilsa_lines'
os.makedirs(G_['WORK'], exist_ok=True); os.makedirs(G_['OUT'], exist_ok=True)
import glob, tarfile, shutil
if not glob.glob(G_['WORK'] + '/**/*.zip*', recursive=True):
    with tarfile.open(G_['SRC'] + '/download.tar') as t: t.extractall(G_['WORK'])
rows = G_['build']()
import collections
c = collections.Counter(r['book'].split('_')[3] if r['book'].count('_') >= 3 else r['book'] for r in rows)
print('작품별 행 수:', c.most_common(40))
shutil.make_archive('/content/pilsa_lines', 'zip', G_['OUT'])
shutil.copy('/content/pilsa_lines.zip', G_['SRC'] + '/aihub_pilsa_lines.zip')
print('드라이브 저장 끝:', G_['SRC'] + '/aihub_pilsa_lines.zip', f"{os.path.getsize('/content/pilsa_lines.zip')/1e6:.0f}MB")
