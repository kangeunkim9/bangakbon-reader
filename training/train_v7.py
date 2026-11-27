# 7차 학습 셀(Colab). KOHICO 전사와 사람 검수 정답은 저장소에 넣지 않았으므로, 드라이브의 private 폴더에서 읽도록 바꿔 둔 공개용 사본입니다.
#@title 7차 학습: 경판 학습 자료 다시 만들기(5차 모델로 정렬) + AI허브 목판본 → 7차 학습 → 금향졍긔 시험
import subprocess, sys
subprocess.run([sys.executable,'-m','pip','install','-q','regex','pymupdf'],check=False)
import os, io, re, glob, json, gzip, lzma, base64, zipfile, shutil, random, time, math, collections, unicodedata, requests, fitz, regex, cv2
import numpy as np
from scipy.signal import find_peaks
from google.colab import drive; drive.mount('/content/drive')
ROOT='/content/drive/MyDrive/방각본_판독기'
# ==== 공통 코드: 데이터 읽기, 모델, 학습, 평가 ====
import os, glob, random, unicodedata, json, math, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image, ImageOps, ImageFilter

H = 64          # 행 이미지를 눕힌 뒤 맞출 높이(px)
MAXW = 1600     # 최대 너비

def nfd(s): return unicodedata.normalize('NFD', s)
def nfc(s): return unicodedata.normalize('NFC', s)

def load_pairs(root):
    pairs = []
    for gt in sorted(glob.glob(os.path.join(root, '**', '*.gt.txt'), recursive=True)):
        img = next((gt[:-7] + e for e in ('.png', '.jpg') if os.path.exists(gt[:-7] + e)), None)
        if img:
            t = open(gt, encoding='utf-8').read().strip()
            if t: pairs.append((img, t))
    return pairs

def prep(path, aug=False):
    im = Image.open(path).convert('L')
    im = im.rotate(90, expand=True)              # 세로행 → 가로: 위쪽이 왼쪽으로
    im = ImageOps.autocontrast(im, cutoff=1)
    if aug:
        if random.random() < 0.5: im = im.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 1.0)))
        if random.random() < 0.5:
            w, h = im.size; dy = random.randint(-4, 4)
            im = im.transform((w, h), Image.AFFINE, (1, random.uniform(-0.03, 0.03), 0, 0, 1, dy), fillcolor=255)
    w, h = im.size
    nw = min(MAXW, max(32, int(w * H / h * (random.uniform(0.9, 1.1) if aug else 1))))
    im = im.resize((nw, H), Image.BILINEAR)
    a = 1.0 - np.asarray(im, dtype=np.float32) / 255.0   # 글자=1, 바탕=0
    if aug: a = np.clip(a * random.uniform(0.8, 1.2) + np.random.normal(0, 0.03, a.shape), 0, 1)
    return a

class Charset:
    def __init__(self, texts):
        chars = sorted(set(''.join(nfd(t) for t in texts)))
        self.chars = chars; self.idx = {c: i + 1 for i, c in enumerate(chars)}   # 0 = blank
    def encode(self, t): return [self.idx[c] for c in nfd(t) if c in self.idx]
    def decode(self, seq):
        out, prev = [], 0
        for s in seq:
            if s != prev and s != 0: out.append(self.chars[s - 1])
            prev = s
        return nfc(''.join(out))
    def __len__(self): return len(self.chars) + 1

class CRNN(nn.Module):
    def __init__(self, n):
        super().__init__()
        def blk(i, o, pool): return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True)] + ([nn.MaxPool2d(pool)] if pool else [])
        self.cnn = nn.Sequential(*blk(1, 64, (2, 2)), *blk(64, 128, (2, 2)), *blk(128, 256, None),
                                 *blk(256, 256, (2, 1)), *blk(256, 384, None), *blk(384, 384, (2, 1)), nn.Dropout2d(0.1))
        self.rnn = nn.LSTM(384 * (H // 16), 256, num_layers=2, bidirectional=True, batch_first=True, dropout=0.25)
        self.fc = nn.Linear(512, n)
    def forward(self, x):                     # x: B,1,H,W
        f = self.cnn(x)                       # B,C,H/16,W/4
        b, c, h, w = f.shape
        f = f.permute(0, 3, 1, 2).reshape(b, w, c * h)
        y, _ = self.rnn(f)
        return self.fc(y)                     # B,T,n

def batchify(arrs):
    W = max(a.shape[1] for a in arrs)
    x = np.zeros((len(arrs), 1, H, W), np.float32)
    for i, a in enumerate(arrs): x[i, 0, :, :a.shape[1]] = a
    return torch.from_numpy(x)

def lev(a, b):
    p = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        c = [i]
        for j, y in enumerate(b, 1): c.append(min(p[j] + 1, c[j - 1] + 1, p[j - 1] + (x != y)))
        p = c
    return p[-1]

def graphemes(s):
    import regex
    return regex.findall(r'\X', nfc(s))

@torch.no_grad()
def predict(model, cs, paths, dev, bs=16):
    model.eval(); out = []
    for i in range(0, len(paths), bs):
        x = batchify([prep(p) for p in paths[i:i + bs]]).to(dev)
        y = model(x).argmax(-1).cpu().numpy()
        out += [cs.decode(list(r)) for r in y]
    return out

def cer(preds, gts):
    e = sum(lev(graphemes(p), graphemes(g)) for p, g in zip(preds, gts))
    n = sum(len(graphemes(g)) for g in gts)
    return e / max(1, n)

def train(model, cs, tr, va, dev, epochs=60, bs=8, lr=1e-3, log=print, save='model.pt'):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * math.ceil(len(tr) / bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True); best = 9; hist = []
    for ep in range(1, epochs + 1):
        model.train(); random.shuffle(tr); t0 = time.time(); tot = 0
        for i in range(0, len(tr), bs):
            b = tr[i:i + bs]
            x = batchify([prep(p, aug=True) for p, _ in b]).to(dev)
            tg = [cs.encode(t) for _, t in b]
            y = model(x).log_softmax(-1).permute(1, 0, 2)          # T,B,n
            loss = ctc(y, torch.tensor(sum(tg, [])), torch.full((len(b),), y.shape[0], dtype=torch.long), torch.tensor([len(t) for t in tg]))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5); opt.step(); sch.step()
            tot += loss.item() * len(b)
        c = cer(predict(model, cs, [p for p, _ in va], dev), [t for _, t in va]) if va else float('nan')
        hist.append((ep, tot / len(tr), c))
        if c < best:
            best = c; torch.save({'state': model.state_dict(), 'chars': cs.chars, 'H': H}, save)
        log(f'epoch {ep:3d}  loss {tot/len(tr):.3f}  검증 CER {c*100:5.1f}%  (최고 {best*100:.1f}%)  {time.time()-t0:.0f}s')
    return hist

def load_model(path, dev):
    ck = torch.load(path, map_location=dev)
    cs = Charset([]); cs.chars = ck['chars']; cs.idx = {c: i + 1 for i, c in enumerate(cs.chars)}
    m = CRNN(len(cs)).to(dev); m.load_state_dict(ck['state']); m.eval()
    return m, cs

from torch.utils.data import Dataset, DataLoader
DEV="cuda" if torch.cuda.is_available() else "cpu"; print("장치",DEV)
def prep2(path, aug=False):
    im = Image.open(path).convert('L').rotate(90, expand=True)
    im = ImageOps.autocontrast(im, cutoff=1)
    if aug:
        r=random.random()
        if r<0.25: im=im.filter(ImageFilter.MinFilter(3))      # 획 굵게(먹 번짐)
        elif r<0.4: im=im.filter(ImageFilter.MaxFilter(3))     # 획 가늘게(마모)
        if random.random()<0.5: im=im.filter(ImageFilter.GaussianBlur(random.uniform(0.2,1.2)))
        if random.random()<0.5:
            w,h=im.size; im=im.transform((w,h),Image.AFFINE,(1,random.uniform(-0.04,0.04),0,0,1,random.randint(-5,5)),fillcolor=255)
        if random.random()<0.3: im=im.rotate(random.uniform(-1.2,1.2),fillcolor=255)
        if random.random()<0.4:
            g=random.uniform(0.6,1.6); im=im.point(lambda v: int(255*((v/255)**g)))
    w,h=im.size
    nw=min(MAXW,max(32,int(w*H/h*(random.uniform(0.85,1.15) if aug else 1))))
    a=1.0-np.asarray(im.resize((nw,H),Image.BILINEAR),dtype=np.float32)/255.0
    if aug: a=np.clip(a*random.uniform(0.75,1.25)+np.random.normal(0,0.04,a.shape),0,1)
    return a
class LineDS(Dataset):
    def __init__(s,pairs,cs,aug): s.p=pairs; s.cs=cs; s.aug=aug
    def __len__(s): return len(s.p)
    def __getitem__(s,i): p,t=s.p[i]; return prep2(p,s.aug), s.cs.encode(t)
def collate(b): return batchify([x for x,_ in b]), [t for _,t in b]
def train2(model,cs,tr,va,dev,epochs,bs=16,lr=1e-3,save='m.pt'):
    dl=DataLoader(LineDS(tr,cs,True),batch_size=bs,shuffle=True,num_workers=2,collate_fn=collate,drop_last=True)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=lr,total_steps=epochs*len(dl),pct_start=0.1)
    ctc=nn.CTCLoss(blank=0,zero_infinity=True); best=9; hist=[]
    for ep in range(1,epochs+1):
        model.train(); t0=time.time(); tot=0
        for x,tg in dl:
            x=x.to(dev); y=model(x).log_softmax(-1).permute(1,0,2)
            loss=ctc(y,torch.tensor(sum(tg,[])),torch.full((len(tg),),y.shape[0],dtype=torch.long),torch.tensor([len(t) for t in tg]))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5); opt.step(); sch.step(); tot+=loss.item()
        c=cer(predict(model,cs,[p for p,_ in va],dev),[t for _,t in va]); hist.append((ep,tot/len(dl),c))
        if c<best: best=c; torch.save({'state':model.state_dict(),'chars':cs.chars,'H':H},save)
        print(f'epoch {ep:3d}  loss {tot/len(dl):.3f}  검증 CER {c*100:5.1f}%  (최고 {best*100:.1f}%)  {time.time()-t0:.0f}s',flush=True)
    return hist

def binar(im):
    g=np.asarray(im.convert('L')); return (cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,51,12)>0).astype(np.uint8)
G=lambda s: regex.findall(r'\X', unicodedata.normalize('NFC', s))
def tri(s): g=G(s); return set(''.join(g[i:i+3]) for i in range(len(g)-2))
def align(a,b):
    n,m=len(a),len(b); D=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): D[i][0]=i
    for j in range(m+1): D[0][j]=j
    for i in range(1,n+1):
        ai=a[i-1]; Di=D[i]; Dp=D[i-1]
        for j in range(1,m+1): Di[j]=min(Dp[j]+1, Di[j-1]+1, Dp[j-1]+(ai!=b[j-1]))
    i,j=n,m; amap=[None]*n
    while i>0 and j>0:
        if D[i][j]==D[i-1][j-1]+(a[i-1]!=b[j-1]): amap[i-1]=j-1; i-=1; j-=1
        elif D[i][j]==D[i-1][j]+1: i-=1
        else: j-=1
    return amap
def get_pdf(title, out):
    if os.path.exists(out) and os.path.getsize(out)>100000: return
    r=requests.get('https://commons.wikimedia.org/w/api.php',params={'action':'query','prop':'imageinfo','iiprop':'url','format':'json','titles':title},headers=UA,timeout=60).json()
    url=list(r['query']['pages'].values())[0]['imageinfo'][0]['url'].split('?')[0]
    for i in range(5):
        resp=requests.get(url,headers=UA,timeout=300)
        if resp.status_code==200 and len(resp.content)>100000: break
        print('재시도',resp.status_code); time.sleep(10)
    open(out,'wb').write(resp.content); print(title,len(resp.content)//1_000_000,'MB')
def page_img(doc,p):
    x=doc[p].get_images(full=True)[0][0]
    return Image.open(io.BytesIO(doc.extract_image(x)['image'])).convert('L')
def segment2(im):
    b=binar(im); H_,W=b.shape
    lines=cv2.morphologyEx(b,cv2.MORPH_OPEN,np.ones((H_//4,1),np.uint8))|cv2.morphologyEx(b,cv2.MORPH_OPEN,np.ones((1,W//4),np.uint8))
    hr=cv2.morphologyEx(b,cv2.MORPH_OPEN,np.ones((1,W//3),np.uint8)).sum(1); rows=np.where(hr>W//4)[0]
    if len(rows)>=2 and rows.max()-rows.min()>H_*0.5: T,B=int(rows.min()),int(rows.max())
    else: T,B=int(H_*0.05),int(H_*0.95)
    bb=b&(1-lines)
    # 세로 범위: 테두리 안에서 글자 먹이 있는 줄까지
    rp=np.convolve(bb[T+5:B-5].mean(1),np.ones(15)/15,'same'); on=np.where(rp>max(0.01,rp.max()*0.08))[0]
    t=T+5+(on.min() if len(on) else 0)-12; bt=T+5+(on.max() if len(on) else B-T-10)+12
    t=max(T+2,t); bt=min(B-2,bt)
    band=bb[t:bt]; prof=np.convolve(band.mean(0),np.ones(7)/7,'same')
    ink=np.where(prof>0.02)[0]; x0,x1=int(ink.min()),int(ink.max())
    p=prof[x0:x1]-prof[x0:x1].mean(); ac=np.correlate(p,p,'full')[len(p)-1:]
    lags=np.arange(45,150); vals=ac[45:150]
    pk,_=find_peaks(vals); pk=[k for k in pk if vals[k]>0.5*vals.max()] or [int(np.argmax(vals))]
    p0=float(lags[pk[0]])                       # 가장 짧은 뚜렷한 주기 = 행 간격
    best=None
    for pp in np.arange(p0*0.94,p0*1.06,0.5):  # 간격과 시작점을 같이 맞춤: 경계가 먹 없는 곳을 지나가도록
        for o in np.arange(0,pp,1.0):
            xs=np.arange(x0-pp+o,x1+pp,pp); xs=xs[(xs>=0)&(xs<W)].astype(int)
            s=prof[xs].mean()
            if best is None or s<best[0]: best=(s,pp,o)
    _,pp,o=best
    xs=np.arange(x0-pp+o,x1+pp,pp); xs=xs[(xs>=0)&(xs<W)]
    r=int(pp*0.2); xs=[int(max(0,x-r)+np.argmin(prof[max(0,int(x)-r):min(W,int(x)+r+1)])) for x in xs]
    cols=[(a,c) for a,c in zip(xs[:-1],xs[1:]) if c-a>pp*0.5 and band[:,a:c].sum()>200]
    return t,bt,cols[::-1],pp
# 위·아래 경계 다듬기: 괘선이 있으면 괘선 바로 안쪽, 없으면 먹이 끝나는 곳까지(이미지 5%/95% 제한 없이)
def _runs(idx):
    out=[]
    for y in idx:
        y=int(y)
        if out and y==out[-1][1]+1: out[-1][1]=y
        else: out.append([y,y])
    return out
def vbounds(b, t, bt):
    H_,W=b.shape
    hr=cv2.morphologyEx(b,cv2.MORPH_OPEN,np.ones((1,W//3),np.uint8)).sum(1)
    R=[r for r in _runs(np.where((b.mean(1)>0.5)|(hr>W//4))[0]) if r[0]>H_*0.01 and r[1]<H_*0.99]  # 스캔 가장자리 띠는 뺀다
    top=[r for r in R if r[1]<H_*0.35 and t-120<r[1]<t+120]
    bot=[r for r in R if r[0]>H_*0.65 and bt-120<r[0]<bt+120]
    if top: t=top[-1][1]+2
    if bot: bt=bot[0][0]-2
    if not top or not bot:
        lines=cv2.morphologyEx(b,cv2.MORPH_OPEN,np.ones((H_//4,1),np.uint8))|cv2.morphologyEx(b,cv2.MORPH_OPEN,np.ones((1,W//4),np.uint8))
        sm=np.convolve((b&(1-lines)).mean(1),np.ones(15)/15,'same'); mid=sm[t:bt]
        th=max(0.01,(mid.max() if len(mid) else 0)*0.25)   # 바탕 얼룩까지 따라가지 않게 본문보다 진한 곳만
        if not bot:
            y=bt-12
            while y+1<min(int(H_*0.99),bt+80) and sm[y+1]>th: y+=1
            bt=max(bt,min(H_-1,y+12))
        if not top:
            y=t+12
            while y-1>max(int(H_*0.01),t-80) and sm[y-1]>th: y-=1
            t=min(t,max(0,y-12))
    return t,bt

_seg2=segment2
def segment3(im):
    t,bt,cols,pp=_seg2(im)
    b=binar(im); H_,W=b.shape
    hr=cv2.morphologyEx(b,cv2.MORPH_OPEN,np.ones((1,W//3),np.uint8)).sum(1); rows=np.where(hr>W//4)[0]
    if len(rows)>=2 and rows.max()-rows.min()>H_*0.5: t,bt=int(rows.min())+4,int(rows.max())-4   # 테두리 바로 안쪽까지
    t,bt=vbounds(b,t,bt)   # 2026-11-19: 맨 아래 글자가 잘리던 문제(이미지 95%에서 자름) 고침
    return t,bt,cols,pp
def read_pdf2(pdf, model, cs, tag):
    doc=fitz.open(pdf); out=[]; os.makedirs(f'/content/{tag}',exist_ok=True)
    for p in range(len(doc)):
        im=page_img(doc,p)
        try: t,bt,cols,_=segment3(im)
        except Exception: continue
        if len(cols)<8: continue
        crops=[]
        for k,(a,cc) in enumerate(cols):
            fn=f'/content/{tag}/{p+1:03d}_{k+1:02d}.png'; im.crop((max(0,a-3),t,cc+3,bt)).save(fn); crops.append(fn)
        out.append((p+1,crops,predict(model,cs,crops,DEV)))
    return out
def align_rows(res, faces):
    TRf={f:tri(t) for f,t in faces.items()}; rr=[]
    for p,crops,pr in res:
        q=tri(''.join(pr)); face=max(TRf, key=lambda f: len(q&TRf[f])/(len(TRf[f])+1))
        if len(q&TRf[face])/(len(TRf[face])+1)<=0.2: continue
        gt=G(faces[face]); pa=[]; owner=[]
        for k,s in enumerate(pr): g=G(s); pa+=g; owner+=[k]*len(g)
        amap=align(pa,gt); first={}
        for i,j in enumerate(amap):
            if j is not None: first.setdefault(owner[i],j)
        ks=sorted(first)
        for n,k in enumerate(ks):
            s=0 if (n==0 and first[k]<=3) else first[k]; e=first[ks[n+1]] if n+1<len(ks) else len(gt)
            span=''.join(gt[s:e])
            if len(G(span))>=5: rr.append((crops[k],pr[k],span,face,p))
    return rr

# ---------------- 1) 월봉긔 (사람이 검수한 정답) ----------------
UA={'User-Agent':'KoreanWoodblockOCR/0.1 (https://sites.google.com/view/kangeunkim/home; academic research) python-requests'}
r=requests.get('https://commons.wikimedia.org/w/api.php',params={'action':'query','prop':'imageinfo','iiprop':'url','format':'json','titles':'File:CNTS-00060574842 월봉긔. 일- -著者未詳-.pdf'},headers=UA).json()
url=list(r['query']['pages'].values())[0]['imageinfo'][0]['url']; print('원본 PDF:',url)
url=url.split('?')[0]
import time
for _t in range(5):
    _r=requests.get(url,headers=UA,timeout=180); print('PDF 응답',_r.status_code,len(_r.content))
    if _r.status_code==200 and _r.content[:4]==b'%PDF': break
    time.sleep(10)
else:
    raise SystemExit('PDF를 내려받지 못했습니다. 잠시 후 다시 실행해 보세요.')
pdf=_r.content; open('/content/wb.pdf','wb').write(pdf); print('PDF 크기',len(pdf)//1_000_000,'MB')
D=json.load(open(ROOT+'/4_말뭉치_비공개/wolbong_gt.json',encoding='utf-8'))   # 월봉긔 사람 검수 정답(행 좌표·판독문). 저장소에는 넣지 않음
doc=fitz.open('/content/wb.pdf'); os.makedirs('/content/data/lines',exist_ok=True)
cache={}
for lab,g in D['grid'].items():
    p=g['pdf']-1
    if p not in cache:
        x=doc[p].get_images(full=True)[0][0]; cache={p:Image.open(io.BytesIO(doc.extract_image(x)['image'])).convert('L')}
    im=cache[p]; xs=g['xs']; fl=lab.replace('\u1100','a').replace('\u1102','b')
    for k in range(15):
        x0,x1=xs[15-k-1],xs[15-k]
        im.crop((max(0,x0-6),g['t']-5,x1+6,g['b']+5)).save(f'/content/data/lines/{fl}_{k+1:02d}.png')
        open(f'/content/data/lines/{fl}_{k+1:02d}.gt.txt','w',encoding='utf-8').write(D['lines'][lab][k]+'\n')
print('행 이미지',len(os.listdir('/content/data/lines'))//2,'개 준비 완료')

# ---------------- 2) 경판 원문 PDF 내려받기 ----------------
KA=json.load(open(ROOT+'/4_말뭉치_비공개/kohico_kyeongpan.json',encoding='utf-8'))   # KOHICO 경판 전사(면 단위). 이용 조건상 저장소에 넣지 않음
KG=json.load(open(ROOT+'/4_말뭉치_비공개/kohico_geumhyang.json',encoding='utf-8'))   # KOHICO 금향졍긔 전사(시험용). 저장소에 넣지 않음
SRC=[('ss1','File:CNTS-00082399741 1 샤시남졍긔.pdf','사씨남정기_상_경판32장본_1851_景印古小說板'),
     ('ss2','File:CNTS-00082399741 2 샤시남졍긔.pdf','사씨남정기_하_경판34장본_1851_景印古小說板'),
     ('sy1','File:CNTS-00082399892 1 셔유긔.pdf','서유기_상_경판31장본_1856_景印古小說板'),
     ('sy2','File:CNTS-00082399892 2 셔유긔.pdf','서유기_하_경판28장본_1856_景印古小說板'),
     ('w00','File:CNTS-00047982706 1 곽분양젼.pdf','곽분양전_상_경판본_미상_景印古小說板'),
     ('w01','File:CNTS-00082399731 2 곽분양젼.pdf','곽분양전_중_경판본_미상_景印古小說板'),
     ('w02','File:CNTS-00047982706 2 곽분양젼.pdf','곽분양전_하_경판본_미상_景印古小說板'),
     ('w03','File:CNTS-00082399889 1 삼국지.pdf','삼국지_1_경판본_미상_景印古小說板'),
     ('w04','File:CNTS-00082399889 2 삼국지.pdf','삼국지_2_경판본_미상_景印古小說板'),
     ('w05','File:CNTS-00082399889 3 삼국지.pdf','삼국지_3_경판본_미상_景印古小說板'),
     ('w06','File:심청전(1920).pdf','심청전_단_경판24장본_미상_韓國古典小說'),
     ('w07','File:CNTS-00047982698 쌍쥬긔연.pdf','쌍주기연_경판16장본_미상_景印古小說板'),
     ('w08','File:CNTS-00060574838 옥쥬호연.pdf','옥주호연_단_경판29장본A_미상_韓國古典小說'),
     ('w09','File:CNTS-00047982704 장한졀효기.pdf','장한절효기_단_경판29장본A_미상_景印古小說板'),
     ('w10','File:홍길동전.pdf','홍길동전_단_경판30장본_1860_'),
     ('w11','File:CNTS-00082399730 3 황운젼.pdf','황운전_경판29장본_미상_韓國古典小說'),
     ('w12','File:CNTS-00082399730 1 황운젼.pdf','황운전_경판30장본_미상_韓國古典小說')]
for tag,title,wk in SRC+[('gh1','File:CNTS-00109637790 금향졍긔 이종. 권하.pdf','')]:
    try: get_pdf(title,f'/content/{tag}.pdf')
    except Exception as e: print('PDF 실패',tag,e)
# ---------------- 3) 5차 모델로 읽고 KOHICO 전사에 맞춰 행 자료 만들기 ----------------
with zipfile.ZipFile(ROOT+'/1_모델/gyeongpan_model_v5.zip') as z: z.extract('gyeongpan_model_v5.pt','/content')
m5,cs5=load_model('/content/gyeongpan_model_v5.pt',DEV)
OUTL='/content/data5/lines'; os.makedirs(OUTL,exist_ok=True)
for tag,title,wk in SRC:
    if not os.path.exists(f'/content/{tag}.pdf') or wk not in KA: print('건너뜀',tag); continue
    rows=align_rows(read_pdf2(f'/content/{tag}.pdf',m5,cs5,tag+'_c'),KA[wk]); ok=[r for r in rows if cer([r[1]],[r[2]])<=0.30]
    for r in ok:
        fn=f"{OUTL}/{tag}_{r[3]}_{os.path.basename(r[0])[:-4]}"; shutil.copy(r[0],fn+'.png'); open(fn+'.gt.txt','w',encoding='utf-8').write(r[2]+'\n')
    print(f'{tag} {wk[:26]:26s} 정렬 {len(rows)}행 채택 {len(ok)}행',flush=True)
shutil.make_archive('/content/gyeongpan_lines_v5align','zip','/content/data5'); shutil.copy('/content/gyeongpan_lines_v5align.zip',ROOT+'/2_학습용_행자료/gyeongpan_lines_v5align.zip')
# 금향졍긔 시험지 (뒤쪽 절반 면, 학습에 절대 쓰지 않음)
rr=align_rows(read_pdf2('/content/gh1.pdf',m5,cs5,'gh2'),KG['1'])
pgs=sorted({r[4] for r in rr}); te=[r for r in rr if r[4] in set(pgs[len(pgs)//2:])]
tp2=[r[0] for r in te]; tg2=[r[2] for r in te]
print('금향졍긔 시험',len(te),'행 / 5차',f"{cer(predict(m5,cs5,tp2,DEV),tg2)*100:.1f}%",flush=True)
# ---------------- 4) AI허브 목판본 행 자료 ----------------
os.makedirs('/content/aihub_lines',exist_ok=True)
with zipfile.ZipFile(ROOT+'/3_AI허브_원본/aihub/aihub_mokpan_lines.zip') as z: z.extractall('/content/aihub_lines')
AH=[json.loads(l) for l in open('/content/aihub_lines/lines.jsonl',encoding='utf-8')]
AH=[(f"/content/aihub_lines/{r['img']}",r['text'],r['book']) for r in AH if len(G(r['text']))>=3 and os.path.exists(f"/content/aihub_lines/{r['img']}")]
hold=lambda b: '심청전_04' in b or '춘향전' in b          # 완판으로 보이는 소설은 시험용으로 빼 둠
AHtr=[(p,t) for p,t,b in AH if not hold(b)]; AHte=[(p,t) for p,t,b in AH if hold(b)]
print('AI허브 학습',len(AHtr),'행 / 소설 시험',len(AHte),'행',flush=True)
# ---------------- 5) 학습 자료 나누기 ----------------
P5={'wb':load_pairs('/content/data/lines')}
for p,t in load_pairs(OUTL): P5.setdefault(os.path.basename(p).split('_')[0],[]).append((p,t))
face5=lambda p: '_'.join(os.path.basename(p).split('_')[:2])
facew=lambda p: os.path.basename(p).rsplit('_',1)[0]
VF5={'wb':{'01b','06b','11b','32b'}}
for k in P5:
    if k!='wb': fs=sorted({face5(p) for p,_ in P5[k]}); random.seed(len(k)*7+len(fs)); VF5[k]=set(random.sample(fs,min(4,max(1,len(fs)//6))))
fw=lambda k,p: facew(p) if k=='wb' else face5(p)
KTR=[x for k in P5 for x in P5[k] if fw(k,x[0]) not in VF5[k]]; KVA=[x for k in P5 for x in P5[k] if fw(k,x[0]) in VF5[k]]
print('경판 학습',len(KTR),'행 / 검증',len(KVA),'행',flush=True)
cs7=Charset([t for _,t in KTR+KVA]+[t for _,t,_ in AH]); print('자모',len(cs5.chars),'→',len(cs7.chars),'종',flush=True)
# ---------------- 6) 7차 학습 (5차 가중치에서 시작, 에폭마다 경판 전부 + AI허브 일부) ----------------
random.seed(0); torch.manual_seed(0)
m7=CRNN(len(cs7)).to(DEV); sd=m5.state_dict(); own=m7.state_dict()
for k,v in sd.items():
    if not k.startswith('fc.'): own[k]=v
for c,i in cs5.idx.items():
    if c in cs7.idx: own['fc.weight'][cs7.idx[c]]=sd['fc.weight'][i]; own['fc.bias'][cs7.idx[c]]=sd['fc.bias'][i]
own['fc.weight'][0]=sd['fc.weight'][0]; own['fc.bias'][0]=sd['fc.bias'][0]; m7.load_state_dict(own)
EP=24; NAH=min(len(AHtr),9000); BS=16; LR=6e-4
steps=EP*((len(KTR)+NAH)//BS)
opt=torch.optim.AdamW(m7.parameters(),lr=LR,weight_decay=1e-4); sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=LR,total_steps=steps,pct_start=0.1)
ctc=nn.CTCLoss(blank=0,zero_infinity=True); best=9; log=[]
SAVE='/content/gyeongpan_model_v7.pt'
for ep in range(1,EP+1):
    tr=KTR+random.sample(AHtr,NAH); dl=DataLoader(LineDS(tr,cs7,True),batch_size=BS,shuffle=True,num_workers=2,collate_fn=collate,drop_last=True)
    m7.train(); t0=time.time(); tot=0; n=0
    for x,tg in dl:
        x=x.to(DEV); y=m7(x).log_softmax(-1).permute(1,0,2)
        loss=ctc(y,torch.tensor(sum(tg,[])),torch.full((len(tg),),y.shape[0],dtype=torch.long),torch.tensor([len(t) for t in tg]))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m7.parameters(),5); opt.step()
        if sch.last_epoch+1<steps: sch.step()
        tot+=loss.item(); n+=1
    c=cer(predict(m7,cs7,[p for p,_ in KVA],DEV),[t for _,t in KVA]); log.append((ep,tot/n,c))
    if c<best: best=c; torch.save({'state':m7.state_dict(),'chars':cs7.chars,'H':H},SAVE)
    print(f'epoch {ep:2d}  loss {tot/n:.3f}  경판 검증 {c*100:.1f}% (최고 {best*100:.1f}%)  {time.time()-t0:.0f}s',flush=True)
# ---------------- 7) 시험 ----------------
m7b,cs7b=load_model(SAVE,DEV)
r={'gh_v5':cer(predict(m5,cs5,tp2,DEV),tg2),'gh_v7':cer(predict(m7b,cs7b,tp2,DEV),tg2)}
if AHte:
    sm=random.Random(1).sample(AHte,min(800,len(AHte)))
    r['novel_v5']=cer(predict(m5,cs5,[p for p,_ in sm],DEV),[t for _,t in sm]); r['novel_v7']=cer(predict(m7b,cs7b,[p for p,_ in sm],DEV),[t for _,t in sm])
print('[처음 보는 작품] 금향졍긔 뒤쪽 절반  5차 %.1f%% → 7차 %.1f%%'%(r['gh_v5']*100,r['gh_v7']*100))
if 'novel_v7' in r: print('[AI허브 소설(완판으로 보이는 것)]  5차 %.1f%% → 7차 %.1f%%'%(r['novel_v5']*100,r['novel_v7']*100))
torch.save({'state':m7b.state_dict(),'chars':cs7b.chars,'H':H,'arch':'CRNN'},ROOT+'/1_모델/gyeongpan_model_v7.pt')
json.dump({'result':r,'log':log,'n_kyeong':len(KTR),'n_aihub':len(AHtr)},open(ROOT+'/기록/result_v7.json','w'),ensure_ascii=False)
print('드라이브 저장 끝')
