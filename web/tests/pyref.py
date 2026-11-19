import json, gzip, math, collections, functools, unicodedata, numpy as np, cv2, onnxruntime as ort
from PIL import Image, ImageOps
from scipy.signal import find_peaks
H=64; MAXW=1600
nfc=lambda s: unicodedata.normalize('NFC',s)
cs=json.load(open('../site/charset.json'))['chars']
def binar(im):
    g=np.asarray(im.convert('L')); return (cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,51,12)>0).astype(np.uint8)
exec(open('seg.py').read())
def prep(im):
    im=ImageOps.autocontrast(im.convert('L').rotate(90,expand=True),cutoff=1); w,h=im.size
    im=im.resize((min(MAXW,max(32,int(w*H/h))),H),Image.BILINEAR); return 1.0-np.asarray(im,dtype=np.float32)/255.0
s=ort.InferenceSession('model_v5_full.onnx',providers=['CPUExecutionProvider'])
def lps(crop):
    a=prep(crop); y=s.run(None,{'x':a[None,None].astype(np.float32)})[0][0]
    y=y-y.max(1,keepdims=True); return y-np.log(np.exp(y).sum(1,keepdims=True)), a
def greedy(lp):
    k=lp.argmax(1); o=[]; prev=0
    for c in k:
        if c!=prev and c!=0: o.append(cs[c-1])
        prev=c
    return nfc(''.join(o))
L=json.loads(gzip.open('../site/lm7.gz.wasm').read()); ctx=L['ctx']; V=L['V']; NG=L['N']
@functools.lru_cache(None)
def lmp(h,c):
    e=ctx.get(h)
    if h=='': return (e[2].get(c,0)+e[1]/V)/(e[0]+e[1])
    lo=lmp(h[1:],c)
    return lo if not e or e[0]==0 else (e[2].get(c,0)+e[1]*lo)/(e[0]+e[1])
def lse(a,b):
    if a<-1e29: return b
    if b<-1e29: return a
    m=max(a,b); return m+math.log(math.exp(a-m)+math.exp(b-m))
def beam(lp,alpha=0.3,beta=1.0,Bw=10,thr=math.log(1e-3)):
    NEG=-1e30; beams={'':(0.0,NEG)}
    for t in range(lp.shape[0]):
        row=lp[t]; cand=[i for i in range(1,len(row)) if row[i]>thr]; nb=collections.defaultdict(lambda:[NEG,NEG])
        for pre,(pb,pn) in beams.items():
            tot=lse(pb,pn); e=nb[pre]; e[0]=lse(e[0],tot+row[0]); last=pre[-1] if pre else None
            for i in cand:
                ch=cs[i-1]; lc=row[i]; cx=('^'*(NG-1)+pre)[-(NG-1):]
                if ch==last:
                    e=nb[pre]; e[1]=lse(e[1],pn+lc)
                    if pb>NEG: f=nb[pre+ch]; f[1]=lse(f[1],pb+lc+alpha*math.log(lmp(cx,ch))+beta)
                else:
                    f=nb[pre+ch]; f[1]=lse(f[1],tot+lc+alpha*math.log(lmp(cx,ch))+beta)
        beams=dict(sorted(nb.items(),key=lambda kv:-lse(*kv[1]))[:Bw])
    return nfc(max(beams.items(),key=lambda kv:lse(*kv[1]))[0])
im=Image.open('sample.jpg').convert('L')
t,bt,cols=segment(im)
out={'t':int(t),'bt':int(bt),'cols':[[int(a),int(c)] for a,c in cols],'greedy':[],'beam':[],'prep0':None}
for k,(a,c) in enumerate(cols):
    lp,a_=lps(im.crop((max(0,a-3),t,c+3,bt)))
    if k==0: out['prep0']=[a_.shape[1], float(a_.sum()), [float(v) for v in a_[30,:20]]]; out['lp0']=[float(v) for v in lp[:3].ravel()[:10]]
    out['greedy'].append(greedy(lp)); out['beam'].append(beam(lp))
json.dump(out,open('pyref_out.json','w'),ensure_ascii=False)
ref=json.load(open('sample_ref.json'))
print('cols same as colab:', out['cols']==ref['cols'], out['t'],out['bt'], ref['t'],ref['bt'])
print('greedy==colab', sum(a==b for a,b in zip(out['greedy'],ref['greedy'])),'/',len(ref['greedy']))
print('beam==colab', sum(a==b for a,b in zip(out['beam'],ref['beam'])),'/',len(ref['beam']))
for a,b in zip(out['beam'],ref['beam']):
    if a!=b: print(' local:',a,'\n colab:',b)
