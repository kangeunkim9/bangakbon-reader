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

def segment(im):
    b=binar(im); H_,W=b.shape
    lines=cv2.morphologyEx(b,cv2.MORPH_OPEN,np.ones((H_//4,1),np.uint8))|cv2.morphologyEx(b,cv2.MORPH_OPEN,np.ones((1,W//4),np.uint8))
    hr=cv2.morphologyEx(b,cv2.MORPH_OPEN,np.ones((1,W//3),np.uint8)).sum(1); rows=np.where(hr>W//4)[0]; rows=np.array([y for r in _runs(rows) if r[0]>H_*0.01 and r[1]<H_*0.99 for y in range(r[0],r[1]+1)],dtype=int)   # 사진 가장자리 띠 제외
    frame=len(rows)>=2 and rows.max()-rows.min()>H_*0.5
    T,B=(int(rows.min()),int(rows.max())) if frame else (int(H_*0.05),int(H_*0.95))
    bb=b&(1-lines)
    rp=np.convolve(bb[T+5:B-5].mean(1),np.ones(15)/15,'same'); on=np.where(rp>max(0.01,rp.max()*0.08))[0]
    t=max(T+2,T+5+(on.min() if len(on) else 0)-12); bt=min(B-2,T+5+(on.max() if len(on) else B-T-10)+12)
    band=bb[t:bt]; prof=np.convolve(band.mean(0),np.ones(7)/7,'same')
    ink=np.where(prof>0.02)[0]
    if len(ink)==0: return t,bt,[]
    x0,x1=int(ink.min()),int(ink.max())
    p=prof[x0:x1]-prof[x0:x1].mean(); ac=np.correlate(p,p,'full')[len(p)-1:]
    # 2026-11-19: 행 간격을 너비의 1/3까지 찾음(큰 글자·고해상도 사진)
    vals=ac[45:max(150,W//3)]; pk,_=find_peaks(vals); pk=[k for k in pk if vals[k]>0.5*vals.max()] or [int(np.argmax(vals))]
    p0=float(45+pk[0]); best=None
    for pp in np.arange(p0*0.94,p0*1.06,0.5):
        for o in np.arange(0,pp,1.0):
            xs=np.arange(x0-pp+o,x1+pp,pp); xs=xs[(xs>=0)&(xs<W)].astype(int); s=prof[xs].mean()
            if best is None or s<best[0]: best=(s,pp,o)
    _,pp,o=best; xs=np.arange(x0-pp+o,x1+pp,pp); xs=xs[(xs>=0)&(xs<W)]; r=int(pp*0.2)
    xs=[int(max(0,x-r)+np.argmin(prof[max(0,int(x)-r):min(W,int(x)+r+1)])) for x in xs]
    cols=[(a,c) for a,c in zip(xs[:-1],xs[1:]) if c-a>pp*0.5 and band[:,a:c].sum()>200]
    # 맨 가장자리 행: 바깥 경계가 이미지 밖이라 빠지던 것. 먹이 넉넉하면 이미지 끝까지를 한 행으로
    if cols:
        med=float(np.median([band[:,a:c].sum() for a,c in cols]))
        if xs[0]>pp*0.5:
            a=max(0,int(np.floor(xs[0]-pp+0.5)))
            if xs[0]-a>=0.7*pp and band[:,a:xs[0]].sum()>0.3*med: cols.insert(0,(a,xs[0]))
        if W-1-xs[-1]>pp*0.5:
            c=min(W-1,int(np.floor(xs[-1]+pp+0.5)))
            if c-xs[-1]>=0.7*pp and band[:,xs[-1]:c].sum()>0.3*med: cols.append((xs[-1],c))
    if frame: t,bt=T+4,B-4
    t,bt=vbounds(b,t,bt)
    return t,bt,cols[::-1]


# ---------- 필사본용 휜 행 경계 (engine.js의 KP.seams, KP.prepLineCurved와 같은 방식) ----------
def seams(g, t, bt, cols):
    """곧은 경계를 출발점으로, 먹을 덜 지나는 휜 경계(위→아래 단조, 서로 안 겹침)를 동적 계획법으로 찾는다."""
    b=binar(g); Hh,W=b.shape
    lines=cv2.morphologyEx(b,cv2.MORPH_OPEN,np.ones((Hh//4,1),np.uint8))
    ink=cv2.GaussianBlur((b&(1-lines)).astype(np.float32),(0,0),2.5)
    xs=sorted({x for ac in cols for x in ac}); pitch=np.median(np.diff(xs)) if len(xs)>1 else 80
    paths=[]
    for i,x in enumerate(xs):
        lo=int(max(0,x-0.4*pitch, (xs[i-1]+x)/2+1 if i else 0)); hi=int(min(W-1,x+0.4*pitch,(xs[i+1]+x)/2-1 if i+1<len(xs) else W-1))
        C=ink[t:bt,lo:hi+1]; n=C.shape[1]; D=C[0].copy()+0.002*np.abs(np.arange(lo,hi+1)-x); P=np.zeros(C.shape,np.int8)
        for y in range(1,C.shape[0]):
            cand=np.stack([np.r_[np.inf,D[:-1]]+0.03, D, np.r_[D[1:],np.inf]+0.03])
            k=cand.argmin(0); D=cand[k,np.arange(n)]+C[y]; P[y]=k-1
        j=int(D.argmin()); path=np.zeros(C.shape[0],int)
        for y in range(C.shape[0]-1,-1,-1): path[y]=j; j+=P[y,j]
        paths.append(path+lo)
    return xs, paths, b

def crop_curved(g, t, bt, L, R, pad=3):
    """휜 두 경계 사이를 잘라, 행 가운데가 곧게 서도록 줄마다 옮겨 붙인다. 경계 밖은 흰색."""
    a=np.asarray(g); c=(L+R)/2.0; k=max(31,(len(c)//8)|1)
    cs=np.convolve(np.pad(c,(k//2,k//2),mode='edge'),np.ones(k)/k,'valid')
    half=int(np.max(np.maximum(cs-L,R-cs)))+pad; w=2*half+1
    out=np.full((len(c),w),255,np.uint8)
    for y in range(len(c)):
        x0=int(round(cs[y]))-half; l=max(L[y]-pad,0); r=min(R[y]+pad,a.shape[1]-1)
        xa,xb=max(x0,l),min(x0+w-1,r)
        if xb>=xa: out[y,xa-x0:xb-x0+1]=a[t+y,xa:xb+1]
    return Image.fromarray(out)
