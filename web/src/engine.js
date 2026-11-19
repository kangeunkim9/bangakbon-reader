// 경판 판독기 엔진: Colab 판독 노트북(②·③)의 행 자르기·전처리·디코딩을 그대로 옮긴 것
(function(){
const KP = {};
const H = 64, MAXW = 1600;

// ---------- 그림 ----------
// gray = {w, h, d: Uint8Array}
KP.grayFromImageData = function(id){
  const n = id.width*id.height, s = id.data, d = new Uint8Array(n);
  for (let i=0,j=0;i<n;i++,j+=4) d[i] = (s[j]*19595 + s[j+1]*38470 + s[j+2]*7471 + 0x8000) >> 16; // PIL convert('L')
  return {w:id.width, h:id.height, d};
};
KP.grayFromCanvasSource = function(src, w, h){
  const c = document.createElement('canvas'); c.width=w; c.height=h;
  const g = c.getContext('2d', {willReadFrequently:true}); g.drawImage(src,0,0,w,h);
  return KP.grayFromImageData(g.getImageData(0,0,w,h));
};

// cv2.adaptiveThreshold(MEAN_C, BINARY_INV, 51, 12) → 먹=1
function binar(im){
  const {w,h,d} = im, r = 25, k = 51;
  const hs = new Int32Array(w*h);
  for (let y=0;y<h;y++){
    const row = y*w; let s = 0;
    for (let i=-r;i<=r;i++) s += d[row + Math.min(w-1, Math.max(0,i))];
    for (let x=0;x<w;x++){
      hs[row+x] = s;
      s += d[row + Math.min(w-1, x+r+1)] - d[row + Math.max(0, x-r)];
    }
  }
  const b = new Uint8Array(w*h), area = k*k;
  for (let x=0;x<w;x++){
    let s = 0;
    for (let i=-r;i<=r;i++) s += hs[Math.min(h-1, Math.max(0,i))*w + x];
    for (let y=0;y<h;y++){
      const m = Math.round(s/area);
      b[y*w+x] = (d[y*w+x] - m <= -12) ? 1 : 0;
      s += hs[Math.min(h-1, y+r+1)*w + x] - hs[Math.max(0, y-r)*w + x];
    }
  }
  return b;
}
// 선 모양 구조 요소로 열기(open). 침식은 바깥=1, 팽창은 바깥=0 (OpenCV 기본값)
function open1d(src, w, h, k, vertical){
  const a = Math.floor(k/2), out = new Uint8Array(w*h);
  const n = vertical ? h : w, m = vertical ? w : h;
  const ero = new Uint8Array(n), zp = new Int32Array(n+1);
  for (let j=0;j<m;j++){
    const at = vertical ? (i)=>i*w+j : (i)=>j*w+i;
    zp[0]=0; for (let i=0;i<n;i++) zp[i+1] = zp[i] + (src[at(i)] ? 0 : 1);
    for (let i=0;i<n;i++){
      const lo = Math.max(0, i-a), hi = Math.min(n-1, i-a+k-1);
      ero[i] = (zp[hi+1]-zp[lo]) === 0 ? 1 : 0;
    }
    // 팽창: 같은 창에서 하나라도 1
    let op = new Int32Array(n+1);
    for (let i=0;i<n;i++) op[i+1] = op[i] + ero[i];
    for (let i=0;i<n;i++){
      const lo = Math.max(0, i-a), hi = Math.min(n-1, i-a+k-1);
      if (hi>=lo && op[hi+1]-op[lo] > 0) out[at(i)] = 1;
    }
  }
  return out;
}
function convSame(x, k){ // np.convolve(x, ones(k)/k, 'same'), k 홀수
  const n = x.length, r = (k-1)>>1, out = new Float64Array(n);
  for (let i=0;i<n;i++){ let s=0; for (let j=i-r;j<=i+r;j++) if (j>=0 && j<n) s += x[j]; out[i] = s/k; }
  return out;
}
function arange(start, stop, step){ const n = Math.max(0, Math.ceil((stop-start)/step)), a = new Array(n); for (let i=0;i<n;i++) a[i]=start+i*step; return a; }
function findPeaks(x){ // scipy.signal.find_peaks (조건 없음)
  const pk=[]; let i=1; const n=x.length;
  while (i<n-1){
    if (x[i-1] < x[i]){
      let ia=i+1; while (ia<n-1 && x[ia]===x[i]) ia++;
      if (x[ia] < x[i]){ pk.push((i + ia - 1) >> 1); i = ia; }
    }
    i++;
  }
  return pk;
}

// 위·아래 경계 다듬기: 괘선이 있으면 괘선 바로 안쪽, 없으면 먹이 끝나는 곳까지 (이미지 5%/95%에서 자르지 않음)
function vbounds(b, bb, h3, W, Hh, t, bt){
  const R=[]; 
  for (let y=0;y<Hh;y++){
    let sb=0, sh=0; for (let x=0;x<W;x++){ sb+=b[y*W+x]; sh+=h3[y*W+x]; }
    if (sb/W>0.5 || sh>Math.floor(W/4)){ if (R.length && y===R[R.length-1][1]+1) R[R.length-1][1]=y; else R.push([y,y]); }
  }
  const RR=R.filter(r=>r[0]>Hh*0.01 && r[1]<Hh*0.99);
  const top=RR.filter(r=>r[1]<Hh*0.35 && r[1]>t-120 && r[1]<t+120), bot=RR.filter(r=>r[0]>Hh*0.65 && r[0]>bt-120 && r[0]<bt+120);
  if (top.length) t=top[top.length-1][1]+2;
  if (bot.length) bt=bot[0][0]-2;
  if (!top.length || !bot.length){
    const rm=new Float64Array(Hh); for (let y=0;y<Hh;y++){ let s=0; for (let x=0;x<W;x++) s+=bb[y*W+x]; rm[y]=s/W; }
    const sm=convSame(Array.from(rm),15); let mx=0; for (let y=t;y<bt;y++) mx=Math.max(mx,sm[y]);
    const th=Math.max(0.01,mx*0.25);
    if (!bot.length){ let y=bt-12; while (y+1<Math.min(Math.trunc(Hh*0.99),bt+80) && sm[y+1]>th) y++; bt=Math.max(bt,Math.min(Hh-1,y+12)); }
    if (!top.length){ let y=t+12; while (y-1>Math.max(Math.trunc(Hh*0.01),t-80) && sm[y-1]>th) y--; t=Math.min(t,Math.max(0,y-12)); }
  }
  return [t, bt];
}

// segment(): 위·아래 경계와 행(세로줄) 목록. 오른쪽 행부터.
KP.segment = function(im){
  const {w:W, h:Hh} = im; const b = binar(im);
  const vl = open1d(b, W, Hh, Math.floor(Hh/4), true), hl = open1d(b, W, Hh, Math.floor(W/4), false);
  const bb = new Uint8Array(W*Hh); for (let i=0;i<bb.length;i++) bb[i] = b[i] & (1 - (vl[i]|hl[i]));
  const h3 = open1d(b, W, Hh, Math.floor(W/3), false);
  const rows=[]; for (let y=0;y<Hh;y++){ let s=0; for (let x=0;x<W;x++) s+=h3[y*W+x]; if (s > Math.floor(W/4)) rows.push(y); }
  const frame = rows.length>=2 && rows[rows.length-1]-rows[0] > Hh*0.5;
  let T, B; if (frame){ T=rows[0]; B=rows[rows.length-1]; } else { T=Math.trunc(Hh*0.05); B=Math.trunc(Hh*0.95); }
  const rm = []; for (let y=T+5;y<B-5;y++){ let s=0; for (let x=0;x<W;x++) s+=bb[y*W+x]; rm.push(s/W); }
  const rp = convSame(rm, 15); let mx=-Infinity; for (const v of rp) mx=Math.max(mx,v);
  const th = Math.max(0.01, mx*0.08); const on=[]; rp.forEach((v,i)=>{ if (v>th) on.push(i); });
  let t = Math.max(T+2, T+5+(on.length?on[0]:0)-12), bt = Math.min(B-2, T+5+(on.length?on[on.length-1]:B-T-10)+12);
  const colsum = new Float64Array(W);
  for (let y=t;y<bt;y++) for (let x=0;x<W;x++) colsum[x]+=bb[y*W+x];
  const prof = convSame(Array.from(colsum, v=>v/(bt-t)), 7);
  const ink=[]; for (let x=0;x<W;x++) if (prof[x]>0.02) ink.push(x);
  if (!ink.length) return {t, bt, cols:[]};
  const x0=ink[0], x1=ink[ink.length-1];
  let mean=0; for (let i=x0;i<x1;i++) mean+=prof[i]; mean/=(x1-x0);
  const p=[]; for (let i=x0;i<x1;i++) p.push(prof[i]-mean);
  // 행 간격은 45px부터 이미지 너비의 1/3까지 찾는다(완판처럼 글자가 큰 판본, 해상도가 높은 사진)
  const LMAX=Math.max(150, Math.floor(W/3));
  const vals=[]; for (let lag=45; lag<LMAX && lag<p.length; lag++){ let s=0; for (let i=0;i+lag<p.length;i++) s+=p[i]*p[i+lag]; vals.push(s); }
  if (!vals.length) return {t, bt, cols:[]};
  let vmax=-Infinity, vam=0; vals.forEach((v,i)=>{ if (v>vmax){ vmax=v; vam=i; } });
  let pk = findPeaks(vals).filter(k=>vals[k]>0.5*vmax); if (!pk.length) pk=[vam];
  const p0 = 45+pk[0]; let best=null;
  for (const pp of arange(p0*0.94, p0*1.06, 0.5)){
    for (const o of arange(0, pp, 1.0)){
      const xs = arange(x0-pp+o, x1+pp, pp).filter(x=>x>=0 && x<W).map(x=>Math.trunc(x));
      let s=0; for (const x of xs) s+=prof[x]; s/=xs.length;
      if (best===null || s<best[0]) best=[s,pp,o];
    }
  }
  const pp=best[1], o=best[2], r=Math.trunc(pp*0.2);
  const xs = arange(x0-pp+o, x1+pp, pp).filter(x=>x>=0 && x<W).map(x=>{
    const lo=Math.max(0,Math.trunc(x)-r), hi=Math.min(W,Math.trunc(x)+r+1);
    let am=lo; for (let i=lo;i<hi;i++) if (prof[i]<prof[am]) am=i;
    return Math.trunc(Math.max(0,x-r)) + (am-lo);
  });
  const cols=[];
  for (let i=0;i+1<xs.length;i++){
    const a=xs[i], c=xs[i+1]; if (!(c-a > pp*0.5)) continue;
    let s=0; for (let x=a;x<c;x++) s+=colsum[x]; if (s>200) cols.push([a,c]);
  }
  // 맨 가장자리 행: 바깥 경계가 이미지 밖이라 빠지던 것. 먹이 넉넉하면 이미지 끝까지를 한 행으로
  let addL=false, addR=false;
  if (cols.length){
    const sum=(a,c)=>{ let s=0; for (let x=a;x<c;x++) s+=colsum[x]; return s; };
    const ss=cols.map(([a,c])=>sum(a,c)).sort((u,v)=>u-v), med=ss.length%2?ss[(ss.length-1)/2]:(ss[ss.length/2-1]+ss[ss.length/2])/2;
    if (xs[0]>pp*0.5){ const a=Math.max(0,Math.round(xs[0]-pp)); if (xs[0]-a>=0.7*pp && sum(a,xs[0])>0.3*med){ cols.unshift([a,xs[0]]); addL=true; } }
    const L=xs[xs.length-1]; if (W-1-L>pp*0.5){ const c=Math.min(W-1,Math.round(L+pp)); if (c-L>=0.7*pp && sum(L,c)>0.3*med){ cols.push([L,c]); addR=true; } }
  }
  if (frame){ t=T+4; bt=B-4; }
  [t, bt] = vbounds(b, bb, h3, W, Hh, t, bt);
  return {t, bt, cols: cols.reverse(), pitch: pp, edge:{first:addR, last:addL}};   // 뒤집었으므로 오른쪽 끝이 첫 행
};

// 한 행 잘라 90도 돌리고 autocontrast(1%) → 높이 64로 줄이기 (PIL 방식 그대로)
function cropRot(im, x0, y0, x1, y1){ // PIL crop: 바깥은 0
  const w0=x1-x0, h0=y1-y0, out=new Uint8Array(w0*h0); // 돌린 그림: 너비 h0, 높이 w0
  for (let yp=0; yp<w0; yp++){
    const xs = x0 + (w0-1-yp);
    for (let xp=0; xp<h0; xp++){
      const ys = y0 + xp;
      out[yp*h0+xp] = (xs>=0 && xs<im.w && ys>=0 && ys<im.h) ? im.d[ys*im.w+xs] : 0;
    }
  }
  return {w:h0, h:w0, d:out};
}
function autocontrast(im){
  const h=new Array(256).fill(0); for (const v of im.d) h[v]++;
  const n=im.d.length; let cut=Math.floor(n*1/100);
  for (let lo=0;lo<256;lo++){ if (cut>h[lo]){ cut-=h[lo]; h[lo]=0; } else { h[lo]-=cut; cut=0; } if (cut<=0) break; }
  cut=Math.floor(n*1/100);
  for (let hi=255;hi>=0;hi--){ if (cut>h[hi]){ cut-=h[hi]; h[hi]=0; } else { h[hi]-=cut; cut=0; } if (cut<=0) break; }
  let lo=0; for (;lo<256;lo++) if (h[lo]) break;
  let hi=255; for (;hi>=0;hi--) if (h[hi]) break;
  const lut=new Uint8Array(256);
  if (hi<=lo) for (let i=0;i<256;i++) lut[i]=i;
  else { const sc=255/(hi-lo), off=-lo*sc; for (let i=0;i<256;i++){ let v=Math.trunc(i*sc+off); lut[i]=v<0?0:v>255?255:v; } }
  const d=new Uint8Array(n); for (let i=0;i<n;i++) d[i]=lut[im.d[i]];
  return {w:im.w, h:im.h, d};
}
function coeffs(inSize, outSize){
  const scale=inSize/outSize, fs=Math.max(scale,1), support=fs, ss=1/fs, list=[];
  for (let xx=0;xx<outSize;xx++){
    const center=(xx+0.5)*scale;
    let xmin=Math.trunc(center-support+0.5); if (xmin<0) xmin=0;
    let xmax=Math.trunc(center+support+0.5); if (xmax>inSize) xmax=inSize; xmax-=xmin;
    const k=[]; let ww=0;
    for (let x=0;x<xmax;x++){ let t=Math.abs((x+xmin-center+0.5)*ss); const v=t<1?1-t:0; k.push(v); ww+=v; }
    const ki=k.map(v=>{ v = ww!==0 ? v/ww : v; const f=v*(1<<22); return f<0 ? Math.trunc(-0.5+f) : Math.trunc(0.5+f); });
    list.push([xmin, ki]);
  }
  return list;
}
function resizeBilinear(im, ow, oh){ // Pillow ImagingResample, 8비트: 가로 → 세로
  let cur=im;
  if (ow!==im.w){
    const cf=coeffs(im.w, ow), d=new Uint8Array(ow*im.h);
    for (let y=0;y<im.h;y++) for (let x=0;x<ow;x++){
      const [xm,k]=cf[x]; let s=1<<21; for (let i=0;i<k.length;i++) s+=im.d[y*im.w+xm+i]*k[i];
      let v=Math.floor(s/4194304); d[y*ow+x]=v<0?0:v>255?255:v;
    }
    cur={w:ow,h:im.h,d};
  }
  if (oh!==cur.h){
    const cf=coeffs(cur.h, oh), d=new Uint8Array(ow*oh);
    for (let y=0;y<oh;y++){ const [ym,k]=cf[y];
      for (let x=0;x<ow;x++){ let s=1<<21; for (let i=0;i<k.length;i++) s+=cur.d[(ym+i)*ow+x]*k[i];
        let v=Math.floor(s/4194304); d[y*ow+x]=v<0?0:v>255?255:v; } }
    cur={w:ow,h:oh,d};
  }
  return cur;
}
KP.prepLine = function(im, a, c, t, bt){
  let r = cropRot(im, Math.max(0,a-3), t, c+3, bt);
  r = autocontrast(r);
  const nw = Math.min(MAXW, Math.max(32, Math.trunc(r.w*H/r.h)));
  r = resizeBilinear(r, nw, H);
  const f = new Float32Array(nw*H); for (let i=0;i<f.length;i++) f[i]=1-r.d[i]/255;
  return {data:f, w:nw};
};

// ---------- 디코딩 ----------
KP.logSoftmax = function(y, T, C){
  const out = new Float32Array(T*C);
  for (let t=0;t<T;t++){ let m=-Infinity; for (let c=0;c<C;c++) m=Math.max(m,y[t*C+c]);
    let s=0; for (let c=0;c<C;c++) s+=Math.exp(y[t*C+c]-m); const l=m+Math.log(s);
    for (let c=0;c<C;c++) out[t*C+c]=y[t*C+c]-l; }
  return out;
};
KP.greedy = function(lp, T, C, chars){
  let prev=0, o='';
  for (let t=0;t<T;t++){ let k=0, m=-Infinity; for (let c=0;c<C;c++) if (lp[t*C+c]>m){ m=lp[t*C+c]; k=c; }
    if (k!==prev && k!==0) o+=chars[k-1]; prev=k; }
  return o.normalize('NFC');
};
const NEG=-1e30;
function lse(a,b){ if (a<-1e29) return b; if (b<-1e29) return a; const m=Math.max(a,b); return m+Math.log(Math.exp(a-m)+Math.exp(b-m)); }
KP.makeLM = function(js){
  const ctx=js.ctx, V=js.V, N=js.N, memo=new Map();
  function lmp(h,c){
    const key=h+'\u0001'+c; const hit=memo.get(key); if (hit!==undefined) return hit;
    let v; const e=ctx[h];
    if (h===''){ v=((e[2][c]||0)+e[1]/V)/(e[0]+e[1]); }
    else { const lo=lmp(h.slice(1),c); v = (!e || e[0]===0) ? lo : ((e[2][c]||0)+e[1]*lo)/(e[0]+e[1]); }
    if (memo.size>2000000) memo.clear(); memo.set(key,v); return v;
  }
  return {lmp, N};
};
KP.beam = function(lp, T, C, chars, LM, alpha=0.3, beta=1.0, Bw=10){
  const thr=Math.log(1e-3), pad='^'.repeat(LM.N-1);
  let beams=new Map([['',[0.0,NEG]]]);
  for (let t=0;t<T;t++){
    const o=t*C, cand=[]; for (let i=1;i<C;i++) if (lp[o+i]>thr) cand.push(i);
    const nb=new Map(); const get=(k)=>{ let e=nb.get(k); if (!e){ e=[NEG,NEG]; nb.set(k,e); } return e; };
    for (const [pre,[pb,pn]] of beams){
      const tot=lse(pb,pn); let e=get(pre); e[0]=lse(e[0], tot+lp[o]); const last=pre.length?pre[pre.length-1]:null;
      const ctx=(pad+pre).slice(-(LM.N-1));
      for (const i of cand){
        const ch=chars[i-1], lc=lp[o+i];
        if (ch===last){
          e=get(pre); e[1]=lse(e[1], pn+lc);
          if (pb>NEG){ const f=get(pre+ch); f[1]=lse(f[1], pb+lc+alpha*Math.log(LM.lmp(ctx,ch))+beta); }
        } else { const f=get(pre+ch); f[1]=lse(f[1], tot+lc+alpha*Math.log(LM.lmp(ctx,ch))+beta); }
      }
    }
    beams=new Map([...nb.entries()].map(([k,v])=>[k,v,lse(v[0],v[1])]).sort((a,b)=>b[2]-a[2]).slice(0,Bw).map(([k,v])=>[k,v]));
  }
  let bk='', bs=-Infinity; for (const [k,v] of beams){ const s=lse(v[0],v[1]); if (s>bs){ bs=s; bk=k; } }
  return bk.normalize('NFC');
};
KP.graphemes = function(s){
  try { return [...new Intl.Segmenter('ko',{granularity:'grapheme'}).segment(s)].length; } catch(e){ return s.length; }
};
// 면 사진 너비를 학습 때(위키미디어 공용 스캔, 약 1500px)와 비슷하게 맞춘다
KP.normWidth = function(im){
  if (im.w>1700 || im.w<1000){ const ow=1500, oh=Math.max(1,Math.round(im.h*ow/im.w)); return resizeBilinear(im, ow, oh); }
  return im;
};
KP.H=H; KP.MAXW=MAXW;
window.KP = KP;
})();
