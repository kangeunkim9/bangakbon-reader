// ---------- 옛한글 자판 (두벌식 + 옛글자 규칙) ----------
// 한/영 상태와 상관없이 치게 하려고, 키 입력은 화면 밖의 password 칸(브라우저가 한글 입력기를 끄는 칸)으로 받고
// 행 글자와 커서는 직접 그린다. 자판을 끄면 행이 보통 편집 칸(contenteditable)으로 돌아간다.
const IME={on:true, comp:null, tx:null, pos0:0, len:0, hist:[]};
const ED={tx:null, pos:0, anc:0, drag:false, undo:new Map()};
const COARSE = matchMedia('(pointer:coarse)').matches && !matchMedia('(any-pointer:fine)').matches;
try { const v=localStorage.getItem('oldIME'); if (v!==null) IME.on=v==='1'; } catch(e){}
if (COARSE) IME.on=false;   // 화면 자판만 있는 기기에서는 기기 자판을 그대로 쓴다
const KM={KeyQ:['ㅂ','ㅃ'],KeyW:['ㅈ','ㅉ'],KeyE:['ㄷ','ㄸ'],KeyR:['ㄱ','ㄲ'],KeyT:['ㅅ','ㅆ'],KeyY:['ㅛ','ㅛ'],KeyU:['ㅕ','ㅕ'],KeyI:['ㅑ','ㅑ'],KeyO:['ㅐ','ㅒ'],KeyP:['ㅔ','ㅖ'],
  KeyA:['ㅁ','ㅿ'],KeyS:['ㄴ','ㄴ'],KeyD:['ㅇ','ㆁ'],KeyF:['ㄹ','ㄹ'],KeyG:['ㅎ','ㆆ'],KeyH:['ㅗ','ㅗ'],KeyJ:['ㅓ','ㅓ'],KeyK:['ㅏ','ㆍ'],KeyL:['ㅣ','ㅣ'],
  KeyZ:['ㅋ','ㅋ'],KeyX:['ㅌ','ㅌ'],KeyC:['ㅊ','ㅊ'],KeyV:['ㅍ','ㅍ'],KeyB:['ㅠ','ㅠ'],KeyN:['ㅜ','ㅜ'],KeyM:['ㅡ','ㅡ']};
const CONS='ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅿㆁㆆ';
const LOF={'ㄱ':'ᄀ','ㄲ':'ᄁ','ㄴ':'ᄂ','ㄷ':'ᄃ','ㄸ':'ᄄ','ㄹ':'ᄅ','ㅁ':'ᄆ','ㅂ':'ᄇ','ㅃ':'ᄈ','ㅅ':'ᄉ','ㅆ':'ᄊ','ㅇ':'ᄋ','ㅈ':'ᄌ','ㅉ':'ᄍ','ㅊ':'ᄎ','ㅋ':'ᄏ','ㅌ':'ᄐ','ㅍ':'ᄑ','ㅎ':'ᄒ','ㅿ':'ᅀ','ㆁ':'ᅌ','ㆆ':'ᅙ',
  'ㅂㅇ':'ᄫ','ㅁㅇ':'ᄝ','ㅃㅇ':'ᄬ','ㅍㅇ':'ᅗ','ㅅㄱ':'ᄭ','ㅅㄷ':'ᄯ','ㅅㅂ':'ᄲ','ㅅㅈ':'ᄶ','ㅂㅅ':'ᄡ','ㅂㄷ':'ᄠ','ㅂㅈ':'ᄧ','ㅂㅌ':'ᄩ','ㅂㄱ':'ᄞ','ㅂㅅㄱ':'ᄢ','ㅂㅅㄷ':'ᄣ'};
const TOF={'ㄱ':'ᆨ','ㄲ':'ᆩ','ㄴ':'ᆫ','ㄷ':'ᆮ','ㄹ':'ᆯ','ㅁ':'ᆷ','ㅂ':'ᆸ','ㅅ':'ᆺ','ㅆ':'ᆻ','ㅇ':'ᆼ','ㅈ':'ᆽ','ㅊ':'ᆾ','ㅋ':'ᆿ','ㅌ':'ᇀ','ㅍ':'ᇁ','ㅎ':'ᇂ','ㅿ':'ᇫ','ㆁ':'ᇰ','ㆆ':'ᇹ',
  'ㄱㅅ':'ᆪ','ㄴㅈ':'ᆬ','ㄴㅎ':'ᆭ','ㄹㄱ':'ᆰ','ㄹㅁ':'ᆱ','ㄹㅂ':'ᆲ','ㄹㅅ':'ᆳ','ㄹㅌ':'ᆴ','ㄹㅍ':'ᆵ','ㄹㅎ':'ᆶ','ㅂㅅ':'ᆹ','ㄹㆆ':'ᇙ','ㄹㅿ':'ᇗ'};
const VOF={'ㅏ':'ᅡ','ㅐ':'ᅢ','ㅑ':'ᅣ','ㅒ':'ᅤ','ㅓ':'ᅥ','ㅔ':'ᅦ','ㅕ':'ᅧ','ㅖ':'ᅨ','ㅗ':'ᅩ','ㅛ':'ᅭ','ㅜ':'ᅮ','ㅠ':'ᅲ','ㅡ':'ᅳ','ㅣ':'ᅵ','ㆍ':'ᆞ'};
const VCOMB={'ᅩᅡ':'ᅪ','ᅩᅢ':'ᅫ','ᅩᅵ':'ᅬ','ᅮᅥ':'ᅯ','ᅮᅦ':'ᅰ','ᅮᅵ':'ᅱ','ᅳᅵ':'ᅴ','ᅡᅡ':'ᆞ','ᆞᅡ':'ᆢ','ᆞᆞ':'ᆢ','ᆞᅵ':'ᆡ'};
const VDISP={'ᅡ':'ㅏ','ᅢ':'ㅐ','ᅣ':'ㅑ','ᅤ':'ㅒ','ᅥ':'ㅓ','ᅦ':'ㅔ','ᅧ':'ㅕ','ᅨ':'ㅖ','ᅩ':'ㅗ','ᅪ':'ㅘ','ᅫ':'ㅙ','ᅬ':'ㅚ','ᅭ':'ㅛ','ᅮ':'ㅜ','ᅯ':'ㅝ','ᅰ':'ㅞ','ᅱ':'ㅟ','ᅲ':'ㅠ','ᅳ':'ㅡ','ᅴ':'ㅢ','ᅵ':'ㅣ','ᆞ':'ㆍ','ᆡ':'ㆎ'};

// --- 행 글자 읽고 쓰기 (자판 켬: 직접 그림 / 자판 끔: contenteditable) ---
const lineOf=tx=>state.pages[state.cur].lines[+tx.closest('li').dataset.k];
function getT(tx){ return IME.on ? lineOf(tx).text : tx.textContent; }
function setT(tx,v,pos){
  if (IME.on){ const L=lineOf(tx); pushUndo(tx,L.text); L.text=v; ED.tx=tx; ED.pos=ED.anc=pos; paint(tx); lineChanged(tx); }
  else { tx.textContent=v; setCaretCE(tx,pos); tx.dispatchEvent(new Event('input')); }
}
function lineChanged(tx){ const L=lineOf(tx); tx.closest('li').classList.toggle('edited', L.text!==L.orig); renderEditInfo(); }
function pushUndo(tx,old){ const k=tx.closest('li').dataset.k+'/'+state.cur; const st=ED.undo.get(k)||[]; if (st[st.length-1]!==old){ st.push(old); if (st.length>200) st.shift(); } ED.undo.set(k,st); }
function caretIn(tx){ return IME.on ? (ED.tx===tx ? ED.pos : null) : caretInCE(tx); }
function setCaret(tx,pos){ if (IME.on){ ED.tx=tx; ED.pos=ED.anc=pos; paint(tx); } else setCaretCE(tx,pos); }
function caretInCE(tx){
  const sel=getSelection(); if (!sel.rangeCount) return null; const r=sel.getRangeAt(0);
  if (!tx.contains(r.startContainer)) return null;
  const pre=document.createRange(); pre.selectNodeContents(tx); pre.setEnd(r.startContainer,r.startOffset);
  return pre.toString().length;
}
function setCaretCE(tx,pos){
  tx.normalize(); let node=tx.firstChild; if (!node){ node=document.createTextNode(''); tx.appendChild(node); }
  const r=document.createRange(); r.setStart(node,Math.min(pos,node.length)); r.collapse(true); const sel=getSelection(); sel.removeAllRanges(); sel.addRange(r);
}
// 행을 그린다: [앞][고른 부분][커서][뒤]. 조각마다 data-o에 시작 위치를 적어 마우스 위치를 글자 위치로 바꾼다.
function paint(tx){
  const t=lineOf(tx).text, on=ED.tx===tx && document.activeElement===$('kbd');
  tx.textContent='';
  const a=on?Math.min(ED.pos,ED.anc):t.length, b=on?Math.max(ED.pos,ED.anc):t.length;
  const seg=(s,o,cls)=>{ const e=document.createElement('span'); e.textContent=s; e.dataset.o=o; if (cls) e.className=cls; tx.appendChild(e); };
  const cut=[[0,a,''],[a,b,'sel']], rest=[b,t.length,''];
  for (const [s,e,c] of cut) if (e>s) seg(t.slice(s,e),s,c);
  if (on){ const c=document.createElement('span'); c.className='caret'+(IME.comp?' comp':''); c.setAttribute('aria-hidden','true');
    if (ED.pos===a) tx.insertBefore(c, tx.querySelector('.sel')); else tx.appendChild(c); }
  if (rest[1]>rest[0]) seg(t.slice(rest[0]),rest[0],'');
  if (on && IME.comp){ const s=IME.pos0, e=IME.pos0+IME.len; tx.querySelectorAll('span[data-o]').forEach(x=>{ const o=+x.dataset.o; if (o<e && o+x.textContent.length>s) x.classList.add('compseg'); }); }
}
function posFromPoint(tx,x,y){
  let node=null, off=0;
  if (document.caretPositionFromPoint){ const p=document.caretPositionFromPoint(x,y); if (p){ node=p.offsetNode; off=p.offset; } }
  else if (document.caretRangeFromPoint){ const r=document.caretRangeFromPoint(x,y); if (r){ node=r.startContainer; off=r.startOffset; } }
  const t=lineOf(tx).text;
  if (!node || !tx.contains(node)) { const r=tx.getBoundingClientRect(); const vert=getComputedStyle(tx).writingMode.startsWith('vertical'); return (vert ? y<r.top+r.height/2 : x<r.left+r.width/2) ? 0 : t.length; }
  const el=node.nodeType===3?node.parentElement:node; if (el.dataset && el.dataset.o!==undefined) return snap(t, +el.dataset.o + (node.nodeType===3?off:0));
  return t.length;
}
const SEG=(()=>{ try { return new Intl.Segmenter('ko',{granularity:'grapheme'}); } catch(e){ return null; } })();
function bounds(t){ const b=[0]; if (SEG){ for (const s of SEG.segment(t)) b.push(s.index+s.segment.length); } else for (let i=1;i<=t.length;i++) b.push(i); return b; }
function snap(t,p){ const b=bounds(t); let best=0; for (const x of b) if (Math.abs(x-p)<Math.abs(best-p)) best=x; return best; }
function stepG(t,p,d){ const b=bounds(t); if (d<0){ let r=0; for (const x of b) if (x<p) r=x; return r; } for (const x of b) if (x>p) return x; return t.length; }

// --- 두벌식 조합 ---
function imeRender(c){
  if (!c) return '';
  const L=c.L?LOF[c.L]:'', T=c.T?TOF[c.T]:'';
  if (c.L && c.V) return (L+c.V+T).normalize('NFC');
  if (c.L) return c.L.length===1 ? c.L : L+'ᅠ';
  return VDISP[c.V] || ('ᅟ'+c.V);
}
function imeDraw(){
  const tx=IME.tx; if (!tx) return; const t=getT(tx), r=imeRender(IME.comp);
  const v=t.slice(0,IME.pos0)+r+t.slice(IME.pos0+IME.len); IME.len=r.length; setT(tx,v,IME.pos0+r.length);
}
function imeCommit(){ const had=!!IME.comp; IME.comp=null; IME.hist=[]; IME.len=0; if (had && IME.on && ED.tx) paint(ED.tx); }
function imeStart(tx,c){
  imeCommit(); IME.tx=tx; let p=caretIn(tx); if (p===null) p=getT(tx).length; IME.pos0=p; IME.len=0; IME.comp=c; IME.hist=[]; imeDraw();
}
function imePush(c){ IME.hist.push(IME.comp); IME.comp=c; imeDraw(); }
function imeNext(c){ IME.pos0+=IME.len; IME.len=0; IME.comp=c; IME.hist=[]; imeDraw(); }
function imeKey(tx,s){
  const c=IME.comp && IME.tx===tx ? IME.comp : null; const isC=CONS.includes(s);
  if (!c){ imeStart(tx, isC?{L:s}:{V:VOF[s]}); return; }
  if (isC){
    if (!c.V){ if (c.L && LOF[c.L+s]) imePush({L:c.L+s}); else imeNext({L:s}); return; }
    if (!c.T){ if (c.L && TOF[s]) imePush({...c,T:s}); else imeNext({L:s}); return; }
    if (c.T.length===1 && TOF[c.T+s]) imePush({...c,T:c.T+s}); else imeNext({L:s});
    return;
  }
  const v=VOF[s];
  if (!c.V){ imePush({...c,V:v}); return; }
  if (!c.T){ if (VCOMB[c.V+v]) imePush({...c,V:VCOMB[c.V+v]}); else imeNext({V:v}); return; }
  // 받침 뒤에 모음: 마지막 자음을 다음 글자 첫소리로
  const last=c.T.slice(-1), keep=c.T.slice(0,-1); IME.hist.push(IME.comp); IME.comp={...c,T:keep||undefined}; if (!keep) delete IME.comp.T; imeDraw(); imeNext({L:last,V:v});
}
function imeBack(){
  if (!IME.comp) return false;
  if (IME.hist.length){ IME.comp=IME.hist.pop(); imeDraw(); }
  else { IME.comp=null; imeDraw(); imeCommit(); }
  return true;
}

// --- 편집 동작 (자판 켬) ---
function delSel(){ const tx=ED.tx, t=getT(tx), a=Math.min(ED.pos,ED.anc), b=Math.max(ED.pos,ED.anc); if (a===b) return false; setT(tx,t.slice(0,a)+t.slice(b),a); return true; }
function insertText(s){ const tx=ED.tx; if (!tx) return; imeCommit(); delSel(); s=s.replace(/[\r\n\t]+/g,'').normalize('NFC'); const t=getT(tx), p=ED.pos; setT(tx,t.slice(0,p)+s+t.slice(p),p+s.length); }
function selText(){ const t=getT(ED.tx), a=Math.min(ED.pos,ED.anc), b=Math.max(ED.pos,ED.anc); return t.slice(a,b); }
function focusLine(tx,pos){
  if (!tx) return; imeCommit(); const old=ED.tx; ED.tx=tx; const t=getT(tx); ED.pos=ED.anc=Math.max(0,Math.min(pos===undefined?t.length:pos,t.length));
  const k=+tx.closest('li').dataset.k; if (active!==k) select(k,false);
  const kb=$('kbd'); if (document.activeElement!==kb) kb.focus({preventScroll:true});
  if (old && old!==tx && old.isConnected) paint(old); paint(tx); $('oldk').hidden=false;
}
function moveTo(p,ext){ imeCommit(); ED.pos=p; if (!ext) ED.anc=p; paint(ED.tx); }
function kbdKeydown(e){
  const tx=ED.tx; if (!tx || !tx.isConnected) return;
  const t=getT(tx), mod=e.ctrlKey||e.metaKey;
  if (mod){
    const k=e.key.toLowerCase();
    if (k==='a'){ e.preventDefault(); imeCommit(); ED.anc=0; ED.pos=t.length; paint(tx); }
    else if (k==='z'){ e.preventDefault(); imeCommit(); const key=tx.closest('li').dataset.k+'/'+state.cur, st=ED.undo.get(key)||[]; if (st.length){ const L=lineOf(tx); L.text=st.pop(); ED.pos=ED.anc=Math.min(ED.pos,L.text.length); paint(tx); lineChanged(tx); } }
    return;   // 복사·잘라내기·붙여넣기는 copy/cut/paste 이벤트로
  }
  if (e.altKey) return;
  const m=KM[e.code];
  if (m && e.key!=='Process'){ e.preventDefault(); if (ED.pos!==ED.anc){ imeCommit(); delSel(); } imeKey(tx, m[e.shiftKey?1:0]); return; }
  switch (e.key){
    case 'Backspace': e.preventDefault(); if (imeBack()) return; if (delSel()) return; if (ED.pos>0) setT(tx,t.slice(0,ED.pos-1)+t.slice(ED.pos),ED.pos-1); return;
    case 'Delete': e.preventDefault(); imeCommit(); if (delSel()) return; if (ED.pos<t.length) setT(tx,t.slice(0,ED.pos)+t.slice(ED.pos+1),ED.pos); return;
    case 'ArrowLeft': case 'ArrowRight': {
      e.preventDefault(); const vert=getComputedStyle(tx).writingMode.startsWith('vertical');
      if (vert){ const li=tx.closest('li'), nx=e.key==='ArrowLeft'?li.nextElementSibling:li.previousElementSibling; if (nx) focusLine(nx.querySelector('.tx'),Math.min(ED.pos,lineOf(nx.querySelector('.tx')).text.length)); return; }
      if (!e.shiftKey && ED.pos!==ED.anc){ moveTo(e.key==='ArrowLeft'?Math.min(ED.pos,ED.anc):Math.max(ED.pos,ED.anc)); return; }
      moveTo(stepG(t,ED.pos,e.key==='ArrowLeft'?-1:1), e.shiftKey); return; }
    case 'ArrowUp': case 'ArrowDown': {
      e.preventDefault(); const vert=getComputedStyle(tx).writingMode.startsWith('vertical');
      if (vert){ moveTo(stepG(t,ED.pos,e.key==='ArrowUp'?-1:1), e.shiftKey); return; }
      const li=tx.closest('li'), nx=e.key==='ArrowUp'?li.previousElementSibling:li.nextElementSibling; if (nx) focusLine(nx.querySelector('.tx'),ED.pos); return; }
    case 'Home': e.preventDefault(); moveTo(0,e.shiftKey); return;
    case 'End': e.preventDefault(); moveTo(t.length,e.shiftKey); return;
    case 'Enter': { e.preventDefault(); const nx=tx.closest('li').nextElementSibling; if (nx) focusLine(nx.querySelector('.tx'),0); else imeCommit(); return; }
    case 'Escape': e.preventDefault(); $('kbd').blur(); return;
    case 'Tab': imeCommit(); return;
  }
  if (e.key==='`'){ e.preventDefault(); insertText('□'); return; }
  if (e.key==='"'){ e.preventDefault(); insertText('〃'); return; }   // 반복 부호 (Shift+')
  if (e.key.length===1){ e.preventDefault(); insertText(e.key); return; }
}
function imeToggle(on){
  IME.on=on; imeCommit(); try{ localStorage.setItem('oldIME',on?'1':'0'); }catch(e){}
  $('imeOn').checked=on; $('imeHelp').hidden=!on;
  if (ED.tx && document.activeElement===$('kbd')) $('kbd').blur();
  ED.tx=null; if (typeof renderCurrent==='function' && state.pages.length) renderCurrent();
}
function setupKbd(){
  const kb=$('kbd');
  kb.addEventListener('keydown',kbdKeydown);
  kb.addEventListener('input',()=>{ if (kb.value){ const v=kb.value; kb.value=''; insertText(v); } });
  kb.addEventListener('paste',e=>{ e.preventDefault(); insertText((e.clipboardData||window.clipboardData).getData('text')); });
  kb.addEventListener('copy',e=>{ if (!ED.tx) return; e.preventDefault(); e.clipboardData.setData('text/plain',selText()); });
  kb.addEventListener('cut',e=>{ if (!ED.tx) return; e.preventDefault(); e.clipboardData.setData('text/plain',selText()); imeCommit(); delSel(); });
  kb.addEventListener('blur',()=>{ imeCommit(); const tx=ED.tx; setTimeout(()=>{ if (document.activeElement!==kb && tx && tx.isConnected){ paint(tx); } },0); });
  document.addEventListener('mousemove',e=>{ if (ED.drag && ED.tx){ ED.pos=posFromPoint(ED.tx,e.clientX,e.clientY); paint(ED.tx); } });
  document.addEventListener('mouseup',()=>{ ED.drag=false; });
}
// 행 하나를 편집할 수 있게 묶는다 (lineItem에서 부름)
function bindLine(tx){
  if (IME.on){
    tx.contentEditable='false'; tx.tabIndex=0; tx.setAttribute('role','textbox'); paint(tx);
    tx.addEventListener('mousedown',e=>{ if (e.button!==0) return; e.preventDefault(); const p=posFromPoint(tx,e.clientX,e.clientY);
      if (e.shiftKey && ED.tx===tx){ ED.pos=p; paint(tx); } else focusLine(tx,p); ED.drag=true; ED.anc=e.shiftKey?ED.anc:p; ED.pos=p; paint(tx); });
    tx.addEventListener('dblclick',e=>{ e.preventDefault(); ED.anc=0; ED.pos=getT(tx).length; paint(tx); });
    tx.addEventListener('focus',()=>focusLine(tx));
  } else {
    tx.contentEditable='true'; tx.textContent=lineOf(tx).text;
    tx.addEventListener('focus',()=>{ select(+tx.closest('li').dataset.k,false); $('oldk').hidden=false; });
    tx.addEventListener('input',()=>{ const L=lineOf(tx); L.text=tx.textContent.replace(/\n/g,'').replace(/ /g,' '); lineChanged(tx); });
    tx.addEventListener('keydown',e=>{ if (e.key==='Enter'){ e.preventDefault(); const nx=tx.closest('li').nextElementSibling; if (nx) nx.querySelector('.tx').focus(); } });
  }
}
