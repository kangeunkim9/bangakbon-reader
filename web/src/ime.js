// ---------- 옛한글 자판 (두벌식 + 옛글자 규칙) ----------
const IME={on:true, comp:null, tx:null, pos0:0, len:0, hist:[]};
try { const v=localStorage.getItem('oldIME'); if (v!==null) IME.on=v==='1'; } catch(e){}
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
function imeRender(c){
  if (!c) return '';
  const L=c.L?LOF[c.L]:'', T=c.T?TOF[c.T]:'';
  if (c.L && c.V) return (L+c.V+T).normalize('NFC');
  if (c.L) return c.L.length===1 ? c.L : L+'ᅠ';
  return VDISP[c.V] || ('ᅟ'+c.V);
}
function imeDraw(){
  const tx=IME.tx; if (!tx) return; const t=tx.textContent, r=imeRender(IME.comp);
  tx.textContent=t.slice(0,IME.pos0)+r+t.slice(IME.pos0+IME.len); IME.len=r.length; setCaret(tx,IME.pos0+r.length); tx.dispatchEvent(new Event('input'));
}
function imeCommit(){ IME.comp=null; IME.hist=[]; IME.len=0; }
function imeStart(tx,c){
  imeCommit(); IME.tx=tx; let p=caretIn(tx); if (p===null) p=tx.textContent.length; IME.pos0=p; IME.len=0; IME.comp=c; IME.hist=[]; imeDraw();
}
function imePush(c){ IME.hist.push(IME.comp); IME.comp=c; imeDraw(); }
function imeNext(c){ const tx=IME.tx; IME.pos0+=IME.len; IME.len=0; IME.comp=c; IME.hist=[]; imeDraw(); }
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
function imeKeydown(e, tx){
  if (!IME.on || e.ctrlKey || e.altKey || e.metaKey) { imeCommit(); return; }
  if (e.key==='Process' || e.isComposing){ $('imeWarn').hidden=false; return; }
  const m=KM[e.code];
  if (m){ e.preventDefault(); $('imeWarn').hidden=true; imeKey(tx, m[e.shiftKey?1:0]); return; }
  if (e.key==='Backspace'){ if (imeBack()) e.preventDefault(); return; }
  if (e.key.length===1 || ['Enter','Tab','ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home','End','Delete','Escape'].includes(e.key)) imeCommit();
}
function imeToggle(on){ IME.on=on; imeCommit(); try{ localStorage.setItem('oldIME',on?'1':'0'); }catch(e){} $('imeOn').checked=on; $('imeHelp').hidden=!on; $('imeWarn').hidden=true; }
