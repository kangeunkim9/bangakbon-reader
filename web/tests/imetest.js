const {chromium}=require('playwright'); const http=require('http'), fs=require('fs'), path=require('path');
const types={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.wasm':'application/wasm','.json':'application/json','.jpg':'image/jpeg'};
const srv=http.createServer((q,r)=>{const f=path.join(__dirname,decodeURIComponent(q.url.split('?')[0])); if(!fs.existsSync(f)){r.writeHead(404);return r.end();} r.writeHead(200,{'content-type':types[path.extname(f)]||'application/octet-stream'}); fs.createReadStream(f).pipe(r);}).listen(8782, async()=>{
  const b=await chromium.launch(); const p=await b.newPage({viewport:{width:1300,height:900}}); p.on('pageerror',e=>console.log('pageerror',e.message));
  await p.goto('http://localhost:8782/local.html'); await p.waitForTimeout(800);
  const tx=p.locator('.lines li .tx').first();
  const cases=[['gkk','ᄒᆞ'],['gkkk','ᄒᆢ'],['gkkl','ᄒᆡ'],['rksk','가나'],['gksk','하나'],['gks dl','한 이'],['tek','ᄯᅡ'],['qdk','ᄫᅡ'],['+Ak','ᅀᅡ'],['dk+D','아ᇰ'],['fkf+G','라ᇙ'],['gks\b','하'],['gkkt','ᄒᆞᆺ'],['dhk','와'],['tkfkd','사랑'],['qtrk','ᄢᅡ']];
  for (const [seq,exp] of cases){
    await tx.click(); await p.evaluate(()=>{const t=document.querySelector('.lines li .tx'); t.textContent=''; t.focus();});
    for (let i=0;i<seq.length;i++){ const ch=seq[i];
      if (ch==='+'){ i++; await p.keyboard.press('Shift+Key'+seq[i]); continue; }
      if (ch===' '){ await p.keyboard.press('Space'); continue; }
      if (ch==='\b'){ await p.keyboard.press('Backspace'); continue; }
      await p.keyboard.press('Key'+ch.toUpperCase()); }
    const v=(await p.evaluate(()=>state.pages[0].lines[0].text)).normalize('NFC'); console.log(v===exp.normalize('NFC')?'OK ':'XX ',seq.replace('\b','⌫'),'→',v,'(기대',exp+')');
  }
  await b.close(); srv.close();
});
