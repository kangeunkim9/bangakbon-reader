const {chromium}=require('playwright'); const http=require('http'), fs=require('fs'), path=require('path');
const types={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.wasm':'application/wasm','.json':'application/json','.jpg':'image/jpeg'};
const srv=http.createServer((q,r)=>{const f=path.join(__dirname,decodeURIComponent(q.url.split('?')[0])); if(!fs.existsSync(f)){r.writeHead(404);return r.end();} r.writeHead(200,{'content-type':types[path.extname(f)]||'application/octet-stream'}); fs.createReadStream(f).pipe(r);}).listen(8781, async()=>{
  const b=await chromium.launch(); const p=await b.newPage({viewport:{width:1300,height:900}}); p.on('pageerror',e=>console.log('pageerror',e.message));
  await p.goto('http://localhost:8781/local.html'); await p.waitForTimeout(800);
  const tx=p.locator('.lines li .tx').first();
  await tx.click(); await p.evaluate(()=>{const t=document.querySelector('.lines li .tx'); t.textContent='하'; const r=document.createRange(); r.setStart(t.firstChild,1); r.collapse(true); getSelection().removeAllRanges(); getSelection().addRange(r);});
  await p.keyboard.type('`'); let v=await tx.textContent(); console.log('after `:',JSON.stringify(v),[...v].map(c=>c.codePointAt(0).toString(16)));
  await p.keyboard.type('`'); v=await tx.textContent(); console.log('after ``:',[...v].map(c=>c.codePointAt(0).toString(16)));
  await p.evaluate(()=>{const t=document.querySelector('.lines li .tx'); t.textContent='사'; const r=document.createRange(); r.setStart(t.firstChild,1); r.collapse(true); getSelection().removeAllRanges(); getSelection().addRange(r);});
  await p.click('#okL button[title="첫소리 ᄯ"]'); v=await tx.textContent(); console.log('ᄯ:',[...v].map(c=>c.codePointAt(0).toString(16)));
  await p.click('#okT button[title="끝소리 ᇰ"]'); v=await tx.textContent(); console.log('+ᇰ:',[...v].map(c=>c.codePointAt(0).toString(16)));
  await p.keyboard.type('다'); v=await tx.textContent(); console.log('typed after:',JSON.stringify(v));
  const st=await p.evaluate(()=>state.pages[0].lines[0].text); console.log('state:',JSON.stringify(st));
  await p.locator('#oldk').screenshot({path:'oldk.png'}); await b.close(); srv.close();
});
