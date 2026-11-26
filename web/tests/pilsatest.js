// 필사본 모드로 면 하나를 판독해 행 목록을 찍는다(BANG=1이면 방각본 모드).  node pilsatest.js <이미지> [ort dist 폴더]
const {chromium}=require('playwright'); const http=require('http'), fs=require('fs'), path=require('path');
const root=path.join(__dirname,'..'), img=process.argv[2], ortDir=process.argv[3];
const types={'.html':'text/html; charset=utf-8','.js':'text/javascript','.mjs':'text/javascript','.wasm':'application/wasm','.json':'application/json','.jpg':'image/jpeg'};
const srv=http.createServer((q,r)=>{let u=decodeURIComponent(q.url.split('?')[0]); if(u.endsWith('/'))u+='index.html'; const f=path.join(root,u); if(!fs.existsSync(f)){r.writeHead(404);return r.end();} r.writeHead(200,{'content-type':types[path.extname(f)]||'application/octet-stream'}); fs.createReadStream(f).pipe(r);}).listen(8783, async()=>{
  const b=await chromium.launch(); const p=await b.newPage({viewport:{width:1300,height:900}}); p.on('pageerror',e=>console.log('pageerror',e.message));
  if (ortDir) await p.route(/cdn\.jsdelivr\.net\/npm\/onnxruntime-web@[^/]+\/dist\/(.*)$/, r=>{ const f=path.join(ortDir, r.request().url().split('/dist/')[1].split('?')[0]); fs.existsSync(f)? r.fulfill({path:f, contentType: f.endsWith('.wasm')?'application/wasm':'text/javascript'}) : r.abort(); });
  await p.route(/fonts\.|goatcounter/, r=>r.abort());
  await p.goto('http://localhost:8783/?cpu'); await p.waitForTimeout(1500);
  if (!process.env.BANG) await p.evaluate(()=>{ const r=document.querySelector('input[name=kind][value=pilsa]'); r.checked=true; r.dispatchEvent(new Event('change')); });
  await p.setInputFiles('#file', img); await p.waitForTimeout(1500);
  if (!(await p.evaluate(()=>state.running))) await p.click('#go').catch(()=>{});
  await p.waitForFunction(()=>state.running,null,{timeout:20000}).catch(()=>{});
  await p.waitForFunction(()=>!state.running && state.pages.length>0,null,{timeout:600000}).catch(e=>console.log('wait',e.message));
  const R=await p.evaluate(()=>({msg:document.getElementById('msg').textContent, pages:state.pages.map(P=>({st:P.status, n:P.cols.length, lines:P.lines.map(l=>l.text)}))}));
  console.log(R.msg); R.pages.forEach(P=>{ console.log('면', P.st, P.n+'행'); P.lines.forEach(t=>console.log('  '+t)); });
  await b.close(); srv.close();
});
