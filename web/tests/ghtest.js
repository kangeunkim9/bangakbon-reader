// 공개판(web/index.html)을 띄워 예시 면을 판독하고, 결과가 sample_out.json의 greedy와 같은지 본다.  node ghtest.js
const {chromium}=require('playwright'); const http=require('http'), fs=require('fs'), path=require('path');
const root=path.join(__dirname,'..');
const types={'.html':'text/html; charset=utf-8','.js':'text/javascript','.wasm':'application/wasm','.json':'application/json','.jpg':'image/jpeg'};
const srv=http.createServer((q,r)=>{let u=decodeURIComponent(q.url.split('?')[0]); if(u.endsWith('/'))u+='index.html'; const f=path.join(root,u); if(!fs.existsSync(f)){r.writeHead(404);return r.end();} r.writeHead(200,{'content-type':types[path.extname(f)]||'application/octet-stream'}); fs.createReadStream(f).pipe(r);}).listen(8779, async()=>{
  const b=await chromium.launch(); const p=await b.newPage({viewport:{width:1300,height:900}}); p.on('pageerror',e=>console.log('pageerror',e.message)); p.on('console',m=>{if(m.type()==='error')console.log('console',m.text())});
  if(fs.existsSync('ort.wasm.min.js')) await p.route('https://cdn.jsdelivr.net/**',r=>r.fulfill({path:'ort.wasm.min.js',contentType:'text/javascript'})); await p.route('https://fonts.**',r=>r.abort());
  await p.goto('http://localhost:8779/'); await p.waitForTimeout(1500);
  console.log(await p.evaluate(()=>({mode:document.compatMode, lmVisible:!!document.querySelector('label.lmonly').offsetParent, lmChecked:document.getElementById('lm').checked, gh:!!document.querySelector('.ghonly').offsetParent, save:!document.getElementById('saveAll').hidden, lines:state.pages[0]&&state.pages[0].lines.length})));
  await p.screenshot({path:'/tmp/gh_rest.png'});
  await p.click('#trySample'); await p.waitForFunction(()=>state.running,null,{timeout:10000});
  await p.waitForFunction(()=>!state.running && state.pages.length>0 && true,null,{timeout:240000}).catch(e=>console.log('wait',e.message));
  const R=await p.evaluate(()=>({msg:document.getElementById('msg').textContent, useLM:state.useLM, lines:state.pages[0].lines.map(l=>l.text)}));
  const J=JSON.parse(fs.readFileSync(path.join(__dirname,'../src/sample_out.json'))); console.log(R.msg, R.useLM, 'same as greedy:', JSON.stringify(R.lines)===JSON.stringify(J.greedy), R.lines.length);
  const [dl]=await Promise.all([p.waitForEvent('download',{timeout:5000}).catch(()=>null), p.click('#saveAll')]); console.log('download', dl&&dl.suggestedFilename()); if(dl){await dl.saveAs('/tmp/dl.txt'); console.log(fs.readFileSync('/tmp/dl.txt','utf8').slice(0,200));}
  await p.screenshot({path:'/tmp/gh_run.png'});
  await b.close(); srv.close();
});
