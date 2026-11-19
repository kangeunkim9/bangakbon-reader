const {chromium}=require('playwright'); const http=require('http'), fs=require('fs'), path=require('path');
const types={'.html':'text/html','.js':'text/javascript','.wasm':'application/wasm','.json':'application/json','.jpg':'image/jpeg'};
const srv=http.createServer((q,r)=>{const f=path.join(__dirname,decodeURIComponent(q.url.split('?')[0])); if(!fs.existsSync(f)){r.writeHead(404);return r.end();} r.writeHead(200,{'content-type':types[path.extname(f)]||'application/octet-stream'}); fs.createReadStream(f).pipe(r);}).listen(8778, async()=>{
  const b=await chromium.launch(); const p=await b.newPage({viewport:{width:1300,height:900}}); p.on('pageerror',e=>console.log('pageerror',e.message)); p.on('console',m=>{if(m.type()==='error')console.log('console',m.text())});
  await p.goto('http://localhost:8778/local.html'); await p.waitForTimeout(800);
  await p.screenshot({path:'shot_rest.png'});
  await p.setInputFiles('#file','test.pdf'); await p.click('#go');
  await p.waitForFunction(()=>!state.running && state.pages.length>0,null,{timeout:240000});
  const R=await p.evaluate(()=>({msg:document.getElementById('msg').textContent, pages:state.pages.map(P=>({no:P.no,status:P.status,cols:P.cols,lines:P.lines.map(l=>l.text)}))}));
  fs.writeFileSync('page_out.json',JSON.stringify(R)); await p.click('#pages button:nth-child(2)'); await p.click('.facs rect[data-k="3"]'); await p.waitForTimeout(400);
  await p.screenshot({path:'shot_run.png'});
  await p.click('#vert'); await p.waitForTimeout(300); await p.screenshot({path:'shot_vert.png'});
  await p.setViewportSize({width:400,height:850}); await p.click('#vert'); await p.waitForTimeout(300); await p.screenshot({path:'shot_phone.png',fullPage:false});
  const ov=await p.evaluate(()=>document.documentElement.scrollWidth); console.log('phone scrollWidth',ov);
  await b.close(); srv.close();
});
