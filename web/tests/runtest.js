const {chromium}=require('playwright'); const http=require('http'), fs=require('fs'), path=require('path');
const types={'.html':'text/html','.js':'text/javascript','.wasm':'application/wasm','.json':'application/json','.jpg':'image/jpeg'};
const srv=http.createServer((q,r)=>{const f=path.join(__dirname,decodeURIComponent(q.url.split('?')[0])); if(!fs.existsSync(f)){r.writeHead(404);return r.end();} r.writeHead(200,{'content-type':types[path.extname(f)]||'application/octet-stream'}); fs.createReadStream(f).pipe(r);}).listen(8777, async()=>{
  const b=await chromium.launch(); const p=await b.newPage(); p.on('console',m=>console.log('console:',m.text())); p.on('pageerror',e=>console.log('err',e.message));
  await p.goto('http://localhost:8777/'+(process.argv[2]||'test.html')); const R=await p.evaluate(()=>run());
  fs.writeFileSync(path.join(__dirname,'js_out.json'),JSON.stringify(R)); await b.close(); srv.close();
});
