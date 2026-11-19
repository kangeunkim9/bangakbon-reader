const {chromium}=require('playwright'); const http=require('http'), fs=require('fs'), path=require('path');
const root=path.join(__dirname,'..');
const srv=http.createServer((q,r)=>{let u=decodeURIComponent(q.url.split('?')[0]); if(u.endsWith('/'))u+='index.html'; const f=path.join(root,u); if(!fs.existsSync(f)){r.writeHead(404);return r.end();} r.writeHead(200,{'content-type':f.endsWith('.html')?'text/html; charset=utf-8':'application/octet-stream'}); fs.createReadStream(f).pipe(r);}).listen(8783, async()=>{
  const b=await chromium.launch(); const p=await b.newPage({viewport:{width:1300,height:900}}); p.on('pageerror',e=>console.log('pageerror',e.message)); p.on('console',m=>{if(m.type()==='error')console.log('console',m.text())});
  await p.route('https://**',r=>r.abort());
  await p.goto('http://localhost:8783/'); await p.waitForTimeout(600);
  const tx=p.locator('.lines li .tx').first();
  const cases=[['gkk','ᄒᆞ'],['gkkk','ᄒᆢ'],['gkkl','ᄒᆡ'],['rksk','가나'],['gksk','하나'],['gks dl','한 이'],['tek','ᄯᅡ'],['qdk','ᄫᅡ'],['+Ak','ᅀᅡ'],['dk+D','아ᇰ'],['fkf+G','라ᇙ'],['gks\b','하'],['gkkt','ᄒᆞᆺ'],['dhk','와'],['tkfkd','사랑'],['qtrk','ᄢᅡ'],['gk`','하□']];
  let ok=0;
  for (const [seq,exp] of cases){
    await p.evaluate(()=>{ state.pages[0].lines[0].text=''; }); await tx.click();
    for (let i=0;i<seq.length;i++){ const ch=seq[i];
      if (ch==='+'){ i++; await p.keyboard.press('Shift+Key'+seq[i]); continue; }
      if (ch===' '){ await p.keyboard.press('Space'); continue; }
      if (ch==='\b'){ await p.keyboard.press('Backspace'); continue; }
      if (ch==='`'){ await p.keyboard.press('Backquote'); continue; }
      await p.keyboard.press('Key'+ch.toUpperCase()); }
    const v=(await p.evaluate(()=>state.pages[0].lines[0].text)).normalize('NFC'); const good=v===exp.normalize('NFC'); ok+=good; console.log(good?'OK ':'XX ',seq.replace('\b','⌫'),'→',v,'(기대',exp+')');
  }
  console.log('통과',ok,'/',cases.length);
  // 한글 입력기 상태 흉내: key='Process'가 아닌 password 칸이므로 그대로 영문 키가 온다. 편집 동작 시험
  await p.evaluate(()=>{ state.pages[0].lines[1].text='가나다라'; renderCurrent(); });
  const t2=p.locator('.lines li .tx').nth(1); await t2.click();
  await p.keyboard.press('Home'); await p.keyboard.press('ArrowRight'); await p.keyboard.press('Shift+ArrowRight'); await p.keyboard.press('KeyG'); await p.keyboard.press('KeyK'); await p.keyboard.press('KeyK');
  console.log('선택 바꾸기:', await p.evaluate(()=>state.pages[0].lines[1].text), '(기대 가ᄒᆞ다라)');
  await p.keyboard.press('Control+KeyZ'); console.log('되돌리기:', await p.evaluate(()=>state.pages[0].lines[1].text));
  await p.keyboard.press('ArrowDown'); console.log('아래 행 active', await p.evaluate(()=>active));
  await p.screenshot({path:'/tmp/ime2.png',clip:{x:600,y:250,width:700,height:350}});
  await p.click('#vert'); await p.waitForTimeout(300); console.log('vert before', await p.evaluate(()=>[document.documentElement.scrollWidth, scrollX])); await p.locator('.lines li .tx').nth(2).click(); await p.keyboard.press('KeyG'); await p.keyboard.press('KeyK');
  console.log('scroll', await p.evaluate(()=>[document.documentElement.scrollWidth, scrollX, document.querySelector('.vert .lines').scrollWidth, document.querySelector('.vert .lines').clientWidth])); console.log('세로 입력:', await p.evaluate(()=>state.pages[0].lines[2].text.slice(-3)));
  await p.screenshot({path:'/tmp/ime3.png'});
  await p.click('#imeOn'); await p.waitForTimeout(200); console.log('자판 끔: contenteditable', await p.evaluate(()=>document.querySelector('.lines li .tx').isContentEditable));
  await b.close(); srv.close();
});
