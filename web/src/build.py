# web/index.html(GitHub Pages 공개판)을 만든다.  python3 build.py
# 예시 면(sample.jpg)의 행 경계와 판독 결과는 sample_out.json에 미리 넣어 둔 것을 쓴다.
import json, os
H = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(f'{H}/sample_out.json', encoding='utf-8'))
# 학습 자료 받는 창구(Apps Script 웹 앱) 주소. inbox_url.txt에 한 줄로 둔다.
INBOX = open(f'{H}/inbox_url.txt').read().strip() if os.path.exists(f'{H}/inbox_url.txt') else ''
src = open(f'{H}/page.src.html', encoding='utf-8').read()
eng = open(f'{H}/engine.js', encoding='utf-8').read()
S = {'name':'금향졍긔 권하','title':'금향졍긔 권하 PDF 21쪽','no':21,'W':1500,'H':1910,
     't':J['t'],'bt':J['bt'],'cols':J['cols'],'lines':J['greedy']}
gh = (src.replace('/*ENGINE*/', eng).replace('/*SAMPLE*/', json.dumps(S, ensure_ascii=False))
         .replace("lm: 'lm7.gz.wasm',", "lm: null,").replace("approxMB: 47, github: false", "approxMB: 45, github: true")
         .replace("inbox: ''", "inbox: " + json.dumps(INBOX)))
assert "lm: null," in gh and "github: true" in gh
SK = ('<!doctype html><html lang="ko"><head><meta charset="utf-8">'
      '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
      '<style>:root{color-scheme:light;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}'
      'body{margin:0;font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}img{max-width:100%}[hidden]{display:none!important}</style>'
      '</head><body>')
open(f'{H}/../index.html', 'w', encoding='utf-8').write(SK + gh + '</body></html>\n')
print('web/index.html', len(gh))
# 함께 고친 사람들 페이지
assert INBOX, 'inbox_url.txt가 없습니다'
open(f'{H}/../contributors.html', 'w', encoding='utf-8').write(open(f'{H}/contributors.src.html', encoding='utf-8').read().replace('__INBOX__', INBOX))
print('web/contributors.html')
