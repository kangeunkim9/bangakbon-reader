#@title ㉙-2 말뭉치 내려받기 단추 만들기 (누르면 국립국어원 내려받기 화면이 새 창으로 열림)
# 국립국어원 내려받기는 브라우저 전용 프로그램(INNORIX)으로만 되어 코드로는 받을 수 없다.
# 인증키는 보안 비밀에서 읽고 화면에 찍지 않는다. 아래 단추를 눌러 받은 zip은 내 컴퓨터 '다운로드' 폴더에 저장된다.
import requests, html
from google.colab import userdata
from IPython.display import display, HTML
API = 'https://kli.korean.go.kr/restapi/v1/corpus/download'
out = []
for yr in ('2023', '2024', '2025'):
    try:
        r = requests.get(API, params={'keyVal': userdata.get(f'KOHICO_KEY_{yr}')}, timeout=60)
        url = r.json() if r.headers.get('content-type', '').startswith('application/json') else r.text
        if isinstance(url, dict): url = next((v for v in url.values() if isinstance(v, str) and v.startswith('http')), '')
        url = str(url).strip().strip('"')
        out.append(f'<p><a href="{html.escape(url)}" target="_blank" style="font-size:16px">국어 역사 자료 말뭉치 {yr} 내려받기 화면 열기</a></p>' if url.startswith('http') else f'<p>{yr}: 주소를 못 받음</p>')
    except Exception as e:
        out.append(f'<p>{yr}: 오류 {type(e).__name__}</p>')
display(HTML(''.join(out) + '<p>열린 화면에서 <b>전체 다운로드</b>를 누르세요. 처음이면 내려받기 도우미(INNORIX) 설치를 물을 수 있어요.</p>'))
