# p1(필사본 더 배운 SVTRv2)을 언어 모델 보정까지 붙여 금향졍긔로 재고, 브라우저용 ONNX 조각을 만든다(Colab 터미널에서 python3로).
# eval_lm_svtrv2.py를 그대로 쓰되, 모델 경로와 결과 위치만 p1용으로 바꾼다(v2 결과는 덮어쓰지 않는다).
import os, requests
RAWG = 'https://raw.githubusercontent.com/kangeunkim9/bangakbon-reader/main/'
src = requests.get(RAWG + 'training/eval_lm_svtrv2.py', timeout=60).text
src = src.replace("from google.colab import drive; drive.mount('/content/drive')", "pass")
src = src.replace("'/result_lm_svtrv2.json'", "'/result_lm_svtrv2_p1.json'").replace("'/web_svtrv2'", "'/web_svtrv2_p1'")
assert "web_svtrv2_p1" in src and "result_lm_svtrv2_p1" in src
G = {'__name__': '__main__', 'SV_MODEL': '/content/drive/MyDrive/방각본_판독기/svtrv2_model_p1.pt'}
exec(src, G)
# 9MB 조각으로 나누기(사이트가 받아서 다시 이어 붙인다)
OUTD = '/content/drive/MyDrive/방각본_판독기/web_svtrv2_p1'
b = open(OUTD + '/svtrv2.onnx', 'rb').read(); S = 9000000
for i in range(0, len(b), S): open(f'{OUTD}/model_vsvtrp1_{i // S}.wasm', 'wb').write(b[i:i + S])
print('조각', (len(b) + S - 1) // S, '개 ->', OUTD)
