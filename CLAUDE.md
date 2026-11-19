# 방각본 판독기 작업 안내 (Claude용)

이 저장소는 성균관대학교 국어국문학과 고전소설 연구실의 방각본 한글 고소설 자동 판독기입니다.
사이트: https://kangeunkim9.github.io/bangakbon-reader/ (main에 올라가면 GitHub Actions가 자동 배포)

## 고칠 곳
- 판독기 화면과 동작: `web/src/page.src.html` (CSS·JS가 한 파일에 들어 있음)
- 행 나누기·전처리·디코딩: `web/src/engine.js`. 행 나누기를 바꾸면 Python 쪽 `web/src/seg.py`도 똑같이 바꾼다(학습·대량 판독이 seg.py를 씀).
- 옛한글 자판: `web/src/ime.js`
- 학습 자료 받는 창구(Google Apps Script): `web/src/inbox.gs` (여기 고쳐도 자동 배포되지 않음. 저장소 주인이 Apps Script에 다시 배포해야 함)
- Colab 대량 판독: `training/bulk_ocr.py`, 학습 코드: `training/train_v*.py`

## 반드시 지킬 것
- `web/index.html`, `web/contributors.html`은 직접 고치지 않는다. `web/src`를 고친 뒤 `cd web/src && python3 build.py`로 만든다.
- 고친 뒤에는 브라우저에서 예시 면으로 판독이 되는지 확인한다(`web/tests`의 시험 스크립트 참고).
- main에 바로 올리지 말고 새 브랜치에서 작업해 풀 리퀘스트를 연다. 저장소 주인이 확인한 뒤 합친다.
- KOHICO(국어 역사 말뭉치) 전사문, 그것으로 만든 언어 모델, 사람 검수 정답 파일은 저장소에 올리지 않는다(이용 조건).
- 학습 자료 창구 주소(`web/src/inbox_url.txt`) 말고 개인 이메일 등은 코드에 넣지 않는다.
- 화면 문구는 한국어로, 줄표(—) 없이, 딱딱한 문어체를 피해 쓴다.
- 판독 규칙: 원문 표기 그대로, 띄어쓰기 없이, 판본 한 행을 한 줄로. 못 읽는 글자 □, 반복 부호 〃.
