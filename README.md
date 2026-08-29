# 기술 임용 A/B 자동검증 모의고사 생성기

이 버전은 이전의 'GPT가 문제와 정답을 모두 생성'하는 구조를 폐기하고,
**정답을 먼저 고정한 뒤 AI는 문항 표현만 담당**하도록 설계했습니다.

## 내장 DB
사용자가 제공한 영역별 서브노트, 02~25 기출 풀이, 모의고사 문제 형식을 텍스트 색인한 `knowledge.db`가 포함되어 있습니다.
원본 PDF 자체는 배포 파일에 포함하지 않습니다.

## 오류 억제 구조
- 계산형: Python 공식 템플릿이 조건/정답/풀이를 생성
- 개념형: 원문 answer + evidence + source/page를 먼저 고정
- AI: 고정 정답을 바꾸지 않고 임용형 상황문장만 재구성
- 검증 실패: 자동 폐기 또는 보수적 원문 문항으로 대체
- A/B 중복: topic/fingerprint 검사

## 실행
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Secrets
AI 문장 재구성을 쓸 때만 필요:
```toml
OPENAI_API_KEY="sk-..."
```
API 키가 없어도 AI 없는 안전모드는 실행됩니다.

## 배포
GitHub 저장소 루트에 아래 파일들을 그대로 올립니다.
- app.py
- knowledge.db
- formula_templates.py
- retrieval.py
- validators.py
- ai_wrapper.py
- exam_builder.py
- pdf_export.py
- config.json
- requirements.txt
- packages.txt

Streamlit Community Cloud의 Main file path는 `app.py`입니다.

## 권장 사용
기본은 `엄격 자동검증` 개념입니다. AI는 켜도 정답 결정권이 없습니다.


## 난이도/회로 정책
- 기본 난이도: **적당히 어려움** — 핵심개념 1~2개를 자료·상황에 적용하는 중간~중상 수준
- 과도한 다단계 계산, 함정형 조건, 3개 이상 독립개념 결합은 기본값에서 억제
- 회로 문제: **최대한 제외**가 기본값
- UI에서 `완전 제외 / 최대한 제외 / 허용` 선택 가능
- 전기·전자 영역 자체는 유지하며 비회로 주제를 우선 출제
