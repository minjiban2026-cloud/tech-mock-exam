# AI QUALITY GATE DESIGN

이 버전은 최근 실물 출력에서 발견된 실패 유형을 직접 겨냥합니다.

## 막으려는 실패 유형
- 지문에 정의가 거의 그대로 제시되어 정답을 베껴 쓰는 4점 문제
- 3~4개의 채점요소가 있어도 모두 단순 회상인 문제
- 가까운 페이지의 무관한 anchor를 억지로 묶은 문제
- 자료가 장식에 불과한 문제
- 배점에 비해 추론거리가 짧은 문제
- 같은 작성방법 문법이 A/B 전체에서 반복되는 문제

## 품질 판단 권한 분리
- 정답/계산: DB + Python
- 관계성 후보 선별: AI (선별만)
- 문항 표현: AI
- 문항 품질 veto: 독립 AI reviewer
- 사실/구조 최종 검사: Python
- 섹션/A-B 전체 편집 품질: AI reviewer

## 중요한 안전장치
- AI reviewer가 PASS해도 deterministic 검사를 통과하지 못하면 폐기
- AI writer가 만든 정답은 사용하지 않음
- 4점 AI-grounded 문항은 master_concept와 relation이 필수
- 원문 정의 복사 유사도 0.86 이상은 AI 판단과 무관하게 폐기
- 4점은 AI reviewer가 최소 2종 이상의 사고행동을 확인해야 통과
