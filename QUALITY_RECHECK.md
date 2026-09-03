# R47 failure-class correction

R46의 18개 전수 Judge 결과를 개별 문항 패치가 아니라 capability class 단위로 반영한다.

- paired_concept_discrimination: 0/7 PASS -> coverage target에서 폐기
- condition_outcome_swap: 0/1 PASS -> coverage target에서 폐기
- deterministic_formula_operation: 3/3 PASS -> 유지
- ordered_sequence_repair: 일부 문항은 Judge가 4점 성립/무 fatal/다른 점수 4+로 평가했지만 inferential_distance=3 하나 때문에 후처리 REJECT. R44와 같은 판정 불일치이므로 이 capability에만 inferential floor 3.0 적용. 다른 품질 floor와 fatal veto는 유지.

중요: 폐기된 두 구조를 다른 이름으로 바꿔 18개를 억지로 채우지 않는다. 현재 DB에서 남는 target 수를 그대로 보고하며, 9영역×2가 아니면 실제 18-Judge 실행을 차단한다.
