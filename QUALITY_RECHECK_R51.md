# R51 source-contract mining

- R49 0-pass 구조 `directional_rule_application`, `cause_intervention_prediction`은 재사용하지 않는다.
- R50의 5/18 인증값은 그대로 보존한다.
- 새 단계는 영역별 1회 AI 호출로 최대 2개의 `source reasoning contract`를 채굴한다.
- AI는 원문 밖 사실을 만들 수 없고, 각 계약은 anchor id를 명시한다.
- Python은 anchor/domain 일치, 정답 grounding, 정답 누출, source 원문 복사, 2단계 reasoning chain, task2 dependency를 검사한다.
- Python 검사를 통과한 계약만 `PYTHON_VALIDATED`; 최종 coverage에는 아직 포함하지 않는다.
- 18개가 모인 뒤 Judge는 각 문항을 정확히 1회 심사하며 REJECT 후 교체/재시도하지 않는다.
- `make_ab` 공개 인터페이스는 변경하지 않았다.
