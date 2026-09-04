# R53 HYBRID CONTRACT COVERAGE

- 기존 실제 Judge PASS 5개를 coverage에서 다시 사용한다. 재Judge하지 않는다.
- PYTHON_VALIDATED contract는 동일 contract_type 반복을 1 capability로만 센다.
- 사용자 R52 결과를 재현하면 전체 후보 coverage는 16/18이며 실제 부족 영역은 건설기술 1, 재료역학 1뿐이다.
- R53 최종 보충 채굴은 이 두 실제 gap만 영역당 최대 1회 호출한다.
- final Judge suite는 과거 PASS 5개를 재호출하지 않고 새 contract 13개만 각각 1회 Judge한다.
- R49 0-pass directional/cause, R46 0-pass paired/condition 구조는 재사용하지 않는다.
- knowledge.db는 배포 ZIP에 포함하지 않는다.
