# R55 검수 기록

- 원인: Streamlit file_uploader는 rerun 후에도 업로드 파일을 유지한다. R54는 매 rerun마다 그 파일을 다시 import하여, 보충 채굴로 늘어난 R54_CONTRACTS를 채굴 전 JSON으로 덮어썼다.
- 수정: 업로드 bytes SHA-256을 R55_IMPORTED_CONTRACT_SHA에 기록하고 같은 파일은 최초 1회만 import한다.
- 수정: R49 `coverage_inventory` / `validation_inventory` 진단 패널 및 구형 `18 capability 전체 일괄 생성 + Judge 전수검증` UI 제거.
- 최종 경로: combined_coverage_inventory -> select_hybrid_validation_contracts -> make_hybrid_contract_validation_suite 하나만 사용.
- 전체 repo top-level Python 15개 py_compile PASS.
- make_ab 인자/순서 유지: db_path,a_count,a_points,b_count,b_points,domains,api_key,model,ai_enabled,ai_quality_enabled,judge_model,seed.
- knowledge.db 미포함.
- 현재 런타임에는 streamlit CLI가 없어 실제 서버 프로세스 기동 시험은 수행할 수 없었음. 정적 startup/import path와 전체 Python 컴파일은 검증함.
