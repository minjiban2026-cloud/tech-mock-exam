# R52 quality recheck

- User-observed R51 inventory reproduced: 8/18 validated slots, 10 missing slots across 7 domains.
- R52 source retrieval no longer takes only high-confidence anchors. It scans up to 700 anchors/domain, scores reasoning affordance, de-duplicates nested evidence, and limits same-page domination.
- New supplemental miner preserves existing contracts and makes at most one AI call per deficient domain.
- Added contract architectures: constraint_choice_justification, structured_mapping_application, comparative_case_discrimination, relation_composition.
- Existing retired 0-pass architectures (directional_rule_application, cause_intervention_prediction) are not reintroduced.
- Python validation retains grounding, direct-answer leakage, near-verbatim source, task2-dependency checks and adds duplicate-answer/source-thinness/multi-relation checks.
- Actual knowledge.db retrieval smoke test: all 9 domains return structurally ranked anchor packets; deficient-domain arithmetic from user result = 8 kept + 10 missing.
- py_compile PASS: app.py, exam_builder.py, capability_contracts.py.
- No knowledge.db included in patch ZIP.
