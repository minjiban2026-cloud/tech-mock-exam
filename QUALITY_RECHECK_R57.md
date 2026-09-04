# R57 final audit

- Root fix: quality_judge.judge_question now recovers source_context_override/evidence whenever a legacy caller passes an empty source_context.
- R57 coverage counts only historical actual Judge PASS + R57_AI_VERIFIED contracts. R51-R56 Python-only/0-pass contracts cannot raise coverage.
- One-click pipeline: fixed candidate pool generated before Judge feedback -> strict Python validation -> each candidate Judge at most once -> PASS only stored.
- Domain-specific past_exam pages (fallback: official_exam 4-point pages) are supplied as structure/style references; subnote anchors remain answer ground truth.
- R57 hard gate rejects direct answer leakage, independent task2, single-clue task1, weak grounding, unrelated anchors, thin/artificial material, and missing transfer chain.
- Normal real-DB material-mechanics contract: PASS. Intentional leak/independent/single-clue mutations: REJECT.
- quality_judge source-context fallback mock: PASS for source_context_override and evidence fallback.
- Full repo Python compile: PASS.
- Streamlit top-level import with a UI stub: PASS.
- Historical verified baseline with no R57 contracts: 5/18.
- Legacy R51-R56 contract rows inserted: baseline remains 5/18.
- make_ab signature preserved.
