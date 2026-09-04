# R58 final audit
- Version: RESILIENT-NARROW-R58-20260904
- Replaced 160-anchor writer prompt with narrow deterministic evidence bundles.
- Writer timeout/format failure now falls back to Python-grounded candidates instead of pool=0.
- Judge source context preserved; Judge timeout increased defensively.
- Historical Judge PASS + R58 real Judge PASS only count toward 18/18.
- Forced-writer-timeout pipeline test: 5/18 -> 18/18 under mock Judge PASS, 13 Judge candidates, all 9 domains nonzero pools.
- make_ab signature preserved.
- knowledge.db excluded from package.
