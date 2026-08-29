"""기출/모의고사 구조를 모방하기 위한 출제 청사진 엔진.
내용을 복제하지 않고 문제의 형식·사고 구조만 사용한다.
"""
import random

PATTERNS={
  "identify_apply": {"label":"개념 판별·적용","material":"상황/설명","steps":2},
  "compare": {"label":"비교·구분","material":"두 자료/두 사례","steps":2},
  "process": {"label":"과정·순서","material":"과정 자료","steps":2},
  "error_fix": {"label":"오류 판단·수정","material":"학생/교사 진술","steps":2},
  "cause_effect": {"label":"원인·효과 설명","material":"현상/조건","steps":2},
  "data_interpret": {"label":"자료 해석·근거","material":"표/수치/설명","steps":2},
  "calc": {"label":"계산·판단","material":"수치/도식","steps":2},
  "diagram": {"label":"도식·구조 해석","material":"구조/관계 자료","steps":2},
}

# 업로드된 2026 모의고사에서 확인되는 형식: 2점도 자료/수치 해석,
# 4점은 복수 빈칸·오류수정·계산·도식·과정/기능 서술이 혼합됨.
A_TYPES=["identify_apply","data_interpret","calc","diagram",
         "compare","process","error_fix","calc","cause_effect","data_interpret","diagram","process"]
B_TYPES=["identify_apply","data_interpret",
         "compare","process","calc","error_fix","cause_effect","data_interpret","diagram","calc","process"]

DOMAIN_PREFS={
 "기술교육론":["identify_apply","compare","process","error_fix","cause_effect","data_interpret"],
 "발명":["identify_apply","process","compare","error_fix","data_interpret"],
 "제조기술":["diagram","process","data_interpret","compare","error_fix","calc"],
 "건설기술":["diagram","process","data_interpret","cause_effect","compare","calc"],
 "생명기술":["process","data_interpret","compare","cause_effect","identify_apply"],
 "전기·전자":["data_interpret","diagram","identify_apply","cause_effect"],
 "통신기술":["data_interpret","process","diagram","calc","error_fix"],
 "재료역학":["calc","diagram","data_interpret","cause_effect"],
 "수송기술":["diagram","process","data_interpret","calc","cause_effect"],
}

def build_blueprint(section, scores, domains, rng=None):
    rng=rng or random.Random()
    types=(A_TYPES if section=="A" else B_TYPES)[:len(scores)]
    # 영역은 순환 배치하되 해당 영역과 유형의 궁합을 높임
    remaining=list(domains)
    rng.shuffle(remaining)
    plan=[]
    counts={d:0 for d in domains}
    for i,(pts,typ) in enumerate(zip(scores,types)):
        candidates=sorted(domains,key=lambda d:(counts[d], 0 if typ in DOMAIN_PREFS.get(d,[]) else 1, rng.random()))
        d=candidates[0]
        counts[d]+=1
        plan.append({"slot":i+1,"points":pts,"domain":d,"type":typ,"pattern":PATTERNS[typ]})
    return plan
