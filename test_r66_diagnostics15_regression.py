import capability_contracts as cc


def test_r66_catches_real_diagnostics15_cut_evidence():
    assert cc._obviously_incomplete_evidence('- 표면거칠기: 날끝이 둥근 노즈 반지름이 r(mm)인 공구로 이송 S(mm/rev)로 절삭할 때, 공작물이 날 형상대로')
    assert cc._obviously_incomplete_evidence('- 단면도의 이뿌리원 : 굵은 실선으로 그림 (단, 측면도에서')
    assert cc._obviously_incomplete_evidence('농약을 살포하') is False  # lexical stem alone is not over-vetoed


def test_r66_keeps_complete_pass_evidence():
    assert not cc._obviously_incomplete_evidence('· 동상 : 2개의 사인파 교류가 시간적으로 똑같은 경우')
    assert not cc._obviously_incomplete_evidence('▶ 디젤 사이클 디젤 기관의 사이클(1개씩의 정압과 정적 과정, 2개의 단열 과정) ※ 차단비 : 열 공급 과정에서 부피 변화의 비율 (예제 1)')
    assert not cc._obviously_incomplete_evidence('▶ 전시와 후시를 같게함으로써 제거되는 오차 - 시준축 오차와 지구의 곡률오차, 빛의 굴절오차 제거')


def test_r66_anchor_rows_remove_diagnostics15_cut_anchors():
    rows=cc._anchor_rows('knowledge.db','제조기술',limit=200)
    ids={int(x['id']) for x in rows}
    assert 1322 not in ids
    assert 1117 not in ids
    assert 1280 in ids and 1281 in ids


def test_r66_selector_prompt_has_reserve_backfill_and_difficulty_rule():
    p=cc._r67_selector_prompt('전기·전자', [], 3)
    assert 'reserve_ids' in p
    assert 'inferential_distance>=4' in p
    assert 'rote_risk<=1' in p
