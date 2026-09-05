"""Certification evidence and rerun-safe imports. No Streamlit or DB mutation.

Receipts detect accidental edits; they are not signatures proving remote API use.
"""
import copy
import hashlib
import json
import math
from quality_judge import SCORE_KEYS


def review_passes(review):
    if not isinstance(review, dict) or review.get('pass') is not True:
        return False
    if review.get('blind_verdict') != 'PASS' or review.get('fatal_flags') != []:
        return False
    try:
        scores = [float(review['scores'][k]) for k in SCORE_KEYS]
    except (KeyError, TypeError, ValueError):
        return False
    return (all(math.isfinite(v) and 0 <= v <= 5 for v in scores)
            and all(v >= (3.5 if k == 'inferential_distance' else 4)
                    for k, v in zip(SCORE_KEYS, scores))
            and sum(scores) / len(scores) >= 4)


def contract_digest(contract):
    excluded = {'validation', 'status', 'ai_quality', 'judge_model', 'certification'}
    content = {k: v for k, v in contract.items() if k not in excluded}
    return hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode('utf-8')).hexdigest()


def attach_receipt(contract):
    if not review_passes(contract.get('ai_quality')):
        raise ValueError('A complete passing Judge review is required')
    contract['certification'] = {'schema': 1, 'content_sha256': contract_digest(contract)}


def verified_contract_receipt(contract):
    if not isinstance(contract, dict) or contract.get('status') != 'R59_AI_VERIFIED':
        return False
    if not review_passes(contract.get('ai_quality')) or not contract.get('judge_model'):
        return False
    try:
        receipt = contract.get('certification') or {}
        return receipt.get('schema') == 1 and receipt.get('content_sha256') == contract_digest(contract)
    except (ValueError, TypeError):
        return False


def upsert_contracts(existing, incoming):
    result = copy.deepcopy(list(existing))
    indices = {x.get('contract_id'): i for i, x in enumerate(result)
               if isinstance(x, dict) and x.get('contract_id')}
    for row in incoming:
        row = copy.deepcopy(row)
        key = row.get('contract_id')
        if key and key in indices:
            result[indices[key]] = row
        else:
            if key:
                indices[key] = len(result)
            result.append(row)
    return result


def import_contract_upload(state, raw):
    sha = hashlib.sha256(raw).hexdigest()
    if state.get('CONTRACT_IMPORT_SHA') == sha:
        return None
    obj = json.loads(raw.decode('utf-8-sig'))
    if not isinstance(obj, dict) or not isinstance(obj.get('contracts'), list):
        raise ValueError('contracts 배열이 있는 JSON 객체가 필요합니다.')
    rows = obj['contracts']
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError('contracts 항목은 모두 JSON 객체여야 합니다.')
    merged = upsert_contracts(state.get('R59_CONTRACTS', []), rows)
    # Commit both state keys only after the complete upload has been validated.
    state['R59_CONTRACTS'] = merged
    state['CONTRACT_IMPORT_SHA'] = sha
    state.pop('R59_CERT_RUN', None)
    return len(rows)
