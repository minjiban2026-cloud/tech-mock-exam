"""Source-bound answer plans fixed before the Writer runs.

These checks establish provenance and explicit dependency structure, not semantic
truth or exam difficulty. The independent Judge retains its full veto.
"""
import re

PLAN_SCHEMA='SOURCE_BOUND_TASK_PLAN_V1'


def clean(text):
    return re.sub(r'\s+', ' ', str(text or '').replace('\x01', ' ')).strip()


def bind(ref, anchors):
    if not isinstance(ref,dict) or type(ref.get('anchor_id')) is not int:
        raise ValueError('BINDING_OBJECT_REQUIRED')
    anchor=anchors.get(ref['anchor_id'])
    quote=clean(ref.get('quote'))
    if not anchor or len(quote)<12 or quote not in clean(anchor.get('evidence')):
        raise ValueError('EXACT_SOURCE_QUOTE_REQUIRED')
    if quote==clean(anchor.get('answer')):
        raise ValueError('NAME_ONLY_ANSWER')
    return quote


def validate_plan(plan, source_anchors):
    errors=[]
    if not isinstance(plan,dict) or plan.get('schema')!=PLAN_SCHEMA:
        return False, {'errors':['SOURCE_BOUND_PLAN_REQUIRED']}
    amap={int(a['id']):a for a in source_anchors}
    try:
        criterion=bind(plan.get('criterion'),amap)
        parts=[]
        for name in ('task1','task2'):
            task=plan.get(name)
            if not isinstance(task,dict): raise ValueError('TASK_PLAN_REQUIRED')
            result=bind(task.get('result'),amap)
            reason=bind(task.get('reason'),amap)
            if result==reason: raise ValueError('RESULT_AND_REASON_IDENTICAL')
            parts.append((result,reason))
        if parts[0]==parts[1]: raise ValueError('DUPLICATE_TASK_ANSWERS')
        dependency=plan.get('dependency')
        if not isinstance(dependency,dict) or dependency.get('input')!='task1.result' or dependency.get('output')!='task2.result':
            raise ValueError('TASK2_INPUT_NOT_TASK1_RESULT')
        # Bind the intermediate conclusion to actual source text, not a boolean declaration.
        required=bind(dependency.get('required_result'),amap)
        if required!=parts[0][0]: raise ValueError('DEPENDENCY_RESULT_MISMATCH')
        transfer=bind(plan.get('transfer_condition'),amap)
        if transfer==criterion: raise ValueError('TRANSFER_REPEATS_CRITERION')
        if len(clean(dependency.get('why_required')))<24:
            raise ValueError('DEPENDENCY_EXPLANATION_REQUIRED')
    except (ValueError,TypeError,KeyError) as ex:
        errors.append(str(ex))
        return False, {'errors':errors}
    return True, {'errors':[], 'criterion':criterion, 'transfer_condition':transfer,
                  'answers':[f'{result}\n근거: {reason}' for result,reason in parts],
                  'rubric':[{'result':result,'reason':reason,'points':2} for result,reason in parts]}

