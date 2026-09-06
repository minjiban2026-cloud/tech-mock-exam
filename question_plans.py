"""Source-bound answer plans fixed before Writer execution.

Provenance and structural checks are necessary but do not prove semantic
correctness or exam difficulty. Judge retains its independent veto.
"""
import re

PLAN_SCHEMA='SOURCE_BOUND_TASK_PLAN_V1'


def clean(text):
    return re.sub(r'\s+', ' ', str(text or '').replace('\x01', ' ')).strip()


def bind(ref, anchors, *, role='evidence'):
    if not isinstance(ref,dict) or type(ref.get('anchor_id')) is not int or not isinstance(ref.get('quote'),str):
        raise ValueError('BINDING_OBJECT_REQUIRED')
    anchor=anchors.get(ref['anchor_id'])
    quote=clean(ref['quote'])
    minimum=1 if role=='result' else 12
    if not anchor or len(quote)<minimum or quote not in clean(anchor.get('evidence')):
        raise ValueError('EXACT_SOURCE_QUOTE_REQUIRED')
    # A result can be a short selected value. A reason cannot be just its name.
    if role!='result' and quote==clean(anchor.get('answer')):
        raise ValueError('NAME_ONLY_ANSWER')
    if role=='result' and re.fullmatch(r'[A-Za-z0-9]+',quote):
        if not re.search(r'(?<![A-Za-z0-9])'+re.escape(quote)+r'(?![A-Za-z0-9])',clean(anchor['evidence'])):
            raise ValueError('PARTIAL_RESULT_TOKEN')
    return quote


def validate_plan(plan, source_anchors, relation_type=None):
    errors=[];details=[]
    def reject(code,path):
        if code not in errors:errors.append(code)
        details.append({'code':code,'path':path})
    if not isinstance(plan,dict) or plan.get('schema')!=PLAN_SCHEMA:
        return False, {'errors':['SOURCE_BOUND_PLAN_REQUIRED'],'error_details':[{'code':'SOURCE_BOUND_PLAN_REQUIRED','path':'source_plan'}]}
    amap={int(a['id']):a for a in source_anchors}
    def bound(ref,path,role='evidence'):
        # R59 selector outputs are allowed to carry explanatory text/value wrappers
        # around the actual source binding.  The canonical provenance object is
        # still {anchor_id, quote}; validation always resolves back to that object.
        try:
            if isinstance(ref,dict) and type(ref.get('anchor_id')) is int and isinstance(ref.get('quote'),str):
                return bind(ref,amap,role=role)
            if isinstance(ref,dict) and isinstance(ref.get('binding'),dict):
                return bind(ref['binding'],amap,role=role)
            if isinstance(ref,dict) and isinstance(ref.get('bindings'),list):
                refs=ref['bindings']
                if not refs:
                    raise ValueError('BINDING_OBJECT_REQUIRED')
                if role=='result' and len(refs)!=1:
                    raise ValueError('SINGLE_RESULT_BINDING_REQUIRED')
                vals=[bind(x,amap,role=role) for x in refs]
                return ' / '.join(vals)
            raise ValueError('BINDING_OBJECT_REQUIRED')
        except (ValueError,TypeError,KeyError) as ex:
            reject(str(ex),path);return None
    criterion=bound(plan.get('criterion'),'criterion')
    transfer=bound(plan.get('transfer_condition'),'transfer_condition')
    # Structural leakage checks must inspect what the examinee will actually see,
    # not the hidden provenance quote. A provenance quote can legitimately contain
    # the scored result while the public transfer_condition paraphrases only the
    # condition. Using the hidden quote here caused valid plans to be rejected.
    def surface(ref, fallback):
        if isinstance(ref,dict):
            for key in ('text','value'):
                value=clean(ref.get(key))
                if value:
                    return value
        return fallback
    criterion_surface=surface(plan.get('criterion'),criterion)
    transfer_surface=surface(plan.get('transfer_condition'),transfer)
    parts=[]
    for name in ('task1','task2'):
        task=plan.get(name)
        if not isinstance(task,dict):
            reject('TASK_PLAN_REQUIRED',name);parts.append((None,None));continue
        result=bound(task.get('result'),name+'.result','result')
        reason=bound(task.get('reason'),name+'.reason')
        if result is not None and reason is not None and result==reason:
            reject('RESULT_AND_REASON_IDENTICAL',name)
        parts.append((result,reason))
    if all(x is not None for pair in parts for x in pair) and parts[0]==parts[1]:
        reject('DUPLICATE_TASK_ANSWERS','task2')
    dependency=plan.get('dependency')
    if not isinstance(dependency,dict):
        reject('TASK2_INPUT_NOT_TASK1_RESULT','dependency')
    else:
        if dependency.get('input')!='task1.result' or dependency.get('output')!='task2.result':
            reject('TASK2_INPUT_NOT_TASK1_RESULT','dependency')
        required=bound(dependency.get('required_result'),'dependency.required_result','result')
        if required is not None and parts[0][0] is not None and required!=parts[0][0]:
            reject('DEPENDENCY_RESULT_MISMATCH','dependency.required_result')
        if len(clean(dependency.get('why_required')))<24:
            reject('DEPENDENCY_EXPLANATION_REQUIRED','dependency.why_required')
    if (transfer_surface is not None and criterion_surface is not None and
            len(clean(transfer_surface))>=8 and len(clean(criterion_surface))>=8 and
            transfer_surface==criterion_surface):
        reject('TRANSFER_REPEATS_CRITERION','transfer_condition')
    if transfer_surface is not None and parts[1][0] is not None and clean(parts[1][0]) in clean(transfer_surface):
        reject('TRANSFER_DISCLOSES_TASK2_RESULT','transfer_condition')
    if relation_type=='conditional_choice':
        condition_markers=r'경우|조건|때|이면|하면|이상|이하|초과|미만|따라|비교|대비|반면|보다'
        # A menu identifies possible options but supplies no basis for choosing one.
        if criterion and len(re.findall(r'[,，、]',criterion))>=2 and not re.search(condition_markers,criterion):
            reject('CHOICE_CRITERION_IS_CATALOG','criterion')
        if parts[0][0] and len(re.findall(r'[,，、]',parts[0][0]))>=2:
            reject('CHOICE_RESULT_HAS_MULTIPLE_OPTIONS','task1.result')
    if errors:return False, {'errors':errors,'error_details':details}
    return True, {'errors':[], 'error_details':[], 'criterion':criterion,'transfer_condition':transfer,
                  'answers':[f'{result}\n근거: {reason}' for result,reason in parts],
                  'rubric':[{'result':result,'reason':reason,'points':2} for result,reason in parts]}
