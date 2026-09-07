
import json, urllib.request, urllib.parse, urllib.error
from datetime import datetime, timezone

TABLE = "generated_exams"
CERT_STATE_TITLE = "__SYSTEM_CAPABILITY_CERTIFICATION_STATE__"

def _clean_base(url):
    return (url or "").rstrip("/")

def _headers(key, prefer=None):
    h={
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if prefer:
        h["Prefer"]=prefer
    return h

def _request(url, key, method="GET", payload=None, prefer=None):
    data=None if payload is None else json.dumps(payload,ensure_ascii=False).encode("utf-8")
    req=urllib.request.Request(url,data=data,headers=_headers(key,prefer),method=method)
    try:
        with urllib.request.urlopen(req,timeout=20) as r:
            raw=r.read().decode("utf-8")
            return json.loads(raw) if raw.strip() else None
    except urllib.error.HTTPError as e:
        body=e.read().decode("utf-8","replace")
        raise RuntimeError(f"Supabase 오류 {e.code}: {body[:1000]}")
    except Exception as e:
        raise RuntimeError(f"Supabase 연결 실패: {e}")

def is_configured(url,key):
    return bool(_clean_base(url) and key)

def ping(url,key):
    base=_clean_base(url)
    q=urllib.parse.urlencode({"select":"id","limit":"1"})
    _request(f"{base}/rest/v1/{TABLE}?{q}",key)
    return True

def list_exams(url,key,limit=100):
    base=_clean_base(url)
    q=urllib.parse.urlencode({
        "select":"id,title,note,model,seed,domains,created_at,updated_at,manually_edited",
        "title":f"neq.{CERT_STATE_TITLE}",
        "order":"created_at.desc",
        "limit":str(limit)
    })
    return _request(f"{base}/rest/v1/{TABLE}?{q}",key) or []

def get_exam(url,key,exam_id):
    base=_clean_base(url)
    q=urllib.parse.urlencode({
        "id":f"eq.{exam_id}",
        "select":"*"
    })
    rows=_request(f"{base}/rest/v1/{TABLE}?{q}",key) or []
    return rows[0] if rows else None

def create_exam(url,key,record):
    base=_clean_base(url)
    rows=_request(
        f"{base}/rest/v1/{TABLE}",key,"POST",record,
        prefer="return=representation"
    ) or []
    return rows[0] if rows else None

def update_exam(url,key,exam_id,patch):
    base=_clean_base(url)
    patch=dict(patch)
    patch["updated_at"]=datetime.now(timezone.utc).isoformat()
    q=urllib.parse.urlencode({"id":f"eq.{exam_id}"})
    rows=_request(
        f"{base}/rest/v1/{TABLE}?{q}",key,"PATCH",patch,
        prefer="return=representation"
    ) or []
    return rows[0] if rows else None

def delete_exam(url,key,exam_id):
    base=_clean_base(url)
    q=urllib.parse.urlencode({"id":f"eq.{exam_id}"})
    _request(
        f"{base}/rest/v1/{TABLE}?{q}",key,"DELETE",None,
        prefer="return=minimal"
    )
    return True


def load_certification_state(url,key):
    """Load durable Judge-PASS contracts from the existing archive table.

    Uses one hidden system row so no Supabase schema migration is required.
    """
    base=_clean_base(url)
    q=urllib.parse.urlencode({
        "title":f"eq.{CERT_STATE_TITLE}",
        "select":"id,exam_a,exam_b,updated_at",
        "order":"updated_at.desc",
        "limit":"1",
    })
    rows=_request(f"{base}/rest/v1/{TABLE}?{q}",key) or []
    if not rows:
        return {"contracts":[],"record_id":None}
    row=rows[0]
    payload=row.get("exam_a")
    if not isinstance(payload,dict):
        payload={}
    contracts=payload.get("contracts")
    if not isinstance(contracts,list):
        contracts=[]
    return {"contracts":contracts,"record_id":row.get("id"),"updated_at":row.get("updated_at")}


def save_certification_state(url,key,contracts,inventory=None):
    """Upsert the hidden durable certification row."""
    base=_clean_base(url)
    state=load_certification_state(url,key)
    payload_a={"schema_version":"R62-CERTIFICATION-STATE-V1","contracts":list(contracts or [])}
    payload_b={"inventory":inventory or {}}
    record={
        "title":CERT_STATE_TITLE,
        "note":"Internal Judge-PASS capability certification state. Hidden from archive list.",
        "model":"system-certification-state",
        "seed":None,
        "domains":[],
        "exam_a":payload_a,
        "exam_b":payload_b,
        "manually_edited":False,
    }
    rid=state.get("record_id")
    if rid:
        return update_exam(url,key,rid,record)
    return create_exam(url,key,record)
