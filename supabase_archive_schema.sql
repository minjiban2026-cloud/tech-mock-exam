
-- Supabase SQL Editor에서 한 번만 실행하세요.
create extension if not exists pgcrypto;

create table if not exists public.generated_exams (
  id uuid primary key default gen_random_uuid(),
  title text not null,
  note text not null default '',
  model text not null default '',
  seed bigint,
  domains jsonb not null default '[]'::jsonb,
  exam_a jsonb not null,
  exam_b jsonb not null,
  manually_edited boolean not null default false,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists generated_exams_created_at_idx
  on public.generated_exams (created_at desc);

-- 이 앱은 Streamlit 서버의 Secrets에 SERVICE ROLE KEY를 넣어 사용하도록 설계했습니다.
-- service_role은 RLS를 우회하므로 외부 브라우저에 키가 노출되지 않습니다.
alter table public.generated_exams enable row level security;

-- anon/public 접근 정책은 의도적으로 만들지 않습니다.
-- 반드시 Streamlit Secrets의 SUPABASE_SERVICE_ROLE_KEY를 사용하세요.
