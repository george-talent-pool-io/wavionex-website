-- Wavionex Investor Portal — production schema (v2).
-- Idempotent: safe to re-run against the existing live database.
-- Apply via the Supabase SQL editor.
--
-- Adds:
--   * invite-only signup (invite_codes + invite_redemptions, enforced by
--     a security-definer trigger on auth.users)
--   * approval gate (profiles.is_approved) — deals + papers require it
--   * admin role (profiles.is_admin) for the admin portal
--   * audit_events for "who looked at the portal" telemetry
--   * papers metadata table backed by a private Supabase Storage bucket
--   * Storage bucket policies on the `papers` bucket
--
-- Storage bucket creation cannot be done from SQL — create the `papers`
-- bucket via the dashboard first (Storage → New bucket → name: papers,
-- public: OFF). Then re-run this file to apply the bucket policies.

------------------------------------------------------------------
-- 0. Extensions
------------------------------------------------------------------
create extension if not exists pgcrypto;

------------------------------------------------------------------
-- 1. profiles — extend with status + admin + invite linkage
------------------------------------------------------------------
create table if not exists public.profiles (
    id          uuid primary key references auth.users(id) on delete cascade,
    email       text not null,
    full_name   text,
    firm        text,
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now()
);

alter table public.profiles add column if not exists is_approved        boolean      not null default false;
alter table public.profiles add column if not exists is_admin           boolean      not null default false;
alter table public.profiles add column if not exists email_verified_at  timestamptz;
alter table public.profiles add column if not exists invite_id          uuid;
alter table public.profiles add column if not exists last_sign_in_at    timestamptz;

create index if not exists profiles_is_approved_idx on public.profiles (is_approved);
create index if not exists profiles_is_admin_idx    on public.profiles (is_admin);
create index if not exists profiles_email_idx       on public.profiles (lower(email));

------------------------------------------------------------------
-- 2. invite codes & redemptions
------------------------------------------------------------------
create table if not exists public.invite_codes (
    id           uuid primary key default gen_random_uuid(),
    code         text not null unique,
    note         text,
    max_uses     integer not null default 1,
    used_count   integer not null default 0,
    expires_at   timestamptz,
    revoked_at   timestamptz,
    created_by   uuid references auth.users(id) on delete set null,
    created_at   timestamptz not null default now(),
    last_used_at timestamptz
);

create table if not exists public.invite_redemptions (
    id          uuid primary key default gen_random_uuid(),
    invite_id   uuid not null references public.invite_codes(id) on delete cascade,
    user_id     uuid not null references auth.users(id) on delete cascade,
    redeemed_at timestamptz not null default now(),
    unique (invite_id, user_id)
);

create index if not exists invite_redemptions_invite_idx on public.invite_redemptions (invite_id);

------------------------------------------------------------------
-- 3. audit events
------------------------------------------------------------------
create table if not exists public.audit_events (
    id          uuid primary key default gen_random_uuid(),
    user_id     uuid references auth.users(id) on delete set null,
    event_type  text not null,
    event_data  jsonb not null default '{}'::jsonb,
    user_agent  text,
    created_at  timestamptz not null default now()
);

create index if not exists audit_events_user_idx       on public.audit_events (user_id, created_at desc);
create index if not exists audit_events_type_idx       on public.audit_events (event_type, created_at desc);
create index if not exists audit_events_created_at_idx on public.audit_events (created_at desc);

------------------------------------------------------------------
-- 4. papers metadata (files themselves live in the `papers` Storage bucket)
------------------------------------------------------------------
create table if not exists public.papers (
    id           uuid primary key default gen_random_uuid(),
    title        text not null,
    description  text,
    storage_path text not null,
    file_size    bigint,
    mime_type    text,
    created_by   uuid references auth.users(id) on delete set null,
    created_at   timestamptz not null default now()
);

create index if not exists papers_created_at_idx on public.papers (created_at desc);

------------------------------------------------------------------
-- 5. deals — already exists; nothing structural changes
------------------------------------------------------------------
create table if not exists public.deals (
    id            uuid primary key default gen_random_uuid(),
    name          text not null,
    stage         text not null,
    target_close  date,
    headline      text,
    created_at    timestamptz not null default now()
);

------------------------------------------------------------------
-- 6. Helper functions (SECURITY DEFINER — bypass RLS for self-checks
--    so calls from inside an RLS policy don't recurse).
------------------------------------------------------------------
create or replace function public.current_user_is_admin()
returns boolean
language sql
security definer
stable
set search_path = public, auth
as $$
    select coalesce((select is_admin from public.profiles where id = auth.uid()), false);
$$;

create or replace function public.current_user_is_approved()
returns boolean
language sql
security definer
stable
set search_path = public, auth
as $$
    select coalesce((select is_approved from public.profiles where id = auth.uid()), false);
$$;

create or replace function public.current_user_is_privileged()
returns boolean
language sql
security definer
stable
set search_path = public, auth
as $$
    select coalesce((select is_approved or is_admin from public.profiles where id = auth.uid()), false);
$$;

------------------------------------------------------------------
-- 7. Trigger: new auth.users row → validate invite + create profile
------------------------------------------------------------------
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public, auth
as $$
declare
    v_code        text;
    v_invite_id   uuid;
    v_max_uses    int;
    v_used_count  int;
    v_revoked_at  timestamptz;
    v_expires_at  timestamptz;
begin
    v_code := nullif(trim(new.raw_user_meta_data ->> 'invite_code'), '');

    if v_code is null then
        raise exception 'invite_code is required';
    end if;

    select id, max_uses, used_count, revoked_at, expires_at
      into v_invite_id, v_max_uses, v_used_count, v_revoked_at, v_expires_at
    from public.invite_codes
    where lower(code) = lower(v_code)
    for update;

    if v_invite_id is null then
        raise exception 'invalid invite code';
    end if;
    if v_revoked_at is not null then
        raise exception 'invite code has been revoked';
    end if;
    if v_expires_at is not null and v_expires_at < now() then
        raise exception 'invite code has expired';
    end if;
    if v_used_count >= v_max_uses then
        raise exception 'invite code has no remaining uses';
    end if;

    update public.invite_codes
       set used_count   = used_count + 1,
           last_used_at = now()
     where id = v_invite_id;

    insert into public.invite_redemptions (invite_id, user_id)
    values (v_invite_id, new.id);

    insert into public.profiles (id, email, full_name, firm, invite_id, email_verified_at)
    values (
        new.id,
        new.email,
        nullif(trim(new.raw_user_meta_data ->> 'full_name'), ''),
        nullif(trim(new.raw_user_meta_data ->> 'firm'), ''),
        v_invite_id,
        new.email_confirmed_at
    )
    on conflict (id) do update set
        email     = excluded.email,
        full_name = coalesce(excluded.full_name, public.profiles.full_name),
        firm      = coalesce(excluded.firm, public.profiles.firm),
        invite_id = coalesce(public.profiles.invite_id, excluded.invite_id),
        updated_at = now();

    return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
    after insert on auth.users
    for each row execute function public.handle_new_user();

------------------------------------------------------------------
-- 8. Trigger: mirror email_confirmed_at + last_sign_in_at into profiles
------------------------------------------------------------------
create or replace function public.handle_user_update()
returns trigger
language plpgsql
security definer
set search_path = public, auth
as $$
begin
    update public.profiles
       set email_verified_at = new.email_confirmed_at,
           last_sign_in_at   = greatest(public.profiles.last_sign_in_at, new.last_sign_in_at),
           email             = new.email,
           updated_at        = now()
     where id = new.id;
    return new;
end;
$$;

drop trigger if exists on_auth_user_updated on auth.users;
create trigger on_auth_user_updated
    after update on auth.users
    for each row execute function public.handle_user_update();

------------------------------------------------------------------
-- 9. RLS policies
------------------------------------------------------------------

-- profiles -------------------------------------------------------
alter table public.profiles enable row level security;

drop policy if exists "profiles: owner can read"   on public.profiles;
drop policy if exists "profiles: owner can insert" on public.profiles;
drop policy if exists "profiles: owner can update" on public.profiles;
drop policy if exists "profiles: admin full read"  on public.profiles;
drop policy if exists "profiles: admin full write" on public.profiles;
drop policy if exists "profiles: self read"        on public.profiles;
drop policy if exists "profiles: self update non-privileged fields" on public.profiles;

create policy "profiles: self read"
    on public.profiles for select
    using (auth.uid() = id);

create policy "profiles: self update non-privileged fields"
    on public.profiles for update
    using (auth.uid() = id)
    with check (auth.uid() = id);

create policy "profiles: admin full read"
    on public.profiles for select
    using (public.current_user_is_admin());

create policy "profiles: admin full write"
    on public.profiles for all
    using (public.current_user_is_admin())
    with check (public.current_user_is_admin());

-- Block non-admin users from changing privileged columns. RLS WITH CHECK
-- can't compare OLD vs NEW directly, so we enforce with a trigger.
create or replace function public.guard_profile_privileged_fields()
returns trigger
language plpgsql
security definer
set search_path = public, auth
as $$
begin
    -- No JWT in context → admin / SQL editor / service_role / system trigger; allow.
    if auth.uid() is null then
        return new;
    end if;
    -- Authenticated admin; allow.
    if public.current_user_is_admin() then
        return new;
    end if;
    if new.is_admin          is distinct from old.is_admin          then raise exception 'only admins may change is_admin';          end if;
    if new.is_approved       is distinct from old.is_approved       then raise exception 'only admins may change is_approved';       end if;
    if new.invite_id         is distinct from old.invite_id         then raise exception 'invite_id is immutable for non-admin';     end if;
    if new.email_verified_at is distinct from old.email_verified_at then raise exception 'email_verified_at is system-managed';      end if;
    if new.last_sign_in_at   is distinct from old.last_sign_in_at   then raise exception 'last_sign_in_at is system-managed';        end if;
    return new;
end;
$$;

drop trigger if exists trg_profiles_guard on public.profiles;
create trigger trg_profiles_guard
    before update on public.profiles
    for each row execute function public.guard_profile_privileged_fields();

create or replace function public.touch_profiles_updated_at()
returns trigger language plpgsql as $$
begin
    new.updated_at := now();
    return new;
end $$;

drop trigger if exists trg_profiles_updated_at on public.profiles;
create trigger trg_profiles_updated_at
    before update on public.profiles
    for each row execute function public.touch_profiles_updated_at();

-- invite_codes ---------------------------------------------------
alter table public.invite_codes enable row level security;

drop policy if exists "invite_codes: admin read"  on public.invite_codes;
drop policy if exists "invite_codes: admin write" on public.invite_codes;

create policy "invite_codes: admin read"  on public.invite_codes for select using (public.current_user_is_admin());
create policy "invite_codes: admin write" on public.invite_codes for all
    using (public.current_user_is_admin())
    with check (public.current_user_is_admin());

-- The signup trigger uses SECURITY DEFINER so it bypasses RLS — no policy needed for anon writes.

-- invite_redemptions --------------------------------------------
alter table public.invite_redemptions enable row level security;

drop policy if exists "invite_redemptions: admin read" on public.invite_redemptions;
drop policy if exists "invite_redemptions: owner read" on public.invite_redemptions;

create policy "invite_redemptions: owner read"
    on public.invite_redemptions for select
    using (user_id = auth.uid());

create policy "invite_redemptions: admin read"
    on public.invite_redemptions for select
    using (public.current_user_is_admin());

-- audit_events --------------------------------------------------
alter table public.audit_events enable row level security;

drop policy if exists "audit_events: insert own" on public.audit_events;
drop policy if exists "audit_events: admin read" on public.audit_events;

create policy "audit_events: insert own"
    on public.audit_events for insert
    with check (auth.uid() = user_id);

create policy "audit_events: admin read"
    on public.audit_events for select
    using (public.current_user_is_admin());

-- papers (metadata) ---------------------------------------------
alter table public.papers enable row level security;

drop policy if exists "papers: approved read" on public.papers;
drop policy if exists "papers: admin write"   on public.papers;

create policy "papers: approved read"
    on public.papers for select
    using (public.current_user_is_privileged());

create policy "papers: admin write"
    on public.papers for all
    using (public.current_user_is_admin())
    with check (public.current_user_is_admin());

-- deals ---------------------------------------------------------
alter table public.deals enable row level security;

drop policy if exists "deals: confirmed users can read" on public.deals;
drop policy if exists "deals: approved users can read"  on public.deals;
drop policy if exists "deals: admin write"              on public.deals;

create policy "deals: approved users can read"
    on public.deals for select
    using (
        (auth.jwt() ->> 'email_confirmed_at') is not null
        and public.current_user_is_privileged()
    );

create policy "deals: admin write"
    on public.deals for all
    using (public.current_user_is_admin())
    with check (public.current_user_is_admin());

------------------------------------------------------------------
-- 10. Storage bucket policies for `papers`
--    Create the bucket in Storage UI first (private, name = 'papers').
------------------------------------------------------------------
drop policy if exists "papers bucket: approved read" on storage.objects;
drop policy if exists "papers bucket: admin write"   on storage.objects;

create policy "papers bucket: approved read"
    on storage.objects for select
    using (
        bucket_id = 'papers'
        and public.current_user_is_privileged()
    );

create policy "papers bucket: admin write"
    on storage.objects for all
    using (bucket_id = 'papers' and public.current_user_is_admin())
    with check (bucket_id = 'papers' and public.current_user_is_admin());

------------------------------------------------------------------
-- 11. Backfill columns added in v2 for users who existed pre-trigger.
--     Idempotent: only touches rows where the field is still null.
------------------------------------------------------------------
update public.profiles p
   set email_verified_at = u.email_confirmed_at
  from auth.users u
 where p.id = u.id
   and p.email_verified_at is null
   and u.email_confirmed_at is not null;

update public.profiles p
   set last_sign_in_at = u.last_sign_in_at
  from auth.users u
 where p.id = u.id
   and p.last_sign_in_at is null
   and u.last_sign_in_at is not null;

-- Make sure every auth.users row has a matching profiles row (pre-v2 users
-- created their profile via the old client-side upsert; just in case anyone
-- slipped through, copy them across now).
insert into public.profiles (id, email, full_name, email_verified_at, last_sign_in_at)
select u.id,
       u.email,
       nullif(trim(u.raw_user_meta_data ->> 'full_name'), ''),
       u.email_confirmed_at,
       u.last_sign_in_at
  from auth.users u
 where not exists (select 1 from public.profiles p where p.id = u.id)
on conflict (id) do nothing;

------------------------------------------------------------------
-- 12. Seed deals if empty (idempotent on row count)
------------------------------------------------------------------
insert into public.deals (name, stage, target_close, headline)
select * from (values
    ('Wave Core IP Round',        'Seed',     date '2026-09-15', 'Foundational wave-computing patent portfolio licensing.'),
    ('Photonic Substrate Pilot',  'Pre-A',    date '2026-12-01', 'First-of-kind programmable wave fabric on photonic substrate.'),
    ('Edge Analog Pilot',         'Pre-Seed', date '2027-02-28', 'Sub-watt analog inference for edge sensor fusion.')
) as v(name, stage, target_close, headline)
where not exists (select 1 from public.deals);
