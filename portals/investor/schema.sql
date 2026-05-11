-- Wavionex Investor Portal — PoC 01 (Supabase) schema.
-- Paste this into the Supabase SQL editor after creating the project.
-- It is idempotent: safe to re-run.

------------------------------------------------------------------
-- profiles: one row per auth.users entry, owner-only read/write.
------------------------------------------------------------------
create table if not exists public.profiles (
    id          uuid primary key references auth.users(id) on delete cascade,
    email       text not null,
    full_name   text,
    firm        text,
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now()
);

alter table public.profiles enable row level security;

drop policy if exists "profiles: owner can read"   on public.profiles;
drop policy if exists "profiles: owner can insert" on public.profiles;
drop policy if exists "profiles: owner can update" on public.profiles;

create policy "profiles: owner can read"
    on public.profiles for select
    using (auth.uid() = id);

create policy "profiles: owner can insert"
    on public.profiles for insert
    with check (auth.uid() = id);

create policy "profiles: owner can update"
    on public.profiles for update
    using (auth.uid() = id)
    with check (auth.uid() = id);

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

------------------------------------------------------------------
-- deals: read-only for any *confirmed* authenticated user.
-- Writes only via the service role (i.e. admin / dashboard).
------------------------------------------------------------------
create table if not exists public.deals (
    id            uuid primary key default gen_random_uuid(),
    name          text not null,
    stage         text not null,
    target_close  date,
    headline      text,
    created_at    timestamptz not null default now()
);

alter table public.deals enable row level security;

drop policy if exists "deals: confirmed users can read" on public.deals;

create policy "deals: confirmed users can read"
    on public.deals for select
    using (
        auth.role() = 'authenticated'
        and (auth.jwt() ->> 'email_confirmed_at') is not null
    );

------------------------------------------------------------------
-- Seed a few stub deals so the dashboard renders something.
------------------------------------------------------------------
insert into public.deals (name, stage, target_close, headline) values
    ('Wave Core IP Round',         'Seed',     date '2026-09-15', 'Foundational wave-computing patent portfolio licensing.'),
    ('Photonic Substrate Pilot',  'Pre-A',   date '2026-12-01', 'First-of-kind programmable wave fabric on photonic substrate.'),
    ('Edge Analog Pilot',         'Pre-Seed', date '2027-02-28', 'Sub-watt analog inference for edge sensor fusion.')
on conflict do nothing;
