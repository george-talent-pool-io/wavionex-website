-- Wavionex Investor Portal — PoC 03 (Clerk + Neon) schema.
-- Run in Neon SQL editor against your project's database.

create extension if not exists pgcrypto;

create table if not exists profiles (
    clerk_user_id  text primary key,
    email          text not null,
    full_name      text,
    firm           text,
    created_at     timestamptz not null default now(),
    updated_at     timestamptz not null default now()
);

create table if not exists deals (
    id            uuid primary key default gen_random_uuid(),
    name          text not null,
    stage         text not null,
    target_close  date,
    headline      text,
    created_at    timestamptz not null default now()
);

insert into deals (name, stage, target_close, headline) values
    ('Wave Core IP Round',        'Seed',     date '2026-09-15', 'Foundational wave-computing patent portfolio licensing.'),
    ('Photonic Substrate Pilot',  'Pre-A',    date '2026-12-01', 'First-of-kind programmable wave fabric on photonic substrate.'),
    ('Edge Analog Pilot',         'Pre-Seed', date '2027-02-28', 'Sub-watt analog inference for edge sensor fusion.')
on conflict do nothing;

-- Authorisation lives in the Pages Function (functions/api/*.js): the JWT
-- is verified there before any query runs, and we filter rows by the
-- `sub` claim. Neon role attached to DATABASE_URL should have the
-- minimum privileges: SELECT/INSERT/UPDATE on profiles + SELECT on deals.
