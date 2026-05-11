-- Wavionex Investor Portal — PoC 04 (Vercel + Auth.js + Neon) schema.
-- Apply with:  psql "$DATABASE_URL" -f schema.sql

create extension if not exists pgcrypto;

create table if not exists users (
    id                 uuid primary key default gen_random_uuid(),
    email              text not null unique,
    password_hash      text not null,        -- argon2id-encoded string
    full_name          text,
    firm               text,
    email_verified_at  timestamptz,
    created_at         timestamptz not null default now(),
    updated_at         timestamptz not null default now()
);

create index if not exists users_email_idx on users (lower(email));

create table if not exists email_verifications (
    token       text primary key,            -- 32-byte CSPRNG hex
    user_id     uuid not null references users(id) on delete cascade,
    expires_at  timestamptz not null,
    used_at     timestamptz,
    created_at  timestamptz not null default now()
);

create index if not exists email_verifications_user_idx on email_verifications (user_id);

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
