-- Wavionex Investor Portal — PoC 02 (Cloudflare Access + D1).
-- Apply with:  wrangler d1 execute wavionex-portal-d1 --file=schema.sql

create table if not exists profiles (
    email             text primary key,
    full_name         text,
    firm              text,
    identity_provider text,
    last_login_at     text,
    created_at        text not null
);

create table if not exists deals (
    id            text primary key,
    name          text not null,
    stage         text not null,
    target_close  text,
    headline      text,
    created_at    text not null default (datetime('now'))
);

-- Seed three stub deals (idempotent on id).
insert or ignore into deals (id, name, stage, target_close, headline) values
    ('seed-1', 'Wave Core IP Round',        'Seed',     '2026-09-15', 'Foundational wave-computing patent portfolio licensing.'),
    ('seed-2', 'Photonic Substrate Pilot',  'Pre-A',    '2026-12-01', 'First-of-kind programmable wave fabric on photonic substrate.'),
    ('seed-3', 'Edge Analog Pilot',         'Pre-Seed', '2027-02-28', 'Sub-watt analog inference for edge sensor fusion.');
