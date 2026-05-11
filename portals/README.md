# Wavionex Investor Portal

The chosen stack is **Supabase** (PoC 01, now living at `investor/`). The
other four PoCs are retained for reference — they can be deleted at any
time without affecting the live portal.

| Folder              | Status                       | Stack                                              |
|---------------------|------------------------------|----------------------------------------------------|
| `investor/`         | **Production** (was PoC 01)  | vanilla HTML + Supabase JS                         |
| `02-cloudflare/`    | Reference                    | static + CF Pages Functions + CF Access + D1       |
| `03-clerk-neon/`    | Reference                    | static + CF Pages Functions + Clerk + Neon         |
| `04-vercel-authjs/` | Reference                    | Next.js + Auth.js + argon2id + Neon                |
| `05-firebase/`      | Reference                    | vanilla HTML + Firebase Web SDK                    |

Production portal: <https://www.wavionex.com/portals/investor/>
Production setup notes: `investor/README.md`.

## Original PoC matrix (for posterity)

Five separate end-to-end implementations of the same investor portal — same
look, same minimum feature set, different auth + storage stacks. Each
subdirectory has its own README with account-creation + deploy steps.

## Shared bits

- `_shared/styles.css` — Wavionex palette + typography. Imported by every
  static front-end. The Next.js PoC ships a copy in
  `04-vercel-authjs/app/globals.css` because Vercel can't reach back into
  this folder at runtime; keep them in sync if you edit either.
- `../portal-poc-comparison.html` — the comparison landing page used during
  the bake-off. No longer linked from the marketing nav.

## What's still missing on the production portal

Tracked in `investor/README.md` under "Known gaps", in short:

- Invite-only sign-up (currently anyone can register)
- Approved-investor gate on deals (currently every verified user sees them)
- Document/PDF gating via Supabase Storage
- CAPTCHA on auth endpoints
- MFA enrolment UI

These are the next pieces of work before pointing real investors at it.
