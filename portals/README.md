# Wavionex Investor Portal — Proof-of-Concept matrix

Five separate, end-to-end implementations of the same investor portal — same
look, same minimum feature set, different auth + storage stacks. The intent
is to compare operational feel, security ceremony, and migration risk before
committing to one for production.

| #  | Folder                | Stack                                        | Hosts on             | Auth UX owner                 | Data store     |
|----|-----------------------|----------------------------------------------|----------------------|-------------------------------|----------------|
| 01 | `01-supabase/`        | vanilla HTML + Supabase JS                   | GitHub Pages         | Supabase hosted               | Postgres + RLS |
| 02 | `02-cloudflare/`      | static + CF Pages Functions + Cloudflare Access | Cloudflare Pages | Cloudflare (edge)             | D1 (SQLite)    |
| 03 | `03-clerk-neon/`      | static + CF Pages Functions + Clerk + jose   | Cloudflare Pages     | Clerk hosted (slickest UI)    | Neon Postgres  |
| 04 | `04-vercel-authjs/`   | Next.js + Auth.js + argon2id                 | Vercel               | Built in-house                | Neon Postgres  |
| 05 | `05-firebase/`        | vanilla HTML + Firebase Web SDK              | GitHub Pages         | Firebase hosted               | Firestore      |

Each subdirectory has its own `README.md` with the exact account-creation and
deploy commands. Every PoC ships the same five things end-to-end:

1. Sign-up with full name, firm, email, password
2. Email verification before privileged data is unlocked
3. Sign-in
4. Protected dashboard showing the user's profile from the data store
5. A read-only stub "deals" panel readable only by authenticated + verified users

## Shared bits

- `portals/_shared/styles.css` — Wavionex palette + typography in plain CSS;
  imported by every static front-end. The Next.js PoC ships a copy in
  `04-vercel-authjs/app/globals.css` because Vercel can't reach back into
  this folder at runtime. Keep them in sync if you edit either.
- The public landing page that links every PoC together is
  `../portal-poc-comparison.html` at the site root, reachable from the nav
  ("Investor Portal").

## Recommended evaluation order

1. **Supabase** first — least configuration, gives you a working portal in
   ~30 minutes and validates whether the design + flow is what you want.
2. **Clerk + Neon** next — same Postgres backing as Vercel/Auth.js but with
   a hosted sign-up UI, so it's a like-for-like comparison of the slick-UI
   vs. fully-custom-UI tradeoff.
3. **Cloudflare Access** if you decide the portal will be invite-only —
   nothing else gets close on operational simplicity once you accept the
   50-user free-tier cap.
4. **Vercel + Auth.js** only if (3) doesn't fit and you need a fully owned
   user table + flow (e.g. for compliance reasons).
5. **Firebase** as a sanity check against the Postgres-based options —
   shows the cost of NoSQL data sovereignty.

## Things that are deliberately NOT in any PoC

- MFA enrolment flow — most providers support it natively (Clerk, Supabase,
  Firebase, CF Access); the in-app screens to enrol/disenrol aren't built.
- Password reset UI for the Auth.js PoC — schema + flow is sketched in the
  README but the actual page isn't wired.
- Admin / RBAC role — every authenticated, verified user sees the same deal
  list. Real investor portals will need per-investor scoping.
- Persistence of audit / sign-in events.
- Production rate-limiting policies on auth endpoints (per-PoC notes
  reference the right place to add them).

Treat these as PoCs — they are wired correctly enough that the auth flow is
real, but ship a security review pass and the gaps above before pointing
real investors at any of them.
