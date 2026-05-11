# PoC 01 — Supabase (Postgres + auth in one)

Single-vendor stack: front-end is plain HTML/JS served from GitHub Pages, talks
directly to Supabase. Auth + email verification + password reset are handled by
Supabase; data security is enforced by Row-Level Security policies in Postgres.

## Architecture

````
Browser  ─►  Supabase Auth   (sign-up, login, email confirm, password reset)
        ─►  Supabase REST   (profiles + deals tables, gated by RLS)

Hosting:   GitHub Pages  →  portals/01-supabase/  (this directory)
Auth/DB:   Supabase project (free tier)
```

## What you need to do once

1. Create a free project at <https://supabase.com>.
2. **Project Settings → API** — copy:
   - `Project URL`
   - `anon` public key (the one labelled "publishable")
3. Copy `config.example.js` to `config.js` and paste the values in. `config.js`
   is git-ignored (see the root `.gitignore`, add an entry if missing).
4. Open **SQL editor** in Supabase and run the contents of `schema.sql`.
5. **Authentication → URL Configuration**:
   - Site URL: `https://<your-domain>/portals/01-supabase/`
   - Add the same URL under "Redirect URLs".
6. **Authentication → Providers → Email**: ensure "Confirm email" is on.

## What you need to do every time you deploy

Nothing — the directory is static. Push to `main` and GitHub Pages serves it.

## Security checklist

| Concern | Where it's handled |
| --- | --- |
| Password hashing | Supabase (bcrypt internally) |
| Email verification | Supabase, gated by the `email_confirmed_at` check in RLS |
| Browser-readable secrets | Anon key only — service-role key never leaves Supabase |
| Cross-user reads | RLS policy `auth.uid() = id` on profiles |
| Tampering with deals | Insert/update/delete denied to anon + authenticated roles by default |
| Password reset | Supabase email-link flow, `resetPasswordForEmail` |
| Session storage | Supabase JS client puts JWT in localStorage with auto-refresh |

## Known gaps / things to add for production

- MFA: Supabase supports TOTP; the UI here doesn't enrol users yet.
- Rate limiting on sign-ups: configure in Supabase **Auth Rate Limits**.
- Disable public sign-up if you want invite-only — set "Allow new users to sign up" off and create users from the dashboard.
- The deals seed data is hard-coded. Replace with real rows or wire a CMS.

## Local preview

Because the page imports an ES module from a relative path, opening
`index.html` directly via `file://` won't work. Use any static server:

```sh
npx serve -p 4173 .
# then open http://localhost:4173/
```
