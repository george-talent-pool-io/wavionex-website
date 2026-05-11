# Wavionex Investor Portal — Supabase

Live at <https://www.wavionex.com/portals/investor/>.

Single-vendor stack: front-end is plain HTML/JS served from GitHub Pages,
talks directly to Supabase. Auth + email verification + password reset are
handled by Supabase; data security is enforced by Row-Level Security policies
in Postgres. Chosen out of five PoCs (see `../README.md` for the comparison).

## Architecture

```
Browser  ─►  Supabase Auth   (sign-up, login, email confirm, password reset)
        ─►  Supabase REST   (profiles + deals tables, gated by RLS)
        ─►  Supabase Storage (PDFs / papers, gated by bucket policies)  [todo]

Hosting:   GitHub Pages  →  portals/investor/  (this directory)
Auth/DB:   Supabase project (free tier)
```

## Supabase dashboard configuration

| Setting | Value |
| --- | --- |
| Project URL | `https://aitzjjxsizvzglypabwu.supabase.co` |
| Site URL | `https://www.wavionex.com/portals/investor/` |
| Redirect URLs | `https://www.wavionex.com/portals/investor/` + `http://localhost:4173/` |
| Email confirmation | ON |
| Public sign-up | OFF (invite-only) — see "Invite flow" below |

## Files

- `index.html` — sign-in / sign-up / dashboard SPA
- `app.js` — Supabase client wiring
- `config.js` — project URL + publishable key (safe to commit; data security comes from RLS)
- `config.example.js` — template
- `schema.sql` — Postgres tables, RLS, seed deals

## Invite flow (production)

The current schema allows anyone to sign up. To switch to invite-only:

1. **Disable public sign-up.** Authentication → Providers → Email → toggle "Allow new users to sign up" OFF. New investors are created from the dashboard or via the Admin API.
2. **OR** keep public sign-up on but gate it with an `invite_codes` table:
   - Schema addition not yet applied — ask the operator to add it when you turn this on.
   - Sign-up form gains an "invite code" field validated against `invite_codes(code, max_uses, expires_at, used_at)`.

## Local preview

```sh
cd portals/investor
npx serve -p 4173 .
# → http://localhost:4173/
```

## Security checklist

| Concern | Where it's handled |
| --- | --- |
| Password hashing | Supabase (bcrypt) |
| Email verification | Supabase; deals RLS requires `email_confirmed_at` to be set |
| Browser-readable secrets | Publishable key only — secret key never leaves Supabase |
| Cross-user reads | `auth.uid() = id` on profiles |
| Tampering with deals | No INSERT/UPDATE/DELETE policy → denied to all browser clients |
| Password reset | Supabase email-link flow (`resetPasswordForEmail`) |
| Sessions | Supabase JS client; refresh tokens auto-rotated |

## Known gaps for production

- **MFA** — Supabase supports TOTP; not yet enrolled in the UI.
- **CAPTCHA on auth endpoints** — Authentication → Attack Protection → enable hCaptcha or Turnstile.
- **Rate limits** — Authentication → Rate Limits; tighten sign-up + sign-in caps.
- **Approved-investor gate** — currently every email-verified user sees deals. For real deal flow, add an `is_approved` column on `profiles` and require it in the deals RLS.
- **Document/PDF gating** — wire Supabase Storage with a `papers` bucket and policy `auth.uid() in (select id from profiles where is_approved)`; serve via signed URLs.
