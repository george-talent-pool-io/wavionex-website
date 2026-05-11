# Wavionex Investor Portal — Supabase

Live at <https://www.wavionex.com/portals/investor/>.

Single-vendor stack: front-end is plain HTML/JS served from GitHub Pages,
talks directly to Supabase. Auth + email verification + password reset are
handled by Supabase; data security is enforced by RLS policies + security-
definer functions in Postgres; file delivery uses Supabase Storage signed
URLs gated by bucket policies.

## What's in this folder

```
investor/
├── index.html       — investor sign-in / sign-up / dashboard
├── app.js           — Supabase client wiring + audit events
├── config.js        — project URL + publishable key (public-by-design)
├── config.example.js
├── schema.sql       — full DDL: profiles, invite_codes, audit_events, papers, RLS, triggers
└── admin/
    ├── index.html   — admin UI (Users / Invites / Audit / Papers / External)
    └── admin.js
```

## Apply / re-apply the schema

`schema.sql` is idempotent. Paste it into the Supabase SQL editor and run
it any time you change it.

<https://supabase.com/dashboard/project/aitzjjxsizvzglypabwu/sql/new>

It will add new columns to `profiles`, create `invite_codes`,
`invite_redemptions`, `audit_events`, `papers`, install the signup trigger,
and replace the RLS policies. Existing rows are kept.

## Create the Storage bucket (one-off)

The papers UI needs a private bucket called `papers`:

<https://supabase.com/dashboard/project/aitzjjxsizvzglypabwu/storage/buckets>

Click **New bucket** → name `papers` → **Public bucket: OFF** → Create.

Then re-run `schema.sql` so the storage object policies attach (the file's
sections 10/11 reference `storage.objects` which only exists after at least
one bucket has been created).

## Bootstrap yourself as admin

### If you already have an account (existing user)

Skip the invite-code dance entirely. Just promote your existing profile row:

```sql
update public.profiles
   set is_admin    = true,
       is_approved = true
 where email = 'YOUR-EMAIL@example.com';
```

Refresh the portal. The "Admin" button appears on the dashboard. From the
**Invites** tab, create real invite codes for new investors.

### Cold-start (nobody has signed up yet)

You can't sign up without an invite, and you can't reach the admin without
signing up. Bootstrap by inserting one invite directly:

```sql
insert into public.invite_codes (code, note, max_uses)
values ('WAVE-2026-BOOT', 'admin bootstrap', 1);
```

Sign up with `WAVE-2026-BOOT`, verify your email, then run the
`update public.profiles set is_admin = true …` query above.

## Supabase dashboard configuration

| Setting | Value |
| --- | --- |
| Project URL | `https://aitzjjxsizvzglypabwu.supabase.co` |
| Site URL | `https://www.wavionex.com/portals/investor/` |
| Redirect URLs | `https://www.wavionex.com/portals/investor/`, `https://www.wavionex.com/portals/investor/admin/`, `http://localhost:4173/` |
| Email confirmation | ON |
| Public sign-up | ON (gated by the invite-code trigger; turning it OFF would block the trigger too) |

URL config page:
<https://supabase.com/dashboard/project/aitzjjxsizvzglypabwu/auth/url-configuration>

## CAPTCHA + rate-limit hardening (no code)

1. **hCaptcha or Cloudflare Turnstile** — Authentication → Attack Protection
   → enable. Add the site key to Supabase; the auth endpoints start requiring
   a token automatically.

   <https://supabase.com/dashboard/project/aitzjjxsizvzglypabwu/auth/protection>

2. **Rate limits** — Authentication → Rate Limits. Tighten:
   - Sign-ups per IP per hour → 5
   - Token refreshes per user per 5 min → 30
   - Recovery emails per email per hour → 2

   <https://supabase.com/dashboard/project/aitzjjxsizvzglypabwu/auth/rate-limits>

3. **Password strength** — Authentication → Providers → Email → require
   12+ chars, no leaked-password matches.

## Security model

| Concern | Where it's handled |
| --- | --- |
| Password hashing | Supabase (bcrypt) |
| Invite-only sign-up | `handle_new_user` trigger on `auth.users` — atomic SELECT … FOR UPDATE on `invite_codes` |
| Email verification | Supabase; `email_verified_at` mirrored into `profiles` by `handle_user_update` trigger |
| Approved-investor gate | `current_user_is_privileged()` referenced by `deals` and `papers` RLS, plus storage bucket policy |
| Admin role | `profiles.is_admin`; flipping it requires already being admin (enforced by `guard_profile_privileged_fields` trigger) |
| Cross-user reads of profiles | self-read only, except admin |
| Tampering with invite_codes | admin-only via RLS; trigger uses SECURITY DEFINER so signup writes don't need a policy |
| Audit log read access | admin only |
| PDF download | signed URL valid for 60 s, issued only if `current_user_is_privileged()` |
| Paper upload | admin only via RLS on `papers` table + storage bucket policy |

## Adding more event types to the audit log

`audit_events` is just a table — insert from any client-side action you
care about. RLS allows authenticated users to insert their own rows. The
admin UI's "Event type" filter dropdown can be extended in
`admin/index.html`.

## Known gaps for production

- **MFA** — Supabase supports TOTP but the UI here doesn't enrol users.
  Add via Supabase Auth UI components when ready.
- **Per-paper RBAC** — every approved investor sees every paper. If you
  need per-investor visibility, add a `paper_access(paper_id, user_id)`
  table and reference it in the papers RLS.
- **Audit log retention** — `audit_events` grows forever; add a cron to
  prune older than, say, 90 days.
- **Admin sign-in audit** — failed-sign-in attempts are in Supabase Auth
  logs (linked from Admin → External tools), not in `audit_events`.

## Local preview

```sh
cd portals/investor
npx serve -p 4173 .
# → http://localhost:4173/         (investor portal)
# → http://localhost:4173/admin/   (admin portal)
```
