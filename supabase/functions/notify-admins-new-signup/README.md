# `notify-admins-new-signup` — Supabase Edge Function

Fires whenever a new `profiles` row appears, looks up every admin
(`is_admin = true`), and emails each of them a "new investor awaiting
approval" message via [Resend](https://resend.com).

## One-time setup

### 1. Resend account + API key

- Create a free Resend account at <https://resend.com> (100 emails/day on
  the free tier — plenty for new-signup alerts).
- Add and verify the domain you want to send from (`wavionex.com` or
  similar). Resend walks you through the DNS records.
- Create an API key (Resend → API Keys → "Create API Key"). Copy it —
  starts with `re_`.

### 2. Install the Supabase CLI and link the project

```sh
brew install supabase/tap/supabase    # macOS; see docs.supabase.com for others
supabase login                        # opens the browser
cd /Users/georgebilchev/wavionex/wavionex-website
supabase link --project-ref aitzjjxsizvzglypabwu
```

### 3. Set the function's secrets

These never leave Supabase — used inside the Edge Function only.

```sh
supabase secrets set RESEND_API_KEY=re_xxxxxxxxxxxxxxxxxxxxxxxxx
supabase secrets set EMAIL_FROM='Wavionex <portal@wavionex.com>'
supabase secrets set ADMIN_PANEL_URL='https://www.wavionex.com/portals/investor/admin/'
```

`SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` are auto-injected.

### 4. Deploy

```sh
supabase functions deploy notify-admins-new-signup --no-verify-jwt
```

`--no-verify-jwt` lets the Database Webhook call the function without
an auth token (the webhook itself is invoked from inside Supabase and
authenticated via the project's internal mechanisms).

### 5. Create the Database Webhook

Dashboard path:
<https://supabase.com/dashboard/project/aitzjjxsizvzglypabwu/integrations/webhooks/overview>

- **Name:** `notify-admins-on-signup`
- **Table:** `public.profiles`
- **Events:** ☑ Insert (leave Update / Delete unchecked)
- **Type:** Supabase Edge Functions
- **Edge Function:** `notify-admins-new-signup`
- **HTTP method:** POST
- **HTTP headers:** leave default

Save. Supabase will trigger this webhook every time a new row appears
in `profiles`.

## Test

Create a fresh test user via the portal sign-up form, using a real invite
code you have. As soon as the row lands in `profiles`, every admin email
in `profiles` (where `is_admin = true`) should get a notification within
a few seconds.

Logs and retries are visible at:
<https://supabase.com/dashboard/project/aitzjjxsizvzglypabwu/functions/notify-admins-new-signup/logs>

## What the email looks like

> **Subject:** New Wavionex investor awaiting approval
>
> A new investor has signed up and is awaiting admin approval.
>
> | | |
> |---|---|
> | Name  | Jane Doe |
> | Email | jane@firm.com |
> | Firm  | Acme Capital |
> | Joined | 2026-05-12T18:24:01Z |
>
> [Open admin panel] ← button linking to ADMIN_PANEL_URL
>
> — Wavionex Investor Portal

Each admin gets a personal copy (we don't BCC).

## Skip conditions

The function returns 200 + a "skipped" reason in these cases:
- The new profile already has `is_approved = true` (admin-created user).
- The new profile already has `is_admin = true` (don't notify on admin creation).
- No admins exist in `profiles` (nothing to do).

## Troubleshooting

- **No emails arrive** → check Resend → Logs for delivery state. Verify
  your sending domain's DNS records are green.
- **"RESEND_API_KEY not set"** in the function logs → re-run step 3.
- **Database Webhook delivers but function 500s** → look at function logs
  for the error; most often the service-role key wasn't auto-injected
  because the CLI deploy didn't link the project.
