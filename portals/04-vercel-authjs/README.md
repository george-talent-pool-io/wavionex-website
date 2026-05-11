# PoC 04 — Vercel + Auth.js + Neon (Postgres)

Self-hosted credentials auth. We own every part of the flow: password hashing
(argon2id), session cookies (HttpOnly JWT), email verification (CSPRNG token in
Postgres, single-use, 24h expiry), and the user table itself. The most code
of the five PoCs, but also the most portable — it's plain Next.js + standard
Node libraries; nothing here is locked to Vercel.

## Architecture

```
Browser  ──►  Next.js (App Router) on Vercel
                  ├── /signup           POST /api/signup       (bcrypt → argon2id, send verify email)
                  ├── /verify?token=…   server component        (single-use token check, mark verified)
                  ├── /login            Auth.js Credentials provider
                  ├── /dashboard        server component        (reads users + deals from Neon)
                  └── /api/auth/[…]     Auth.js handlers        (CSRF, sessions, signOut)
                          │
                          ▼
                Neon Postgres (@neondatabase/serverless over HTTP)
```

## Directory layout

```
04-vercel-authjs/
├── app/
│   ├── layout.tsx
│   ├── page.tsx                 ← redirects to /dashboard or /login
│   ├── login/page.tsx
│   ├── signup/page.tsx
│   ├── verify/page.tsx
│   ├── dashboard/page.tsx
│   ├── api/
│   │   ├── auth/[...nextauth]/route.ts  ← Auth.js handlers
│   │   └── signup/route.ts              ← create user + send verify email
│   └── globals.css
├── lib/
│   ├── auth.ts                  ← NextAuth() config (credentials, jwt sessions)
│   ├── db.ts                    ← Neon serverless client
│   ├── passwords.ts             ← argon2id wrappers
│   └── email.ts                 ← Resend (or stdout fallback)
├── middleware.ts                ← protects /dashboard
├── schema.sql                   ← users, email_verifications, deals
├── package.json
├── next.config.mjs
├── tsconfig.json
└── .env.example
```

## One-time setup

### 1. Neon
- Create a project at <https://console.neon.tech>.
- Copy the **pooled** connection string (ends in `?sslmode=require`).
- Run the schema: `psql "$DATABASE_URL" -f schema.sql` (or paste it into the Neon SQL editor).

### 2. Resend (optional but recommended)
- Create a project at <https://resend.com>. Add an API key.
- Add and verify a sending domain (e.g. `portal.wavionex.com`).
- Without Resend, verification emails are logged to stdout — fine for local dev, useless in production.

### 3. Vercel
- `vercel link` (or import the GitHub repo in the Vercel dashboard).
- Set the project root to `portals/04-vercel-authjs/`.
- Environment variables (Production + Preview + Development):
  - `DATABASE_URL`
  - `AUTH_SECRET`     (generate with `openssl rand -base64 48`)
  - `NEXTAUTH_URL`    (your deployed URL, no trailing slash)
  - `RESEND_API_KEY`  (optional)
  - `EMAIL_FROM`
- `vercel --prod` to deploy.

## Local dev

```sh
cd portals/04-vercel-authjs
cp .env.example .env.local        # fill in the values
npm install
npm run db:init                   # one-time, applies schema.sql
npm run dev                       # → http://localhost:3004
```

## Security checklist

| Concern | Where it's handled |
| --- | --- |
| Password hashing | `lib/passwords.ts` → argon2id, OWASP params (19MiB / t=2 / p=1) |
| Password length / format | `zod` schema rejects <10 chars at sign-up |
| Email verification | CSPRNG 32-byte token in `email_verifications`, single-use, 24h expiry, marked `used_at` on consumption |
| Email enumeration | `/api/signup` returns the same response whether the email is new or already registered |
| Sessions | Auth.js v5 JWT strategy, HttpOnly + Secure + SameSite=lax cookies (Vercel sets Secure on https) |
| CSRF | Auth.js handlers verify the CSRF cookie/token pair on POSTs |
| Cross-user reads | `dashboard/page.tsx` filters by `session.user.id`; deals only shown if email is verified |
| DB credentials | Only on the server (`DATABASE_URL` env var); never reaches the browser |
| Route protection | `middleware.ts` redirects unauthenticated requests to `/login` |

## Known gaps for production

- **MFA**: not yet implemented. Add a TOTP column + `otplib` enrolment flow before opening to investors.
- **Rate limiting**: Vercel has WAF rules; add one for `/api/signup` and `/api/auth/callback/credentials` to slow credential-stuffing.
- **Password reset**: stub it the same way as `/verify` — separate `password_resets` table, similar single-use token.
- **Audit log**: insert a row into a `login_events` table from the Credentials `authorize` callback on each success / failure.
- **Vercel Hobby ToS**: Vercel's free tier technically prohibits commercial use. For a private investor portal this sits in a grey area — for production, upgrade to Pro or migrate hosting to Cloudflare Pages or a self-hosted Node server.
