# PoC 03 — Clerk (auth) + Neon (Postgres) + Cloudflare Pages (host)

Three-vendor stack chosen for: (a) the slickest investor-facing sign-up UI of
the free options (Clerk), (b) Postgres data sovereignty (Neon), (c) free,
commercial-friendly hosting with edge functions (Cloudflare Pages).

## Architecture

```
Browser
  ├── Clerk JS (sign-in / sign-up UI, session JWT)
  └── fetch /api/profile, /api/deals  (Authorization: Bearer <session jwt>)
                  │
                  ▼
        Cloudflare Pages Function
          1. verify Clerk JWT via JWKS (jose)
          2. query Neon Postgres (@neondatabase/serverless)
          3. return JSON
```

## Directory layout

```
03-clerk-neon/
├── public/                 ← static front-end, served by Pages
│   ├── index.html
│   ├── app.js
│   └── config.example.js   → copy to config.js (publishable Clerk key)
├── functions/              ← Pages Functions = server-side endpoints
│   ├── _lib/auth.js        ← Clerk JWT verification helper
│   └── api/
│       ├── profile.js      ← GET / PATCH /api/profile
│       └── deals.js        ← GET /api/deals
├── schema.sql              ← Neon DDL + seed deals
├── package.json            ← server-side deps (jose, neon)
├── wrangler.toml           ← Pages project config
└── .dev.vars.example       → copy to .dev.vars for local dev
```

## One-time setup

### 1. Clerk
- Create an application at <https://dashboard.clerk.com>.
- Note the **Publishable key** (`pk_test_…`) — for the browser.
- Note the **Frontend API URL** (something like
  `https://YOUR-INSTANCE.clerk.accounts.dev`) — needed for JWT verification.
- **JWT Templates** → ensure a default session token; it'll include `sub`,
  `email`, and `email_verified` claims by default.

### 2. Neon
- Create a project at <https://console.neon.tech>.
- Copy the **pooled** connection string from the dashboard
  (`postgres://…?sslmode=require`).
- Open the SQL editor and paste `schema.sql`.

### 3. Cloudflare Pages
- Create a new Pages project, "Connect to Git", point at this repo.
- **Build settings**:
  - Framework preset: *None*
  - Build command: `(none)` — or `npm install` if the dashboard insists
  - Build output dir: `portals/03-clerk-neon/public`
  - Root directory: `portals/03-clerk-neon`
- **Environment variables (Production + Preview)**:
  - `CLERK_JWT_ISSUER` = your Clerk frontend API URL
  - `DATABASE_URL` = the Neon pooled connection string
- Save and deploy.

### 4. Wire the publishable key into the browser
- Copy `public/config.example.js` to `public/config.js`.
- Paste your Clerk publishable key.

### 5. Tell Clerk to accept the new origin
- In Clerk, **Domains → Add domain**, e.g. `wavionex-portal-clerk-neon.pages.dev`
  or your custom domain.

## Local dev

```sh
cd portals/03-clerk-neon
cp .dev.vars.example .dev.vars   # fill in real values
cp public/config.example.js public/config.js
npm install
npm run dev
# → http://localhost:8788/
```

## Security checklist

| Concern | Where it's handled |
| --- | --- |
| Password hashing / breach detection / MFA | Clerk |
| Session token | Clerk JWT, short-lived, refreshed by Clerk JS |
| JWT verification on the server | `jose` against Clerk's JWKS in `_lib/auth.js` |
| Forging userId | Impossible — `sub` claim comes from a JWT we verify |
| Cross-user reads | Pages Function filters by `auth.userId` (which is `sub`) |
| Email verification gating | `auth.emailVerified` checked before returning deals |
| DB credentials exposure | Neon URL lives only in CF Pages env vars, never the browser |
| CORS | Same-origin (front-end + functions share the host) |

## Known gaps for production

- Add a Clerk webhook → CF Pages Function to mirror user create / delete
  events into the Neon `profiles` table (right now profiles are created
  lazily on first GET).
- Add rate limiting (Cloudflare Rules → Rate Limiting Rules) on `/api/*`.
- Move deal admin to a separate authenticated route + Clerk role.
- Wire Clerk MFA enforcement at organisation level.
