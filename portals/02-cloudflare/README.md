# PoC 02 — Cloudflare Access + Pages + D1

Edge auth: Cloudflare Access gates the entire portal subdomain before any
HTML is served. The app itself never sees a login form — by the time the page
is loaded, the visitor has already proven they own a permitted email. A Pages
Function reads the signed Access JWT to identify them and persists profile +
serves deal data from D1 (SQLite at the edge).

## Architecture

```
   Visitor ──► Cloudflare Edge ──► Cloudflare Access ──► (if allowed) ──► Pages app
                                  (email-OTP / SSO / MFA / IdP federation)
                                              │
                                              ▼
                                       inject headers:
                                         Cf-Access-Authenticated-User-Email
                                         Cf-Access-Jwt-Assertion
                                              │
                                              ▼
                                     Pages Function /api/*
                                       1. read & (optionally) verify JWT
                                       2. read/write D1
                                       3. return JSON
```

## Directory layout

```
02-cloudflare/
├── public/
│   ├── index.html
│   └── app.js
├── functions/
│   ├── _lib/access.js          ← Cf-Access header trust + JWT verify
│   └── api/
│       ├── profile.js
│       └── deals.js
├── schema.sql                  ← D1 schema + seed deals
├── package.json
├── wrangler.toml               ← D1 binding (`DB`)
└── README.md
```

## One-time setup

### 1. Cloudflare account
Sign up if you don't have one. **Zero Trust** must be enabled
(<https://one.dash.cloudflare.com> → choose a free team domain like
`wavionex.cloudflareaccess.com`).

### 2. Create the D1 database
```sh
cd portals/02-cloudflare
npm install
wrangler d1 create wavionex-portal-d1
# → paste the printed `database_id` into wrangler.toml
npm run db:init
```

### 3. Deploy the Pages project
```sh
npm run deploy   # creates the Pages project on first run
```
You'll get a `*.pages.dev` URL.

### 4. Bind a Cloudflare Access policy to the Pages project
- **Zero Trust → Access → Applications → Add → Self-hosted**.
- Application domain: the `*.pages.dev` URL (or your custom domain).
- Policy: allow your investor list. Examples:
  - "Emails ending in `@wavionex.com`" — for the team.
  - "Emails listed" — for explicit investor allow-list.
  - "Identity provider: Google / Microsoft / GitHub" — federate to their IdP.
- (Optional) Require MFA — set under the policy's "Require" rules.

### 5. Strict JWT verification (recommended)
In the Pages project → **Settings → Environment variables** add:
- `CF_ACCESS_TEAM_DOMAIN` = `https://<team>.cloudflareaccess.com`
- `CF_ACCESS_AUD` = the AUD tag for the Access application
  (shown in the Access app's overview tab)

With these set, `_lib/access.js` verifies the signed JWT instead of trusting
the email header. Belt-and-braces.

## Security checklist

| Concern | Where it's handled |
| --- | --- |
| Authentication | Cloudflare Access (email OTP / SSO / SAML / GitHub / Google etc.) |
| MFA | Access policy "Require" rule |
| Session management | Access-issued cookie, scoped to the application |
| Sign-out | `/cdn-cgi/access/logout` (linked from the page footer) |
| Header forgery | Avoided: Pages app is only reachable via the Access proxy; optional JWT verify provides defence-in-depth |
| Cross-user reads | Profile keyed by Access email; Pages Function filters by that |
| DB credentials | None to leak — D1 is a binding, not a connection string |
| Audit | Zero Trust → Logs → Access shows every successful + denied attempt |

## Known gaps

- The 50-user free tier on Access is hard. For >50 investors, the next tier
  is Cloudflare Zero Trust Standard (~$3/user/mo at time of writing).
- D1 is SQLite — fine for profile + deal metadata, less suited for heavy
  analytics. If you need joins across millions of rows, swap for Neon
  (mirror the pattern from PoC 03).
- No self-serve sign-up: investors must be added to the Access policy by an
  admin. That's typically the desired behaviour for investor portals.
