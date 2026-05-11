# PoC 05 — Firebase (Auth + Firestore)

Front-end is plain HTML/JS served from GitHub Pages, talks directly to Firebase
Auth and Firestore. Data security lives in `firestore.rules`.

## Architecture

```
Browser ─►  Firebase Auth      (signup, login, email verification, reset)
       ─►  Firestore           (profiles/{uid} + deals/{id}, gated by rules)

Hosting:  GitHub Pages → portals/05-firebase/
Auth+DB:  Firebase project (Spark plan / free)
```

## What you need to do once

1. Create a project at <https://console.firebase.google.com>.
2. **Build → Authentication → Get started → Email/Password → Enable**.
3. **Build → Firestore Database → Create database** (production mode, location of your choice).
4. **Project Settings (cog) → General → Your apps → Web (`</>`) → Register app.**
   Copy the `firebaseConfig` object.
5. Copy `config.example.js` to `config.js` and paste the values in.
6. Install Firebase CLI and deploy the rules:
   ```sh
   npm i -g firebase-tools
   firebase login
   firebase use --add <your-project>
   firebase deploy --only firestore:rules
   ```
   (or paste `firestore.rules` into the **Firestore → Rules** tab.)
7. **Authentication → Settings → Authorized domains**: add the domain you'll
   host the portal on (e.g. `wavionex.com` or `<user>.github.io`).
8. Seed deal documents via the Firestore Console UI, or follow the instructions
   in `seed-deals.js`.

## Security checklist

| Concern | Where it's handled |
| --- | --- |
| Password hashing | Firebase (scrypt internally) |
| Email verification | Firebase, enforced in `firestore.rules` via `request.auth.token.email_verified` |
| Browser secrets | Web API key only — it's a public identifier, not a secret |
| Cross-user reads of profiles | Rule: `request.auth.uid == uid` |
| Tampering with deals | `allow write: if false` (admin only) |
| Password reset | `sendPasswordResetEmail` flow with verified domain |
| Session storage | Firebase Auth manages tokens; refreshes automatically |

## Known gaps / things to add for production

- MFA / phone second factor: available on the Identity Platform upgrade (still free at low scale, but a paid tier).
- App Check (anti-abuse, attests requests come from your real domain): enable in console under "Build → App Check".
- Cloud Functions for any admin endpoints (creating deals, RBAC, audit).
- Firestore composite indexes if you add filtered queries.

## Local preview

```sh
npx serve -p 4173 .
# then open http://localhost:4173/
```

## Caveat: data portability

Firestore is NoSQL and Google-only. Migrating off Firebase later means
rewriting queries against a different DB. If portability of investor data is a
priority, prefer the Supabase, Clerk+Neon, or Vercel+Auth.js PoCs which all
store data in Postgres.
