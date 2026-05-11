/* Copy to `public/config.js`. Only the *publishable* Clerk key goes here —
   it is safe to expose to browsers. The secret key + Neon connection string
   are server-only (set as Cloudflare Pages environment variables). */

export const CLERK_PUBLISHABLE_KEY = 'pk_test_XXXXXXXXXXXXXXXX';

/* Optional: pin the Clerk JS version. Leave undefined to use the major channel. */
export const CLERK_JS_VERSION = '5';
