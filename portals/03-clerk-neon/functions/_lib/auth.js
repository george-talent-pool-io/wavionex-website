/* Shared helpers for the Pages Functions:
   - Pull the Clerk JWT from the Authorization header.
   - Verify it against Clerk's JWKS (using `jose`).
   - Return a small `{ userId, email, emailVerified }` object on success.

   Note: @clerk/backend also offers `verifyToken`; we use the lower-level
   `jose` flow so this PoC has minimal dependencies and runs cleanly on the
   Cloudflare Pages Functions runtime (V8 isolates, no Node built-ins). */

import { createRemoteJWKSet, jwtVerify } from 'jose';

let _jwks = null;
function jwks(env) {
    if (_jwks) return _jwks;
    const issuer = env.CLERK_JWT_ISSUER;
    if (!issuer) throw new Error('CLERK_JWT_ISSUER env var is required');
    _jwks = createRemoteJWKSet(new URL(issuer + '/.well-known/jwks.json'));
    return _jwks;
}

export async function requireUser(request, env) {
    const auth = request.headers.get('Authorization') || '';
    const m = auth.match(/^Bearer\s+(.+)$/i);
    if (!m) return { error: new Response('Missing bearer token', { status: 401 }) };
    const token = m[1];

    try {
        const { payload } = await jwtVerify(token, jwks(env), {
            issuer: env.CLERK_JWT_ISSUER,
            /* Clerk session JWTs carry the user id in `sub`. */
        });
        if (!payload.sub) return { error: new Response('Token missing sub', { status: 401 }) };
        return {
            userId: payload.sub,
            email: payload.email || null,
            emailVerified: payload.email_verified === true || payload.email_verified === 'true',
            payload
        };
    } catch (ex) {
        return { error: new Response('Invalid token: ' + (ex.message || ex), { status: 401 }) };
    }
}

export function json(body, init = {}) {
    return new Response(JSON.stringify(body), {
        ...init,
        headers: {
            'Content-Type': 'application/json',
            'Cache-Control': 'no-store',
            ...(init.headers || {})
        }
    });
}
