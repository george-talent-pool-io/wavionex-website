/* Cloudflare Access identity helper.

   When a request reaches a Pages app gated by Access, Cloudflare adds two
   headers we can trust at the edge boundary:

     Cf-Access-Authenticated-User-Email
     Cf-Access-Jwt-Assertion    (signed JWT with full identity)

   For an extra-paranoid PoC we verify the JWT against Access's JWKS so
   the function can't be tricked by header injection if the app is ever
   moved to run somewhere outside the Access boundary. */

import { createRemoteJWKSet, jwtVerify } from 'jose';

let _jwks = null;
function jwks(env) {
    if (_jwks) return _jwks;
    if (!env.CF_ACCESS_TEAM_DOMAIN) throw new Error('CF_ACCESS_TEAM_DOMAIN env var required');
    _jwks = createRemoteJWKSet(new URL(`${env.CF_ACCESS_TEAM_DOMAIN.replace(/\/$/, '')}/cdn-cgi/access/certs`));
    return _jwks;
}

export async function requireAccessUser(request, env) {
    const email = request.headers.get('Cf-Access-Authenticated-User-Email');
    const jwt = request.headers.get('Cf-Access-Jwt-Assertion');
    if (!email || !jwt) {
        return { error: new Response('Cloudflare Access headers missing — is the route gated?', { status: 401 }) };
    }

    /* If CF_ACCESS_AUD is configured, verify the JWT properly. Otherwise
       trust the headers — fine when we know the function is only reachable
       through the Access proxy (which is true for Pages projects with an
       Access policy attached to the entire app). */
    if (env.CF_ACCESS_AUD && env.CF_ACCESS_TEAM_DOMAIN) {
        try {
            const { payload } = await jwtVerify(jwt, jwks(env), {
                issuer: env.CF_ACCESS_TEAM_DOMAIN,
                audience: env.CF_ACCESS_AUD
            });
            return {
                email: payload.email || email,
                identityProvider: payload.identity_nonce ? 'oidc' : (payload.amr?.[0] || 'access'),
                payload
            };
        } catch (ex) {
            return { error: new Response('Bad Access JWT: ' + ex.message, { status: 401 }) };
        }
    }

    return { email, identityProvider: 'access' };
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
