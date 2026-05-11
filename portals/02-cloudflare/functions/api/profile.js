import { requireAccessUser, json } from '../_lib/access.js';

export async function onRequest({ request, env }) {
    const auth = await requireAccessUser(request, env);
    if (auth.error) return auth.error;
    if (!env.DB) return new Response('D1 binding `DB` missing', { status: 500 });

    if (request.method === 'GET') {
        const row = await env.DB.prepare(
            'select email, full_name, firm, identity_provider, last_login_at, created_at from profiles where email = ?1'
        ).bind(auth.email).first();
        if (row) {
            await env.DB.prepare('update profiles set last_login_at = ?1, identity_provider = ?2 where email = ?3')
                .bind(new Date().toISOString(), auth.identityProvider, auth.email).run();
            row.last_login_at = new Date().toISOString();
            return json(row);
        }
        const now = new Date().toISOString();
        await env.DB.prepare(
            'insert into profiles (email, identity_provider, last_login_at, created_at) values (?1, ?2, ?3, ?3)'
        ).bind(auth.email, auth.identityProvider, now).run();
        return json({ email: auth.email, full_name: null, firm: null, identity_provider: auth.identityProvider, last_login_at: now, created_at: now });
    }

    if (request.method === 'PATCH') {
        let body = {};
        try { body = await request.json(); } catch (_) {}
        const firm = typeof body.firm === 'string' ? body.firm.slice(0, 200) : null;
        const fullName = typeof body.full_name === 'string' ? body.full_name.slice(0, 200) : null;
        const now = new Date().toISOString();
        await env.DB.prepare(`
            insert into profiles (email, firm, full_name, identity_provider, last_login_at, created_at)
            values (?1, ?2, ?3, ?4, ?5, ?5)
            on conflict(email) do update set
                firm           = coalesce(excluded.firm,      profiles.firm),
                full_name      = coalesce(excluded.full_name, profiles.full_name),
                last_login_at  = excluded.last_login_at
        `).bind(auth.email, firm, fullName, auth.identityProvider, now).run();
        const row = await env.DB.prepare(
            'select email, full_name, firm, identity_provider, last_login_at, created_at from profiles where email = ?1'
        ).bind(auth.email).first();
        return json(row);
    }

    return new Response('Method not allowed', { status: 405, headers: { Allow: 'GET, PATCH' } });
}
