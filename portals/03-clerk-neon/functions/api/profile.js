import { neon } from '@neondatabase/serverless';
import { requireUser, json } from '../_lib/auth.js';

export async function onRequest({ request, env }) {
    const auth = await requireUser(request, env);
    if (auth.error) return auth.error;
    if (!env.DATABASE_URL) return new Response('DATABASE_URL env var missing', { status: 500 });

    const sql = neon(env.DATABASE_URL);

    if (request.method === 'GET') {
        const rows = await sql`
            select clerk_user_id, email, full_name, firm, created_at, updated_at
            from profiles
            where clerk_user_id = ${auth.userId}
            limit 1
        `;
        if (rows.length === 0) {
            const inserted = await sql`
                insert into profiles (clerk_user_id, email)
                values (${auth.userId}, ${auth.email})
                on conflict (clerk_user_id) do update set email = excluded.email, updated_at = now()
                returning clerk_user_id, email, full_name, firm, created_at, updated_at
            `;
            return json(inserted[0]);
        }
        return json(rows[0]);
    }

    if (request.method === 'PATCH') {
        let body = {};
        try { body = await request.json(); } catch (_) {}
        const firm = typeof body.firm === 'string' ? body.firm.slice(0, 200) : null;
        const fullName = typeof body.full_name === 'string' ? body.full_name.slice(0, 200) : null;
        const rows = await sql`
            insert into profiles (clerk_user_id, email, firm, full_name)
            values (${auth.userId}, ${auth.email}, ${firm}, ${fullName})
            on conflict (clerk_user_id) do update
                set firm      = coalesce(excluded.firm,      profiles.firm),
                    full_name = coalesce(excluded.full_name, profiles.full_name),
                    email     = excluded.email,
                    updated_at = now()
            returning clerk_user_id, email, full_name, firm, created_at, updated_at
        `;
        return json(rows[0]);
    }

    return new Response('Method not allowed', { status: 405, headers: { Allow: 'GET, PATCH' } });
}
