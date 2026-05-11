import { neon } from '@neondatabase/serverless';
import { requireUser, json } from '../_lib/auth.js';

export async function onRequest({ request, env }) {
    const auth = await requireUser(request, env);
    if (auth.error) return auth.error;
    if (!env.DATABASE_URL) return new Response('DATABASE_URL env var missing', { status: 500 });

    /* Investor portal rule: only verified emails see real deal flow. */
    if (!auth.emailVerified) {
        return json([]);
    }

    const sql = neon(env.DATABASE_URL);
    const rows = await sql`
        select id, name, stage, target_close, headline
        from deals
        order by target_close asc nulls last
    `;
    return json(rows);
}
