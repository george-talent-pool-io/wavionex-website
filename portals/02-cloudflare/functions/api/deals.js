import { requireAccessUser, json } from '../_lib/access.js';

export async function onRequest({ request, env }) {
    const auth = await requireAccessUser(request, env);
    if (auth.error) return auth.error;
    if (!env.DB) return new Response('D1 binding `DB` missing', { status: 500 });

    const { results } = await env.DB.prepare(
        'select id, name, stage, target_close, headline from deals order by target_close asc'
    ).all();
    return json(results || []);
}
