import { neon } from '@neondatabase/serverless';

/* Lazy singleton — Neon's `neon()` returns a query function bound to the
   connection string. Safe to reuse across requests. */
let _sql: ReturnType<typeof neon> | null = null;

export function sql() {
    if (_sql) return _sql;
    const url = process.env.DATABASE_URL;
    if (!url) throw new Error('DATABASE_URL is not set');
    _sql = neon(url);
    return _sql;
}

export type ProfileRow = {
    id: string;
    email: string;
    full_name: string | null;
    firm: string | null;
    email_verified_at: string | null;
    created_at: string;
};
