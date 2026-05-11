import { sql } from '@/lib/db';
import { redirect } from 'next/navigation';

export default async function VerifyPage({ searchParams }: { searchParams: { token?: string } }) {
    const token = searchParams.token;
    if (!token || token.length < 16) {
        return (
            <section className="portal-card">
                <h1>Verification link missing or malformed</h1>
                <p className="portal-muted">Open the link in the email you received. If it expired, sign in and request a new one.</p>
                <a href="/login" className="portal-btn">Back to sign in</a>
            </section>
        );
    }

    const rows = await sql()`
        select user_id, expires_at, used_at
        from email_verifications
        where token = ${token}
        limit 1
    ` as Array<{ user_id: string; expires_at: string; used_at: string | null }>;

    if (rows.length === 0) {
        return (
            <section className="portal-card">
                <h1>Token not recognised</h1>
                <p className="portal-muted">Either the link was already used, or it has been invalidated. Sign in and request a new verification email.</p>
                <a href="/login" className="portal-btn">Sign in</a>
            </section>
        );
    }
    const row = rows[0];
    if (row.used_at) {
        return (
            <section className="portal-card">
                <h1>Already verified</h1>
                <p className="portal-muted">This link was used before. You can sign in normally.</p>
                <a href="/login?verified=1" className="portal-btn">Sign in</a>
            </section>
        );
    }
    if (new Date(row.expires_at).getTime() < Date.now()) {
        return (
            <section className="portal-card">
                <h1>Link expired</h1>
                <p className="portal-muted">Verification links are valid for 24 hours. Sign in and request a fresh one.</p>
                <a href="/login" className="portal-btn">Sign in</a>
            </section>
        );
    }

    await sql()`
        update users set email_verified_at = now() where id = ${row.user_id} and email_verified_at is null
    `;
    await sql()`
        update email_verifications set used_at = now() where token = ${token}
    `;
    redirect('/login?verified=1');
}
