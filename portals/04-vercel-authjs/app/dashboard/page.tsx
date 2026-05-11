import { redirect } from 'next/navigation';
import { auth, signOut } from '@/lib/auth';
import { sql } from '@/lib/db';

type ProfileRow = {
    id: string; email: string; full_name: string | null; firm: string | null;
    email_verified_at: string | null; created_at: string;
};

type DealRow = { id: string; name: string; stage: string; target_close: string | null; headline: string | null };

export default async function DashboardPage() {
    const session = await auth();
    if (!session?.user) redirect('/login');
    const uid = (session.user as any).id as string;

    const profiles = await sql()`
        select id, email, full_name, firm, email_verified_at, created_at
        from users where id = ${uid} limit 1
    ` as ProfileRow[];
    const profile = profiles[0];
    if (!profile) redirect('/login');

    const verified = !!profile.email_verified_at;
    const deals = verified
        ? (await sql()`select id, name, stage, target_close, headline from deals order by target_close asc nulls last` as DealRow[])
        : [];

    async function doSignOut() {
        'use server';
        await signOut({ redirectTo: '/login' });
    }

    return (
        <section className="portal-card">
            <div className="portal-row portal-row--between" style={{ marginBottom: '1rem' }}>
                <h1 style={{ margin: 0 }}>Welcome, {profile.full_name || profile.email.split('@')[0]}.</h1>
                <form action={doSignOut}>
                    <button className="portal-btn portal-btn--ghost" type="submit">Sign out</button>
                </form>
            </div>
            <p className="portal-muted">
                {verified
                    ? 'You are signed in. Session is an HttpOnly JWT cookie; deal data is fetched server-side from Neon.'
                    : 'You are signed in, but your email is not yet verified — deal data is hidden until you click the verification link.'}
            </p>

            <hr className="portal-divider" />

            <h2>Profile</h2>
            <dl className="portal-kv">
                <dt>Email</dt><dd>{profile.email}</dd>
                <dt>Firm</dt><dd>{profile.firm || '—'}</dd>
                <dt>Email verified</dt><dd>{verified ? 'yes' : 'no'}</dd>
                <dt>Member since</dt><dd>{new Date(profile.created_at).toLocaleDateString()}</dd>
                <dt>User id</dt><dd style={{ fontFamily: 'ui-monospace, monospace', fontSize: '0.78rem' }}>{profile.id}</dd>
            </dl>

            <hr className="portal-divider" />

            <h2>Open deals</h2>
            {!verified ? (
                <p className="portal-muted">Verify your email to unlock the deal pipeline.</p>
            ) : deals.length === 0 ? (
                <p className="portal-muted">No deals seeded — run schema.sql or insert rows manually.</p>
            ) : (
                <div className="portal-grid-deals">
                    {deals.map((d) => (
                        <article key={d.id} className="portal-deal">
                            <span className="meta">{d.stage}</span>
                            <h4>{d.name}</h4>
                            <p className="portal-muted" style={{ margin: '0.25rem 0 0 0' }}>{d.headline}</p>
                            <p className="portal-muted" style={{ margin: '0.5rem 0 0 0', fontSize: '0.78rem' }}>
                                Target close {d.target_close ? new Date(d.target_close).toLocaleDateString() : '—'}
                            </p>
                        </article>
                    ))}
                </div>
            )}
        </section>
    );
}
