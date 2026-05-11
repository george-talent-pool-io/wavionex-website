'use client';
import { useState } from 'react';

export default function SignupPage() {
    const [pending, setPending] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [okMsg, setOk] = useState<string | null>(null);

    async function onSubmit(e: React.FormEvent<HTMLFormElement>) {
        e.preventDefault();
        setError(null);
        setOk(null);
        setPending(true);
        const fd = new FormData(e.currentTarget);
        const body = {
            email: String(fd.get('email') || '').toLowerCase().trim(),
            password: String(fd.get('password') || ''),
            full_name: String(fd.get('full_name') || '').trim(),
            firm: String(fd.get('firm') || '').trim()
        };
        try {
            const res = await fetch('/api/signup', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            if (!res.ok) {
                const txt = await res.text();
                throw new Error(txt || `HTTP ${res.status}`);
            }
            setOk('Account created. Check your inbox for a verification email — you must click the link before signing in.');
            (e.target as HTMLFormElement).reset();
        } catch (ex: any) {
            setError(ex.message || String(ex));
        } finally {
            setPending(false);
        }
    }

    return (
        <section className="portal-card">
            <h1>Request access</h1>
            <p className="portal-muted">Create your investor account. We&rsquo;ll email you a verification link before you can sign in.</p>

            <form onSubmit={onSubmit} className="portal-form" autoComplete="on">
                <div className="portal-field">
                    <label htmlFor="full_name">Full name</label>
                    <input id="full_name" name="full_name" type="text" required autoComplete="name" />
                </div>
                <div className="portal-field">
                    <label htmlFor="firm">Firm / fund</label>
                    <input id="firm" name="firm" type="text" autoComplete="organization" />
                </div>
                <div className="portal-field">
                    <label htmlFor="email">Email</label>
                    <input id="email" name="email" type="email" required autoComplete="email" />
                </div>
                <div className="portal-field">
                    <label htmlFor="password">Password (10+ chars)</label>
                    <input id="password" name="password" type="password" required minLength={10} autoComplete="new-password" />
                </div>
                {error && <div className="portal-alert portal-alert--error">{error}</div>}
                {okMsg && <div className="portal-alert portal-alert--ok">{okMsg}</div>}
                <div className="portal-row portal-row--between">
                    <button type="submit" className="portal-btn" disabled={pending}>{pending ? 'Creating…' : 'Create account'}</button>
                    <a href="/login" className="portal-btn portal-btn--ghost" style={{ textDecoration: 'none' }}>Already have one?</a>
                </div>
            </form>
        </section>
    );
}
