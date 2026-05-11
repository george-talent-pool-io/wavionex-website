import { signIn } from '@/lib/auth';

export default function LoginPage({ searchParams }: { searchParams: { error?: string; verified?: string } }) {
    async function handle(formData: FormData) {
        'use server';
        await signIn('credentials', {
            email: formData.get('email'),
            password: formData.get('password'),
            redirectTo: '/dashboard'
        });
    }

    return (
        <section className="portal-card">
            <h1>Sign in</h1>
            <p className="portal-muted">Email + password. Sessions are HttpOnly JWT cookies; passwords are hashed with argon2id at rest.</p>

            {searchParams.verified === '1' && (
                <div className="portal-alert portal-alert--ok">Email verified — you can sign in now.</div>
            )}
            {searchParams.error && (
                <div className="portal-alert portal-alert--error">Sign-in failed. Check your credentials.</div>
            )}

            <form action={handle} className="portal-form" autoComplete="on">
                <div className="portal-field">
                    <label htmlFor="email">Email</label>
                    <input id="email" name="email" type="email" required autoComplete="email" />
                </div>
                <div className="portal-field">
                    <label htmlFor="password">Password</label>
                    <input id="password" name="password" type="password" required minLength={10} autoComplete="current-password" />
                </div>
                <div className="portal-row portal-row--between">
                    <button type="submit" className="portal-btn">Sign in</button>
                    <a href="/signup" className="portal-btn portal-btn--ghost" style={{ textDecoration: 'none' }}>Need an account?</a>
                </div>
            </form>
        </section>
    );
}
