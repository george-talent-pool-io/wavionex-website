import type { ReactNode } from 'react';
import './globals.css';

export const metadata = {
    title: 'Wavionex Investor Portal — Vercel + Auth.js PoC',
    robots: 'noindex'
};

export default function RootLayout({ children }: { children: ReactNode }) {
    return (
        <html lang="en">
            <body>
                <main className="portal-shell">
                    <header className="portal-header">
                        <a href="/" className="portal-brand">
                            <span className="portal-brand-dot" />
                            <span>Wavionex Investor Portal</span>
                        </a>
                        <span className="portal-tag">PoC&nbsp;04 · Vercel + Auth.js</span>
                    </header>
                    {children}
                </main>
            </body>
        </html>
    );
}
