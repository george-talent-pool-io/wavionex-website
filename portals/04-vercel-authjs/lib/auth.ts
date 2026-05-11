import NextAuth from 'next-auth';
import Credentials from 'next-auth/providers/credentials';
import { z } from 'zod';
import { sql } from './db';
import { verifyPassword } from './passwords';

const credSchema = z.object({
    email: z.string().email().max(200),
    password: z.string().min(10).max(200)
});

export const { handlers, auth, signIn, signOut } = NextAuth({
    trustHost: true,
    session: { strategy: 'jwt', maxAge: 60 * 60 * 8 /* 8h */ },
    pages: {
        signIn: '/login'
    },
    providers: [
        Credentials({
            name: 'Email + password',
            credentials: { email: {}, password: {} },
            async authorize(raw) {
                const parsed = credSchema.safeParse(raw);
                if (!parsed.success) return null;
                const { email, password } = parsed.data;

                const rows = await sql()`
                    select id, email, full_name, firm, password_hash, email_verified_at
                    from users
                    where email = ${email.toLowerCase()}
                    limit 1
                ` as Array<{
                    id: string;
                    email: string;
                    full_name: string | null;
                    firm: string | null;
                    password_hash: string;
                    email_verified_at: string | null;
                }>;
                if (rows.length === 0) return null;
                const row = rows[0];
                const ok = await verifyPassword(row.password_hash, password);
                if (!ok) return null;

                return {
                    id: row.id,
                    email: row.email,
                    name: row.full_name || row.email,
                    emailVerified: row.email_verified_at ? new Date(row.email_verified_at) : null
                } as any;
            }
        })
    ],
    callbacks: {
        async jwt({ token, user }) {
            if (user) {
                token.uid = (user as any).id;
                token.emailVerified = (user as any).emailVerified || null;
            }
            return token;
        },
        async session({ session, token }) {
            if (token?.uid) (session.user as any).id = token.uid;
            (session as any).emailVerified = token.emailVerified || null;
            return session;
        }
    }
});
