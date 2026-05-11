import { NextResponse } from 'next/server';
import { z } from 'zod';
import { sql } from '@/lib/db';
import { hashPassword } from '@/lib/passwords';
import { sendVerificationEmail } from '@/lib/email';

const schema = z.object({
    email: z.string().email().max(200),
    password: z.string().min(10).max(200),
    full_name: z.string().min(1).max(200),
    firm: z.string().max(200).optional().nullable()
});

export async function POST(req: Request) {
    let body: unknown;
    try { body = await req.json(); }
    catch { return new NextResponse('Bad JSON', { status: 400 }); }

    const parsed = schema.safeParse(body);
    if (!parsed.success) return new NextResponse(parsed.error.issues[0]?.message || 'Invalid input', { status: 400 });
    const { email, password, full_name, firm } = parsed.data;
    const e = email.toLowerCase();

    const existing = await sql()`select id from users where email = ${e} limit 1` as Array<{ id: string }>;
    if (existing.length > 0) {
        /* Return generic message to avoid revealing whether email is registered. */
        return new NextResponse('If that email is new, a verification link has been sent.', { status: 200 });
    }

    const hash = await hashPassword(password);
    const inserted = await sql()`
        insert into users (email, password_hash, full_name, firm)
        values (${e}, ${hash}, ${full_name}, ${firm || null})
        returning id
    ` as Array<{ id: string }>;
    const userId = inserted[0].id;

    const token = randomToken();
    await sql()`
        insert into email_verifications (token, user_id, expires_at)
        values (${token}, ${userId}, now() + interval '24 hours')
    `;

    const base = process.env.NEXTAUTH_URL || new URL(req.url).origin;
    const link = `${base.replace(/\/$/, '')}/verify?token=${encodeURIComponent(token)}`;
    await sendVerificationEmail(e, link).catch((err) => console.error('email send failed', err));

    return NextResponse.json({ ok: true });
}

function randomToken(): string {
    /* 32 bytes of CSPRNG → 64-char hex. */
    const bytes = new Uint8Array(32);
    crypto.getRandomValues(bytes);
    return Array.from(bytes).map((b) => b.toString(16).padStart(2, '0')).join('');
}
