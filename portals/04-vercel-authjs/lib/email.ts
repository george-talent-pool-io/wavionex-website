import { Resend } from 'resend';

const FROM = process.env.EMAIL_FROM || 'portal@wavionex.local';

export async function sendVerificationEmail(to: string, link: string) {
    const subject = 'Confirm your Wavionex investor portal email';
    const text =
        `Welcome to the Wavionex investor portal.\n\n` +
        `Confirm your email by clicking the link below — it expires in 24 hours:\n\n${link}\n\n` +
        `If you didn't request this, you can ignore this email.`;

    const apiKey = process.env.RESEND_API_KEY;
    if (!apiKey) {
        console.warn('[email] RESEND_API_KEY not set — logging instead:\n', { to, subject, text });
        return { logged: true };
    }
    const resend = new Resend(apiKey);
    return resend.emails.send({ from: FROM, to, subject, text });
}
