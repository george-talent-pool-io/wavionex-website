/* Wavionex Investor Portal — admin signup notification.
   Triggered by a Supabase Database Webhook on profiles INSERT. Looks up all
   active admins and emails each of them via Resend.

   Required Edge Function secrets:
     RESEND_API_KEY            re_xxxxxxxx (from resend.com)
     EMAIL_FROM                portal@<your-verified-domain>
     ADMIN_PANEL_URL           https://www.wavionex.com/portals/investor/admin/
     SUPABASE_URL              auto-injected
     SUPABASE_SERVICE_ROLE_KEY auto-injected
*/

import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.45.4';

const RESEND_API_KEY = Deno.env.get('RESEND_API_KEY');
const EMAIL_FROM     = Deno.env.get('EMAIL_FROM') || 'portal@wavionex.com';
const ADMIN_URL      = Deno.env.get('ADMIN_PANEL_URL') || 'https://www.wavionex.com/portals/investor/admin/';
const SUPABASE_URL              = Deno.env.get('SUPABASE_URL')!;
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!;

Deno.serve(async (req) => {
    if (req.method !== 'POST') {
        return new Response('method not allowed', { status: 405 });
    }

    let payload: any;
    try { payload = await req.json(); }
    catch { return new Response('bad payload', { status: 400 }); }

    /* Database Webhook payload shape:
       { type: 'INSERT', table: 'profiles', schema: 'public', record: {...}, old_record: null } */
    if (payload?.type !== 'INSERT' || payload?.table !== 'profiles') {
        return new Response(JSON.stringify({ skipped: 'not a profiles insert' }), { status: 200 });
    }
    const profile = payload.record || {};

    /* Skip if the new account is already approved (e.g. admin-created) or is
       itself an admin. We only want to alert on *pending* signups. */
    if (profile.is_approved === true || profile.is_admin === true) {
        return new Response(JSON.stringify({ skipped: 'already privileged' }), { status: 200 });
    }

    if (!RESEND_API_KEY) {
        return new Response(JSON.stringify({ error: 'RESEND_API_KEY not set' }), { status: 500 });
    }

    /* Query admin emails using the service role (bypasses RLS). */
    const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, {
        auth: { persistSession: false }
    });
    const { data: admins, error: adminErr } = await supabase
        .from('profiles')
        .select('email, full_name')
        .eq('is_admin', true);

    if (adminErr) {
        return new Response(JSON.stringify({ error: 'admin query failed', detail: adminErr.message }), { status: 500 });
    }
    if (!admins || admins.length === 0) {
        return new Response(JSON.stringify({ skipped: 'no admins to notify' }), { status: 200 });
    }

    const adminEmails = admins.map((a) => a.email).filter(Boolean);

    const subject = 'New Wavionex investor awaiting approval';
    const text =
        `A new investor has signed up and is awaiting admin approval.\n\n` +
        `Name:  ${profile.full_name || '(not provided)'}\n` +
        `Email: ${profile.email || '(not provided)'}\n` +
        `Firm:  ${profile.firm || '(not provided)'}\n` +
        `Joined: ${profile.created_at || new Date().toISOString()}\n\n` +
        `Approve or revoke them in the admin panel:\n${ADMIN_URL}\n\n` +
        `— Wavionex Investor Portal`;

    const html =
        `<p>A new investor has signed up and is awaiting admin approval.</p>` +
        `<table style="border-collapse:collapse;font-family:Inter,Arial,sans-serif;font-size:14px;line-height:1.5;">` +
        `<tr><td style="padding:4px 12px 4px 0;color:#64748b;">Name</td><td><strong>${esc(profile.full_name) || '(not provided)'}</strong></td></tr>` +
        `<tr><td style="padding:4px 12px 4px 0;color:#64748b;">Email</td><td><strong>${esc(profile.email) || '(not provided)'}</strong></td></tr>` +
        `<tr><td style="padding:4px 12px 4px 0;color:#64748b;">Firm</td><td>${esc(profile.firm) || '(not provided)'}</td></tr>` +
        `<tr><td style="padding:4px 12px 4px 0;color:#64748b;">Joined</td><td>${esc(profile.created_at) || ''}</td></tr>` +
        `</table>` +
        `<p style="margin-top:18px;"><a href="${esc(ADMIN_URL)}" style="background:#6366f1;color:#fff;padding:10px 18px;border-radius:6px;text-decoration:none;font-family:Inter,Arial,sans-serif;font-weight:600;">Open admin panel</a></p>` +
        `<p style="color:#64748b;font-size:12px;">— Wavionex Investor Portal</p>`;

    /* Resend supports an array of recipients in `to`, but for clean audit
       trails (and so each admin sees only themselves on the To: line) we
       send individual messages. */
    const results: Array<{ to: string; ok: boolean; status?: number; error?: string }> = [];
    await Promise.all(adminEmails.map(async (to) => {
        try {
            const resp = await fetch('https://api.resend.com/emails', {
                method: 'POST',
                headers: {
                    Authorization: `Bearer ${RESEND_API_KEY}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ from: EMAIL_FROM, to, subject, text, html })
            });
            results.push({ to, ok: resp.ok, status: resp.status });
            if (!resp.ok) console.error('Resend failed', to, resp.status, await resp.text());
        } catch (ex) {
            results.push({ to, ok: false, error: String(ex) });
            console.error('Resend error', to, ex);
        }
    }));

    return new Response(JSON.stringify({ notified: results }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' }
    });
});

function esc(s: unknown): string {
    return String(s ?? '')
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
