/* Wavionex Investor Portal — PoC 01 (Supabase).
   The browser talks straight to Supabase using the public anon key. All data
   security lives in Postgres RLS policies (see schema.sql), not in this file. */

import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.45.4';

let supabase = null;
let configOk = false;
try {
    const cfg = await import('./config.js');
    if (cfg && cfg.SUPABASE_URL && cfg.SUPABASE_ANON_KEY && !cfg.SUPABASE_URL.includes('YOUR-PROJECT')) {
        supabase = createClient(cfg.SUPABASE_URL, cfg.SUPABASE_ANON_KEY, {
            auth: { persistSession: true, autoRefreshToken: true, detectSessionInUrl: true }
        });
        configOk = true;
    }
} catch (_err) {
    /* config.js missing — handled below */
}

const $ = (id) => document.getElementById(id);
const views = ['view-signin', 'view-signup', 'view-dashboard', 'view-verify'];

function show(viewId) {
    for (const id of views) $(id).hidden = id !== viewId;
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function showError(el, msg) {
    el.textContent = msg;
    el.hidden = !msg;
}

function showOk(el, msg) {
    el.textContent = msg;
    el.hidden = !msg;
}

if (!configOk) {
    $('config-warning').hidden = false;
    show('view-signin');
    document.querySelectorAll('button[type="submit"], #btn-signout, #link-reset').forEach((b) => (b.disabled = true));
} else {
    bootstrap();
}

async function bootstrap() {
    const { data: { session } } = await supabase.auth.getSession();
    if (session) {
        renderDashboard(session);
    } else {
        show('view-signin');
    }

    supabase.auth.onAuthStateChange((event, session) => {
        if (event === 'SIGNED_OUT' || !session) {
            show('view-signin');
        } else if (event === 'SIGNED_IN' || event === 'TOKEN_REFRESHED') {
            renderDashboard(session);
        }
    });

    /* View switching buttons */
    document.querySelectorAll('[data-view]').forEach((btn) => {
        btn.addEventListener('click', () => show('view-' + btn.dataset.view));
    });

    $('form-signin').addEventListener('submit', onSignin);
    $('form-signup').addEventListener('submit', onSignup);
    $('btn-signout').addEventListener('click', onSignout);
    $('link-reset').addEventListener('click', onPasswordReset);
}

async function onSignin(e) {
    e.preventDefault();
    const err = $('signin-error');
    showError(err, '');
    const email = $('signin-email').value.trim().toLowerCase();
    const password = $('signin-password').value;
    const btn = e.submitter;
    btn.disabled = true;
    try {
        const { error } = await supabase.auth.signInWithPassword({ email, password });
        if (error) showError(err, error.message);
        /* onAuthStateChange will route to dashboard */
    } finally {
        btn.disabled = false;
    }
}

async function onSignup(e) {
    e.preventDefault();
    const err = $('signup-error');
    const ok = $('signup-ok');
    showError(err, '');
    showOk(ok, '');
    const email = $('signup-email').value.trim().toLowerCase();
    const password = $('signup-password').value;
    const fullName = $('signup-name').value.trim();
    const firm = $('signup-firm').value.trim();
    const btn = e.submitter;
    btn.disabled = true;
    try {
        const { data, error } = await supabase.auth.signUp({
            email,
            password,
            options: {
                /* Supabase will email a magic confirmation link that lands back here. */
                emailRedirectTo: window.location.href,
                data: { full_name: fullName, firm }
            }
        });
        if (error) {
            showError(err, error.message);
            return;
        }
        if (data.user && !data.session) {
            $('verify-email').textContent = email;
            show('view-verify');
        } else {
            showOk(ok, 'Account created.');
        }
    } finally {
        btn.disabled = false;
    }
}

async function onSignout() {
    await supabase.auth.signOut();
}

async function onPasswordReset() {
    const email = $('signin-email').value.trim().toLowerCase();
    const err = $('signin-error');
    if (!email) {
        showError(err, 'Type your email above first, then click "Forgot password".');
        return;
    }
    const { error } = await supabase.auth.resetPasswordForEmail(email, {
        redirectTo: window.location.href
    });
    if (error) showError(err, error.message);
    else showError(err, 'Reset email sent. Check your inbox.');
}

async function renderDashboard(session) {
    show('view-dashboard');
    const user = session.user;
    const meta = user.user_metadata || {};
    $('dash-name').textContent = meta.full_name || user.email.split('@')[0];
    $('dash-email').textContent = user.email;
    $('dash-firm').textContent = meta.firm || '—';
    $('dash-verified').textContent = user.email_confirmed_at ? 'yes' : 'no';
    $('dash-created').textContent = new Date(user.created_at).toLocaleDateString();
    $('dash-subtitle').textContent = user.email_confirmed_at
        ? 'You are signed in. RLS-gated investor data is loaded below.'
        : 'Your email is not yet confirmed — deal data will be hidden until you click the link we sent.';

    /* Upsert a profile row tied to auth.uid() so it shows up in the DB
       even on the very first login. RLS allows insert/update by the
       owner only (see schema.sql). */
    await supabase.from('profiles').upsert(
        {
            id: user.id,
            email: user.email,
            full_name: meta.full_name || null,
            firm: meta.firm || null
        },
        { onConflict: 'id' }
    );

    /* Fetch stub deals — readable by any authenticated user per RLS policy. */
    const grid = $('dash-deals');
    grid.innerHTML = '';
    const { data: deals, error } = await supabase
        .from('deals')
        .select('id, name, stage, target_close, headline')
        .order('target_close', { ascending: true });

    if (error) {
        grid.innerHTML = `<p class="portal-alert portal-alert--error">Couldn’t load deals: ${escapeHtml(error.message)}</p>`;
        return;
    }
    if (!deals || deals.length === 0) {
        grid.innerHTML = '<p class="portal-muted">No deals yet — seed some rows via schema.sql.</p>';
        return;
    }
    for (const d of deals) {
        const el = document.createElement('article');
        el.className = 'portal-deal';
        el.innerHTML = `
            <span class="meta">${escapeHtml(d.stage || '')}</span>
            <h4>${escapeHtml(d.name)}</h4>
            <p class="portal-muted" style="margin: 0.25rem 0 0 0;">${escapeHtml(d.headline || '')}</p>
            <p class="portal-muted" style="margin: 0.5rem 0 0 0; font-size: 0.78rem;">
                Target close ${d.target_close ? new Date(d.target_close).toLocaleDateString() : '—'}
            </p>`;
        grid.appendChild(el);
    }
}

function escapeHtml(s) {
    return String(s == null ? '' : s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}
