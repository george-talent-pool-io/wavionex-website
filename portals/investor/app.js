/* Wavionex Investor Portal — Supabase.
   The browser talks to Supabase using the public publishable key. All data
   security lives in Postgres RLS policies (see schema.sql), not in this file. */

import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.45.4';
import { mountNav }     from '../_shared/portal-nav.js';

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
} catch (_err) { /* missing config.js handled below */ }

/* Render the shared nav up-front so the page never looks unstyled even if
   Supabase / config is broken. setUser(...) is called later when we know
   who is logged in. */
const nav = mountNav(document.getElementById('portal-nav-mount'), {
    onSignOut: async () => { if (supabase) await supabase.auth.signOut(); }
});
nav.setUser(null);

const $ = (id) => document.getElementById(id);
const views = ['view-signin', 'view-signup', 'view-dashboard', 'view-verify'];

function show(viewId) {
    for (const id of views) $(id).hidden = id !== viewId;
    /* Vertically centre the card for auth views; let dashboard scroll normally. */
    if (viewId === 'view-dashboard') {
        document.body.classList.remove('portal-auth');
    } else {
        document.body.classList.add('portal-auth');
    }
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function showError(el, msg) { el.textContent = msg; el.hidden = !msg; }
function showOk(el, msg)    { el.textContent = msg; el.hidden = !msg; }

if (!configOk) {
    $('config-warning').hidden = false;
    show('view-signin');
    document.querySelectorAll('button[type="submit"], #link-reset').forEach((b) => (b.disabled = true));
} else {
    bootstrap();
}

async function bootstrap() {
    const { data: { session } } = await supabase.auth.getSession();
    if (session) {
        renderDashboard(session);
    } else {
        /* Honour ?invite=CODE + #signup so admin-shared links land on a prefilled form. */
        const url = new URL(window.location.href);
        const presetCode = url.searchParams.get('invite');
        const wantsSignup = window.location.hash === '#signup' || !!presetCode;
        if (wantsSignup) {
            show('view-signup');
            if (presetCode) {
                $('signup-invite').value = presetCode;
                /* Trigger validation immediately so the user sees confirmation. */
                queueMicrotask(validateInviteOnChange);
            }
        } else {
            show('view-signin');
        }
    }

    supabase.auth.onAuthStateChange((event, session) => {
        if (event === 'SIGNED_OUT' || !session) {
            nav.setUser(null);
            show('view-signin');
        } else if (event === 'SIGNED_IN' || event === 'TOKEN_REFRESHED') {
            renderDashboard(session);
        }
    });

    document.querySelectorAll('[data-view]').forEach((btn) => {
        btn.addEventListener('click', () => show('view-' + btn.dataset.view));
    });

    $('form-signin').addEventListener('submit', onSignin);
    $('form-signup').addEventListener('submit', onSignup);
    $('link-reset').addEventListener('click', onPasswordReset);
    $('signup-invite').addEventListener('input', () => {
        clearTimeout(inviteDebounceT);
        inviteDebounceT = setTimeout(validateInviteOnChange, 350);
    });
    $('signup-invite').addEventListener('blur', validateInviteOnChange);
}

let inviteDebounceT = null;
let lastValidatedCode = '';

async function validateInviteOnChange() {
    const input = $('signup-invite');
    const fb    = $('invite-feedback');
    const raw   = input.value.trim();
    if (!raw) {
        fb.textContent = '';
        fb.style.color = '';
        lastValidatedCode = '';
        return;
    }
    if (raw === lastValidatedCode) return;
    lastValidatedCode = raw;
    fb.textContent = 'Checking…';
    fb.style.color = '';
    try {
        const { data, error } = await supabase.rpc('validate_invite_code', { p_code: raw });
        if (raw !== input.value.trim()) return; /* user kept typing; stale */
        if (error) {
            fb.textContent = 'Couldn’t check the code: ' + error.message;
            fb.style.color = '';
            return;
        }
        switch (data) {
            case 'ok':       fb.textContent = '✓ Valid invite code.';            fb.style.color = 'var(--brand-success)'; break;
            case 'invalid':  fb.textContent = '✗ Unknown invite code.';          fb.style.color = '#FCA5A5'; break;
            case 'revoked':  fb.textContent = '✗ This code has been revoked.';   fb.style.color = '#FCA5A5'; break;
            case 'expired':  fb.textContent = '✗ This code has expired.';        fb.style.color = '#FCA5A5'; break;
            case 'used_up':  fb.textContent = '✗ This code has been used up.';   fb.style.color = '#FCA5A5'; break;
            default:         fb.textContent = '';
        }
    } catch (ex) {
        fb.textContent = '';
    }
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
    } finally {
        btn.disabled = false;
    }
}

async function onSignup(e) {
    e.preventDefault();
    const err = $('signup-error');
    const ok  = $('signup-ok');
    showError(err, '');
    showOk(ok, '');
    const inviteCode = $('signup-invite').value.trim();
    const email      = $('signup-email').value.trim().toLowerCase();
    const password   = $('signup-password').value;
    const fullName   = $('signup-name').value.trim();
    const firm       = $('signup-firm').value.trim();
    if (!inviteCode) {
        showError(err, 'Invite code is required.');
        return;
    }
    const btn = e.submitter;
    btn.disabled = true;
    try {
        const { data, error } = await supabase.auth.signUp({
            email,
            password,
            options: {
                emailRedirectTo: window.location.href,
                data: { full_name: fullName, firm, invite_code: inviteCode }
            }
        });
        if (error) {
            showError(err, friendlySignupError(error.message));
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

function friendlySignupError(msg) {
    /* The trigger raises Postgres exceptions that Supabase surfaces as
       'Database error saving new user: <reason>'. Pull out the human bit. */
    if (!msg) return 'Sign-up failed.';
    const m = String(msg).match(/Database error saving new user:?\s*(.*)$/i);
    const reason = m ? m[1].trim() : msg;
    if (/invite_code is required/i.test(reason))        return 'Invite code is required.';
    if (/invalid invite code/i.test(reason))            return 'That invite code isn’t valid.';
    if (/revoked/i.test(reason))                        return 'That invite code has been revoked.';
    if (/expired/i.test(reason))                        return 'That invite code has expired.';
    if (/no remaining uses/i.test(reason))              return 'That invite code has been used up.';
    if (/already.*registered|duplicate.*email/i.test(reason)) return 'An account with that email already exists.';
    return reason || msg;
}

async function onPasswordReset(e) {
    if (e && typeof e.preventDefault === 'function') e.preventDefault();
    const email = $('signin-email').value.trim().toLowerCase();
    const err   = $('signin-error');
    if (!email) {
        showError(err, 'Type your email above first, then click "Forgot password?".');
        return;
    }
    const { error } = await supabase.auth.resetPasswordForEmail(email, {
        redirectTo: window.location.href
    });
    if (error) showError(err, error.message);
    else       showError(err, 'Reset email sent. Check your inbox.');
}

async function renderDashboard(session) {
    show('view-dashboard');
    const user = session.user;
    const meta = user.user_metadata || {};

    /* Read our own profile row. Created by the signup trigger; RLS allows self-read. */
    const { data: profile, error: pErr } = await supabase
        .from('profiles')
        .select('email, full_name, firm, is_approved, is_admin, email_verified_at, created_at')
        .eq('id', user.id)
        .maybeSingle();

    if (pErr) console.warn('profile load error', pErr);

    const fullName = (profile && profile.full_name) || meta.full_name || user.email.split('@')[0];
    const firm     = (profile && profile.firm)      || meta.firm      || '—';

    $('dash-name').textContent     = fullName;
    $('dash-email').textContent    = user.email;
    $('dash-firm').textContent     = firm;
    $('dash-verified').textContent = user.email_confirmed_at ? 'yes' : 'no';
    $('dash-created').textContent  = profile ? new Date(profile.created_at).toLocaleDateString() : new Date(user.created_at).toLocaleDateString();

    const verified  = !!user.email_confirmed_at;
    const approved  = !!(profile && profile.is_approved);
    const isAdmin   = !!(profile && profile.is_admin);

    $('dash-approval').textContent = isAdmin ? 'admin' : (approved ? 'approved' : 'pending');

    /* Push profile data into the shared nav so the avatar + dropdown reflect it.
       Wrapped in try/catch so a nav bug doesn't block papers + deals loading. */
    try {
        nav.setUser({
            email: user.email,
            fullName,
            firm: profile && profile.firm ? profile.firm : null,
            isApproved: approved,
            isAdmin
        });
    } catch (navErr) {
        console.error('nav.setUser failed:', navErr);
    }

    if (!verified) {
        $('dash-subtitle').textContent = 'Your email is not yet confirmed — check your inbox for the link we sent.';
    } else if (!approved && !isAdmin) {
        $('dash-subtitle').textContent = 'Signed in. Awaiting admin approval before deals + papers unlock.';
    } else {
        $('dash-subtitle').textContent = 'Signed in. RLS-gated investor data is loaded below.';
    }
    $('dash-pending').hidden = !(verified && !approved && !isAdmin);

    /* Fire a portal-open audit event. RLS lets us insert our own rows. */
    void supabase.from('audit_events').insert({
        user_id: user.id,
        event_type: 'portal_open',
        event_data: { path: window.location.pathname },
        user_agent: navigator.userAgent.slice(0, 240)
    });

    await loadPapers(approved || isAdmin);
    await loadDeals(approved || isAdmin);
}

async function loadPapers(unlocked) {
    const list = $('dash-papers');
    list.innerHTML = '';
    if (!unlocked) {
        list.innerHTML = '<li class="portal-muted">Papers will appear here once your account is approved.</li>';
        return;
    }
    const { data: papers, error } = await supabase
        .from('papers')
        .select('id, title, description, storage_path, file_size, mime_type, created_at')
        .order('created_at', { ascending: false });
    if (error) {
        list.innerHTML = `<li class="portal-alert portal-alert--error">Couldn’t load papers: ${escapeHtml(error.message)}</li>`;
        return;
    }
    if (!papers || papers.length === 0) {
        list.innerHTML = '<li class="portal-muted">No papers yet.</li>';
        return;
    }
    for (const p of papers) {
        const li = document.createElement('li');
        li.style.padding = '0.7rem 0';
        li.style.borderBottom = '1px solid var(--brand-border)';
        const size = p.file_size ? humanSize(p.file_size) : '';
        li.innerHTML = `
            <div class="portal-row portal-row--between">
                <div>
                    <div style="font-weight:600;color:var(--brand-light);">${escapeHtml(p.title)}</div>
                    <div class="portal-muted" style="font-size:0.85rem;">${escapeHtml(p.description || '')}</div>
                    <div class="portal-muted" style="font-size:0.75rem; margin-top:0.25rem;">${escapeHtml(p.mime_type || '')} ${size ? '· ' + size : ''}</div>
                </div>
                <button class="portal-btn portal-btn--ghost" type="button" data-paper-id="${escapeHtml(p.id)}" data-paper-path="${escapeHtml(p.storage_path)}" data-paper-title="${escapeHtml(p.title)}">Download</button>
            </div>`;
        list.appendChild(li);
    }
    list.querySelectorAll('button[data-paper-id]').forEach((btn) => {
        btn.addEventListener('click', () => downloadPaper(btn.dataset.paperId, btn.dataset.paperPath, btn.dataset.paperTitle));
    });
}

async function downloadPaper(id, path, title) {
    /* Generate a short-lived signed URL. Storage RLS still gates this. */
    const { data, error } = await supabase.storage.from('papers').createSignedUrl(path, 60);
    if (error) {
        alert('Download failed: ' + error.message);
        return;
    }
    void supabase.from('audit_events').insert({
        user_id: (await supabase.auth.getUser()).data.user.id,
        event_type: 'paper_download',
        event_data: { paper_id: id, title, path },
        user_agent: navigator.userAgent.slice(0, 240)
    });
    window.open(data.signedUrl, '_blank', 'noopener,noreferrer');
}

async function loadDeals(unlocked) {
    const grid = $('dash-deals');
    grid.innerHTML = '';
    if (!unlocked) {
        grid.innerHTML = '<p class="portal-muted">Deal pipeline unlocks after admin approval.</p>';
        return;
    }
    const { data: deals, error } = await supabase
        .from('deals')
        .select('id, name, stage, target_close, headline')
        .order('target_close', { ascending: true });
    if (error) {
        grid.innerHTML = `<p class="portal-alert portal-alert--error">Couldn’t load deals: ${escapeHtml(error.message)}</p>`;
        return;
    }
    if (!deals || deals.length === 0) {
        grid.innerHTML = '<p class="portal-muted">No deals yet — seed via schema.sql.</p>';
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

function humanSize(n) {
    if (n < 1024) return n + ' B';
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' KB';
    return (n / 1024 / 1024).toFixed(2) + ' MB';
}

function escapeHtml(s) {
    return String(s == null ? '' : s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}
