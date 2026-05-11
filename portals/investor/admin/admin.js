/* Wavionex Investor Portal — Admin.
   Same Supabase auth as the investor app. RLS + the security-definer
   helper functions in schema.sql gate everything; we trust the API
   responses and don't replicate authorisation logic here. */

import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.45.4';

let supabase = null;
let cfg = null;
try {
    cfg = await import('../config.js');
    if (cfg && cfg.SUPABASE_URL && cfg.SUPABASE_ANON_KEY && !cfg.SUPABASE_URL.includes('YOUR-PROJECT')) {
        supabase = createClient(cfg.SUPABASE_URL, cfg.SUPABASE_ANON_KEY, {
            auth: { persistSession: true, autoRefreshToken: true, detectSessionInUrl: true }
        });
    }
} catch (_) { /* missing */ }

const $ = (id) => document.getElementById(id);

if (!supabase) {
    $('config-warning').hidden = false;
    $('view-locked').hidden = false;
} else {
    bootstrap();
}

async function bootstrap() {
    const { data: { session } } = await supabase.auth.getSession();
    if (!session) {
        $('view-locked').hidden = false;
        $('locked-msg').textContent = 'You need to sign in (use the main investor portal) before opening the admin.';
        return;
    }
    const { data: profile, error } = await supabase
        .from('profiles')
        .select('email, full_name, is_admin')
        .eq('id', session.user.id)
        .maybeSingle();
    if (error || !profile || !profile.is_admin) {
        $('view-locked').hidden = false;
        $('locked-msg').textContent = 'Your account is not flagged as admin. Promote it with `update profiles set is_admin = true where email = …` in the Supabase SQL editor.';
        return;
    }
    $('view-admin').hidden = false;
    $('admin-whoami').textContent = profile.email + ' · admin';

    $('btn-signout').addEventListener('click', () => supabase.auth.signOut().then(() => location.href = '../'));
    wireTabs();
    wireExternalLinks();

    await Promise.all([loadUsers(), loadInvites(), loadAudit(), loadPapers()]);

    $('users-filter').addEventListener('input', loadUsers);
    $('audit-filter').addEventListener('input', loadAudit);
    $('audit-type').addEventListener('change', loadAudit);
    $('audit-refresh').addEventListener('click', loadAudit);
    $('form-invite').addEventListener('submit', onCreateInvite);
    $('form-paper').addEventListener('submit', onUploadPaper);

    /* Tab links inside body text */
    document.querySelectorAll('[data-tab-link]').forEach((el) => {
        el.addEventListener('click', (e) => {
            e.preventDefault();
            switchTab(el.dataset.tabLink);
        });
    });
}

function wireTabs() {
    document.querySelectorAll('.admin-tab').forEach((btn) => {
        btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });
}

function switchTab(name) {
    document.querySelectorAll('.admin-tab').forEach((b) => b.setAttribute('aria-selected', b.dataset.tab === name ? 'true' : 'false'));
    document.querySelectorAll('.admin-section').forEach((s) => {
        if (s.id === 'tab-' + name) s.setAttribute('data-active', '');
        else s.removeAttribute('data-active');
    });
}

function wireExternalLinks() {
    const ref = projectRef();
    if (!ref) return;
    const base = `https://supabase.com/dashboard/project/${ref}`;
    $('ext-auth-logs').href   = base + '/logs/explorer';
    $('ext-auth-users').href  = base + '/auth/users';
    $('ext-storage').href     = base + '/storage/buckets/papers';
    $('ext-sql').href         = base + '/sql/new';
    $('ext-rate-limits').href = base + '/auth/rate-limits';
    $('ext-attack').href      = base + '/auth/protection';
}

function projectRef() {
    try {
        const u = new URL(cfg.SUPABASE_URL);
        return u.hostname.split('.')[0];
    } catch (_) { return null; }
}

/* ---------- Users tab ---------- */

let allUsers = [];

async function loadUsers() {
    const filter = $('users-filter').value.trim().toLowerCase();
    const { data, error } = await supabase
        .from('profiles')
        .select('id, email, full_name, firm, is_approved, is_admin, email_verified_at, last_sign_in_at, created_at')
        .order('created_at', { ascending: false });
    if (error) {
        $('users-table').querySelector('tbody').innerHTML = `<tr><td colspan="8" class="portal-alert portal-alert--error">${esc(error.message)}</td></tr>`;
        return;
    }
    allUsers = data || [];
    const filtered = filter
        ? allUsers.filter((u) => [u.email, u.full_name, u.firm].some((v) => (v || '').toLowerCase().includes(filter)))
        : allUsers;
    $('users-count').textContent = `${filtered.length} of ${allUsers.length}`;
    const tbody = $('users-table').querySelector('tbody');
    tbody.innerHTML = '';
    if (filtered.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" class="portal-muted">No users.</td></tr>';
        return;
    }
    for (const u of filtered) {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${esc(u.email)}</td>
            <td>${esc(u.full_name || '—')}<div class="portal-muted" style="font-size:0.75rem;">${esc(u.firm || '')}</div></td>
            <td>${u.email_verified_at ? '<span class="pill pill--on">verified</span>' : '<span class="pill pill--off">no</span>'}</td>
            <td><button class="portal-btn portal-btn--ghost" style="padding:0.25rem 0.55rem; font-size:0.78rem;" data-act="toggle-approve" data-id="${esc(u.id)}" data-cur="${u.is_approved ? '1' : '0'}">${u.is_approved ? 'Revoke' : 'Approve'}</button></td>
            <td><button class="portal-btn portal-btn--ghost" style="padding:0.25rem 0.55rem; font-size:0.78rem;" data-act="toggle-admin"   data-id="${esc(u.id)}" data-cur="${u.is_admin    ? '1' : '0'}">${u.is_admin    ? 'Demote' : 'Make admin'}</button></td>
            <td>${u.last_sign_in_at ? esc(new Date(u.last_sign_in_at).toLocaleString()) : '—'}</td>
            <td>${esc(new Date(u.created_at).toLocaleDateString())}</td>
            <td class="mono">${esc(u.id.slice(0, 8))}…</td>`;
        tbody.appendChild(tr);
    }
    tbody.querySelectorAll('button[data-act]').forEach((btn) => {
        btn.addEventListener('click', () => onToggle(btn.dataset.act, btn.dataset.id, btn.dataset.cur === '1'));
    });
}

async function onToggle(act, id, current) {
    const field = act === 'toggle-approve' ? 'is_approved' : 'is_admin';
    const next = !current;
    const { error } = await supabase.from('profiles').update({ [field]: next }).eq('id', id);
    if (error) {
        alert(`${field} update failed: ${error.message}`);
        return;
    }
    await loadUsers();
}

/* ---------- Invites tab ---------- */

async function loadInvites() {
    const { data, error } = await supabase
        .from('invite_codes')
        .select('id, code, note, invited_email, invited_name, max_uses, used_count, expires_at, revoked_at, created_at')
        .order('created_at', { ascending: false });
    const tbody = $('invites-table').querySelector('tbody');
    tbody.innerHTML = '';
    if (error) {
        tbody.innerHTML = `<tr><td colspan="7" class="portal-alert portal-alert--error">${esc(error.message)}</td></tr>`;
        return;
    }
    if (!data || data.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="portal-muted">No invites yet.</td></tr>';
        return;
    }
    for (const i of data) {
        const status = i.revoked_at
            ? '<span class="pill pill--off">revoked</span>'
            : (i.expires_at && new Date(i.expires_at) < new Date())
                ? '<span class="pill pill--off">expired</span>'
                : (i.used_count >= i.max_uses)
                    ? '<span class="pill pill--off">used up</span>'
                    : '<span class="pill pill--on">active</span>';
        const recipient = i.invited_name || i.invited_email
            ? `${esc(i.invited_name || '')}${i.invited_email ? `<div class="portal-muted" style="font-size:0.75rem;">${esc(i.invited_email)}</div>` : ''}`
            : '<span class="portal-muted">—</span>';
        const noteLine = i.note ? `<div class="portal-muted" style="font-size:0.72rem; margin-top:0.15rem;">${esc(i.note)}</div>` : '';
        const canShare = !i.revoked_at && i.used_count < i.max_uses;
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><span class="copy-code" data-copy="${esc(i.code)}" title="Click to copy">${esc(i.code)}</span></td>
            <td>${recipient}${noteLine}</td>
            <td>${i.used_count} / ${i.max_uses}</td>
            <td>${i.expires_at ? esc(new Date(i.expires_at).toLocaleDateString()) : '—'}</td>
            <td>${status}</td>
            <td>${esc(new Date(i.created_at).toLocaleDateString())}</td>
            <td>
                ${canShare ? `<button class="portal-btn portal-btn--ghost" style="padding:0.25rem 0.55rem; font-size:0.78rem;" data-share='${esc(JSON.stringify(i))}'>Share</button>` : ''}
                ${i.revoked_at ? '' : `<button class="portal-btn portal-btn--ghost" style="padding:0.25rem 0.55rem; font-size:0.78rem; margin-left:0.25rem;" data-revoke="${esc(i.id)}">Revoke</button>`}
            </td>`;
        tbody.appendChild(tr);
    }
    tbody.querySelectorAll('.copy-code').forEach((el) => {
        el.addEventListener('click', () => {
            navigator.clipboard?.writeText(el.dataset.copy);
            el.textContent = 'copied';
            setTimeout(() => (el.textContent = el.dataset.copy), 900);
        });
    });
    tbody.querySelectorAll('button[data-revoke]').forEach((btn) => {
        btn.addEventListener('click', () => onRevoke(btn.dataset.revoke));
    });
    tbody.querySelectorAll('button[data-share]').forEach((btn) => {
        btn.addEventListener('click', () => {
            try { showInviteResult(JSON.parse(btn.dataset.share)); } catch (_) {}
        });
    });
}

async function onRevoke(id) {
    if (!confirm('Revoke this invite? Nobody else will be able to use it.')) return;
    const { error } = await supabase.from('invite_codes').update({ revoked_at: new Date().toISOString() }).eq('id', id);
    if (error) alert('Revoke failed: ' + error.message);
    else      await loadInvites();
}

async function onCreateInvite(e) {
    e.preventDefault();
    const msg = $('invite-msg');
    msg.hidden = true;
    msg.className = 'portal-alert';
    const code     = $('invite-code').value.trim() || generateCode();
    const note     = $('invite-note').value.trim() || null;
    const invName  = $('invite-recipient-name').value.trim()  || null;
    const invEmail = $('invite-recipient-email').value.trim().toLowerCase() || null;
    const maxUses  = Math.max(1, parseInt($('invite-max').value, 10) || 1);
    const expiresD = $('invite-expires').value;
    const expires_at = expiresD ? new Date(expiresD + 'T23:59:59Z').toISOString() : null;
    const { data: { user } } = await supabase.auth.getUser();
    const { data, error } = await supabase
        .from('invite_codes')
        .insert({
            code, note,
            invited_name: invName, invited_email: invEmail,
            max_uses: maxUses, expires_at,
            created_by: user?.id
        })
        .select()
        .single();
    if (error) {
        msg.textContent = error.message;
        msg.className = 'portal-alert portal-alert--error';
        msg.hidden = false;
        return;
    }
    $('form-invite').reset();
    $('invite-max').value = '1';
    showInviteResult(data);
    await loadInvites();
}

/* Renders the share-an-invite card with email subject/body, mailto, and copy buttons.
   Called both right after creation and from each row's "Share" button. */
function showInviteResult(inv) {
    const portalBase = inviteUrlBase();
    const link = `${portalBase}?invite=${encodeURIComponent(inv.code)}#signup`;
    const subject = 'Wavionex Investor Portal — your access invitation';
    const expires = inv.expires_at
        ? `The code expires on ${new Date(inv.expires_at).toLocaleDateString()}.`
        : 'The code does not expire.';
    const usesLine = inv.max_uses === 1
        ? 'It is a single-use code.'
        : `It is good for ${inv.max_uses} sign-ups.`;
    const greet = inv.invited_name ? `Hi ${inv.invited_name},` : 'Hello,';
    const body = [
        greet, '',
        'You have been invited to the Wavionex Investor Portal — our private space for diligence materials, deal pipeline, and research papers.',
        '',
        'Your invite code:', '',
        '    ' + inv.code, '',
        'Sign up: ' + link, '',
        expires + ' ' + usesLine,
        '',
        'After verifying your email, an admin will approve your account and the deal pipeline + papers will become visible.',
        '',
        '— Wavionex'
    ].join('\n');

    $('result-code').textContent  = inv.code;
    $('result-code').dataset.copy = inv.code;
    $('result-link').textContent  = link;
    $('result-link').href         = link;
    $('result-subject').value     = subject;
    $('result-body').value        = body;

    const mailto = (() => {
        const params = new URLSearchParams();
        params.set('subject', subject);
        params.set('body', body);
        const to = inv.invited_email ? encodeURIComponent(inv.invited_email) : '';
        return `mailto:${to}?${params.toString()}`;
    })();
    $('result-mailto').href = mailto;

    $('invite-result').hidden = false;
    $('invite-result').scrollIntoView({ behavior: 'smooth', block: 'start' });

    /* Re-wire button handlers each time so they always reference the current invite. */
    $('result-code').onclick = () => copyAndFlash($('result-code'), inv.code);
    $('result-copy-body').onclick = () => copyText(body,    $('result-copy-body'), 'Copy email body');
    $('result-copy-link').onclick = () => copyText(link,    $('result-copy-link'), 'Copy link');
    $('result-dismiss').onclick = () => { $('invite-result').hidden = true; };
}

function inviteUrlBase() {
    /* The admin lives at .../portals/investor/admin/ — strip the admin/ suffix. */
    const u = new URL(window.location.href);
    u.pathname = u.pathname.replace(/admin\/?$/, '');
    u.search = '';
    u.hash = '';
    return u.toString();
}

function copyText(text, btn, originalLabel) {
    navigator.clipboard?.writeText(text);
    if (btn) {
        const prev = btn.textContent;
        btn.textContent = 'Copied';
        setTimeout(() => (btn.textContent = originalLabel || prev), 900);
    }
}

function copyAndFlash(el, text) {
    navigator.clipboard?.writeText(text);
    const prev = el.textContent;
    el.textContent = 'copied';
    setTimeout(() => (el.textContent = prev), 900);
}

function generateCode() {
    const part = (n) => Array.from({ length: n }, () => 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'[Math.floor(Math.random() * 32)]).join('');
    return `WAVE-${new Date().getFullYear()}-${part(4)}`;
}

/* ---------- Audit tab ---------- */

async function loadAudit() {
    const type   = $('audit-type').value;
    const filter = $('audit-filter').value.trim().toLowerCase();
    let q = supabase.from('audit_events')
        .select('id, user_id, event_type, event_data, user_agent, created_at')
        .order('created_at', { ascending: false })
        .limit(200);
    if (type) q = q.eq('event_type', type);
    const { data, error } = await q;
    const tbody = $('audit-table').querySelector('tbody');
    tbody.innerHTML = '';
    if (error) {
        tbody.innerHTML = `<tr><td colspan="4" class="portal-alert portal-alert--error">${esc(error.message)}</td></tr>`;
        return;
    }
    const userEmailFor = new Map(allUsers.map((u) => [u.id, u.email]));
    /* fill in any missing emails (admin can read all profiles) */
    const missing = (data || []).map((r) => r.user_id).filter((id) => id && !userEmailFor.has(id));
    if (missing.length) {
        const { data: extras } = await supabase.from('profiles').select('id, email').in('id', missing);
        (extras || []).forEach((e) => userEmailFor.set(e.id, e.email));
    }
    const filtered = (data || []).filter((r) => {
        if (!filter) return true;
        const e = userEmailFor.get(r.user_id) || '';
        return e.toLowerCase().includes(filter);
    });
    if (filtered.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" class="portal-muted">No events.</td></tr>';
        return;
    }
    for (const r of filtered) {
        const tr = document.createElement('tr');
        const email = userEmailFor.get(r.user_id) || '—';
        tr.innerHTML = `
            <td>${esc(new Date(r.created_at).toLocaleString())}</td>
            <td>${esc(email)}<div class="portal-muted" style="font-size:0.7rem;">${esc((r.user_agent || '').slice(0, 60))}</div></td>
            <td><span class="pill">${esc(r.event_type)}</span></td>
            <td class="mono">${esc(JSON.stringify(r.event_data || {}))}</td>`;
        tbody.appendChild(tr);
    }
}

/* ---------- Papers tab ---------- */

async function loadPapers() {
    const { data, error } = await supabase
        .from('papers')
        .select('id, title, description, storage_path, file_size, mime_type, created_at')
        .order('created_at', { ascending: false });
    const tbody = $('papers-table').querySelector('tbody');
    tbody.innerHTML = '';
    if (error) {
        tbody.innerHTML = `<tr><td colspan="5" class="portal-alert portal-alert--error">${esc(error.message)}</td></tr>`;
        return;
    }
    if (!data || data.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="portal-muted">No papers uploaded yet.</td></tr>';
        return;
    }
    for (const p of data) {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><strong>${esc(p.title)}</strong></td>
            <td class="portal-muted">${esc(p.description || '—')}</td>
            <td>${p.file_size ? esc(humanSize(p.file_size)) : '—'}</td>
            <td>${esc(new Date(p.created_at).toLocaleDateString())}</td>
            <td><button class="portal-btn portal-btn--ghost" style="padding:0.25rem 0.55rem; font-size:0.78rem;" data-paper-del="${esc(p.id)}" data-path="${esc(p.storage_path)}">Delete</button></td>`;
        tbody.appendChild(tr);
    }
    tbody.querySelectorAll('button[data-paper-del]').forEach((btn) => {
        btn.addEventListener('click', () => onDeletePaper(btn.dataset.paperDel, btn.dataset.path));
    });
}

async function onUploadPaper(e) {
    e.preventDefault();
    const msg = $('paper-msg');
    msg.hidden = true;
    msg.className = 'portal-alert';
    const file = $('paper-file').files[0];
    const title = $('paper-title').value.trim();
    const desc  = $('paper-desc').value.trim() || null;
    if (!file || !title) {
        msg.textContent = 'Title and file are both required.';
        msg.className = 'portal-alert portal-alert--error';
        msg.hidden = false;
        return;
    }
    if (file.size > 25 * 1024 * 1024) {
        msg.textContent = 'File is larger than 25 MB.';
        msg.className = 'portal-alert portal-alert--error';
        msg.hidden = false;
        return;
    }
    const path = `${new Date().getFullYear()}/${crypto.randomUUID()}-${file.name.replace(/[^A-Za-z0-9._-]/g, '_')}`;
    const up = await supabase.storage.from('papers').upload(path, file, { contentType: file.type || 'application/pdf', upsert: false });
    if (up.error) {
        msg.textContent = 'Upload failed: ' + up.error.message + ' (is the `papers` bucket created and policies applied?)';
        msg.className = 'portal-alert portal-alert--error';
        msg.hidden = false;
        return;
    }
    const { data: { user } } = await supabase.auth.getUser();
    const { error } = await supabase.from('papers').insert({
        title, description: desc,
        storage_path: path,
        file_size: file.size,
        mime_type: file.type || 'application/pdf',
        created_by: user?.id
    });
    if (error) {
        msg.textContent = 'Metadata insert failed: ' + error.message;
        msg.className = 'portal-alert portal-alert--error';
        msg.hidden = false;
        return;
    }
    msg.textContent = 'Uploaded.';
    msg.className = 'portal-alert portal-alert--ok';
    msg.hidden = false;
    $('form-paper').reset();
    await loadPapers();
}

async function onDeletePaper(id, path) {
    if (!confirm('Delete this paper permanently?')) return;
    const rm = await supabase.storage.from('papers').remove([path]);
    if (rm.error) console.warn('storage delete failed (will still remove metadata):', rm.error);
    const { error } = await supabase.from('papers').delete().eq('id', id);
    if (error) alert('Delete failed: ' + error.message);
    else      await loadPapers();
}

function humanSize(n) {
    if (n < 1024) return n + ' B';
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' KB';
    return (n / 1024 / 1024).toFixed(2) + ' MB';
}

function esc(s) {
    return String(s == null ? '' : s)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
