/* Wavionex Investor Portal — PoC 03 (Clerk + Neon).
   Clerk provides auth (hosted SignIn/SignUp components). When the user
   is signed in we call /api/profile and /api/deals on the same origin;
   those are Cloudflare Pages Functions (see ../functions/api/) that
   verify the Clerk JWT before reaching Neon. */

let CLERK_PUBLISHABLE_KEY = null;
let CLERK_JS_VERSION = '5';
let configOk = false;

try {
    const cfg = await import('./config.js');
    if (cfg && cfg.CLERK_PUBLISHABLE_KEY && !cfg.CLERK_PUBLISHABLE_KEY.includes('XXXXXXX')) {
        CLERK_PUBLISHABLE_KEY = cfg.CLERK_PUBLISHABLE_KEY;
        if (cfg.CLERK_JS_VERSION) CLERK_JS_VERSION = cfg.CLERK_JS_VERSION;
        configOk = true;
    }
} catch (_) { /* missing config */ }

const $ = (id) => document.getElementById(id);

if (!configOk) {
    $('config-warning').hidden = false;
} else {
    bootstrap().catch((e) => {
        console.error(e);
        $('config-warning').hidden = false;
        $('config-warning').textContent = 'Clerk load failed: ' + (e && e.message || e);
    });
}

async function bootstrap() {
    /* Dynamically load Clerk JS from the official CDN, branded with our publishable key. */
    const frontendApi = decodeFrontendApi(CLERK_PUBLISHABLE_KEY);
    await loadScript(`https://${frontendApi}/npm/@clerk/clerk-js@${CLERK_JS_VERSION}/dist/clerk.browser.js`, {
        'data-clerk-publishable-key': CLERK_PUBLISHABLE_KEY
    });
    const Clerk = window.Clerk;
    if (!Clerk) throw new Error('Clerk failed to expose window.Clerk');
    await Clerk.load({
        appearance: { variables: { colorPrimary: '#6366F1', colorBackground: '#0F172A', colorText: '#E2E8F0' } }
    });

    function render() {
        if (Clerk.user) {
            $('view-signedout').hidden = true;
            $('view-dashboard').hidden = false;
            const u = Clerk.user;
            $('dash-name').textContent = u.firstName || (u.primaryEmailAddress?.emailAddress?.split('@')[0]) || 'investor';
            $('dash-email').textContent = u.primaryEmailAddress?.emailAddress || '—';
            $('dash-verified').textContent = u.primaryEmailAddress?.verification?.status === 'verified' ? 'yes' : 'no';
            $('dash-uid').textContent = u.id;
            const ub = $('user-button-mount');
            ub.innerHTML = '';
            Clerk.mountUserButton(ub, { afterSignOutUrl: window.location.pathname });
            void hydrateFromNeon();
        } else {
            $('view-dashboard').hidden = true;
            $('view-signedout').hidden = false;
            const m = $('clerk-mount');
            m.innerHTML = '';
            Clerk.mountSignIn(m, {
                signUpUrl: window.location.href + '#sign-up',
                appearance: { elements: { rootBox: { width: '100%' } } }
            });
        }
    }

    Clerk.addListener(({ user }) => render());
    render();

    $('form-firm').addEventListener('submit', async (e) => {
        e.preventDefault();
        const msg = $('firm-msg');
        msg.hidden = true;
        msg.className = 'portal-alert';
        const firm = $('firm-input').value.trim();
        try {
            const token = await Clerk.session.getToken();
            const res = await fetch('/api/profile', {
                method: 'PATCH',
                headers: {
                    'Content-Type': 'application/json',
                    Authorization: 'Bearer ' + token
                },
                body: JSON.stringify({ firm })
            });
            if (!res.ok) throw new Error('HTTP ' + res.status + ': ' + (await res.text()));
            msg.textContent = 'Saved.';
            msg.className = 'portal-alert portal-alert--ok';
            msg.hidden = false;
            $('dash-firm').textContent = firm || '—';
        } catch (ex) {
            msg.textContent = 'Failed: ' + (ex.message || ex);
            msg.className = 'portal-alert portal-alert--error';
            msg.hidden = false;
        }
    });
}

async function hydrateFromNeon() {
    try {
        const token = await window.Clerk.session.getToken();
        const [profileRes, dealsRes] = await Promise.all([
            fetch('/api/profile', { headers: { Authorization: 'Bearer ' + token } }),
            fetch('/api/deals',   { headers: { Authorization: 'Bearer ' + token } })
        ]);
        if (profileRes.ok) {
            const p = await profileRes.json();
            $('dash-firm').textContent = p.firm || '—';
            $('firm-input').value = p.firm || '';
            $('dash-subtitle').textContent = 'Signed in and reading from Neon Postgres.';
        } else {
            $('dash-subtitle').textContent = 'Could not load profile: ' + (await profileRes.text());
        }
        const grid = $('dash-deals');
        grid.innerHTML = '';
        if (dealsRes.ok) {
            const deals = await dealsRes.json();
            if (!deals.length) {
                grid.innerHTML = '<p class="portal-muted">No deals seeded yet — see schema.sql.</p>';
                return;
            }
            for (const d of deals) {
                const el = document.createElement('article');
                el.className = 'portal-deal';
                el.innerHTML = `
                    <span class="meta">${escapeHtml(d.stage || '')}</span>
                    <h4>${escapeHtml(d.name)}</h4>
                    <p class="portal-muted" style="margin: 0.25rem 0 0 0;">${escapeHtml(d.headline || '')}</p>
                    <p class="portal-muted" style="margin: 0.5rem 0 0 0; font-size: 0.78rem;">Target close ${d.target_close ? new Date(d.target_close).toLocaleDateString() : '—'}</p>`;
                grid.appendChild(el);
            }
        } else {
            grid.innerHTML = `<p class="portal-alert portal-alert--error">Couldn’t load deals: ${escapeHtml(await dealsRes.text())}</p>`;
        }
    } catch (ex) {
        $('dash-subtitle').textContent = 'Error: ' + (ex.message || ex);
    }
}

/* The Clerk publishable key encodes the frontend API host as base64 after `pk_test_` or `pk_live_`. */
function decodeFrontendApi(pk) {
    const parts = pk.split('_');
    const b64 = parts[parts.length - 1];
    try { return atob(b64).replace(/\$$/, ''); }
    catch (_) { return 'clerk.com'; }
}

function loadScript(src, attrs) {
    return new Promise((resolve, reject) => {
        const s = document.createElement('script');
        s.src = src;
        s.async = true;
        for (const [k, v] of Object.entries(attrs || {})) s.setAttribute(k, v);
        s.onload = () => resolve();
        s.onerror = (e) => reject(new Error('Failed to load ' + src));
        document.head.appendChild(s);
    });
}

function escapeHtml(s) {
    return String(s == null ? '' : s)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
