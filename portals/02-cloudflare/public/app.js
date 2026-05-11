/* Wavionex Investor Portal — PoC 02 (Cloudflare Access + D1).
   No client-side auth: if the page loaded, CF Access already authenticated.
   The browser fetches profile + deals from same-origin Pages Functions which
   trust the Cf-Access-* headers that the edge injects. */

const $ = (id) => document.getElementById(id);

async function load() {
    try {
        const [profileRes, dealsRes] = await Promise.all([
            fetch('/api/profile'),
            fetch('/api/deals')
        ]);
        if (profileRes.ok) {
            const p = await profileRes.json();
            $('dash-name').textContent = p.full_name || (p.email ? p.email.split('@')[0] : 'investor');
            $('dash-email').textContent = p.email || '—';
            $('dash-firm').textContent = p.firm || '—';
            $('firm-input').value = p.firm || '';
            $('dash-idp').textContent = p.identity_provider || '—';
            $('dash-last').textContent = p.last_login_at ? new Date(p.last_login_at).toLocaleString() : '—';
            $('dash-subtitle').textContent = 'Signed in via Cloudflare Access. Profile served from D1.';
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
                    <span class="meta">${esc(d.stage)}</span>
                    <h4>${esc(d.name)}</h4>
                    <p class="portal-muted" style="margin: 0.25rem 0 0 0;">${esc(d.headline)}</p>
                    <p class="portal-muted" style="margin: 0.5rem 0 0 0; font-size: 0.78rem;">Target close ${d.target_close ? new Date(d.target_close).toLocaleDateString() : '—'}</p>`;
                grid.appendChild(el);
            }
        } else {
            grid.innerHTML = `<p class="portal-alert portal-alert--error">Couldn't load deals: ${esc(await dealsRes.text())}</p>`;
        }
    } catch (ex) {
        $('dash-subtitle').textContent = 'Error: ' + (ex.message || ex);
    }
}

$('form-firm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const msg = $('firm-msg');
    msg.hidden = true;
    msg.className = 'portal-alert';
    try {
        const res = await fetch('/api/profile', {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ firm: $('firm-input').value.trim() })
        });
        if (!res.ok) throw new Error(await res.text());
        const p = await res.json();
        $('dash-firm').textContent = p.firm || '—';
        msg.textContent = 'Saved.';
        msg.className = 'portal-alert portal-alert--ok';
        msg.hidden = false;
    } catch (ex) {
        msg.textContent = 'Failed: ' + (ex.message || ex);
        msg.className = 'portal-alert portal-alert--error';
        msg.hidden = false;
    }
});

function esc(s) {
    return String(s == null ? '' : s)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

load();
