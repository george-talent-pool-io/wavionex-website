/* Shared portal nav — renders the same top bar as the marketing site,
   plus a profile avatar dropdown when the page calls .setUser(...).

   Usage from a portal page:

       import { mountNav } from '../_shared/portal-nav.js';   // adjust path
       const nav = mountNav(document.getElementById('portal-nav-mount'), {
           onSignOut: async () => { await supabase.auth.signOut(); }
       });
       // later, when you know who is signed in:
       nav.setUser({
           email: 'george@wavionex.com',
           fullName: 'George Bilchev',
           firm: 'Wavionex',
           isApproved: true,
           isAdmin: true,
           adminUrl: 'admin/'          // optional override; default: ./admin/
       });
       // sign-out / not-yet-loaded:
       nav.setUser(null);
*/

const SITE = 'https://www.wavionex.com';

const PRIMARY_LINKS = [
    { label: 'Home',       href: SITE + '/#top' },
    { label: 'Technology', href: SITE + '/#technology' },
    { label: 'Research',   href: SITE + '/#research' },
    { label: 'Solutions',  href: SITE + '/#solutions' },
    { label: 'Investors',  href: SITE + '/#investors' }
];

const COMPANY_LINKS = [
    { label: 'About Us', href: SITE + '/#team' },
    { label: 'Contact',  href: SITE + '/#contact' }
];

export function mountNav(container, options = {}) {
    if (!container) throw new Error('mountNav: container is required');
    const { onSignOut } = options;

    /* If the page pre-rendered the nav HTML directly (the recommended path,
       avoids first-paint flicker), skip rendering and just wire up events.
       Otherwise inject the template now. */
    if (!container.querySelector('.wpn')) {
        container.innerHTML = renderNavHtml();
    }
    /* Tells the shared stylesheet that this page is using the portal nav,
       so body padding-top can match the nav height. */
    document.body.classList.add('wpn-active');
    wireDropdowns(container);
    wireMobile(container);
    wireSignout(container, onSignOut);

    return {
        setUser(user) {
            const setText = (sel, text) => {
                const el = container.querySelector(sel);
                if (el) el.textContent = text;
            };
            const setHtml = (sel, html) => {
                const el = container.querySelector(sel);
                if (el) el.innerHTML = html;
            };
            const setAttr = (sel, attr, val) => {
                const el = container.querySelector(sel);
                if (el) {
                    if (attr === 'hidden') { el.hidden = val; return; }
                    el.setAttribute(attr, val);
                }
            };

            const profileOut = container.querySelector('[data-profile-out]');
            const profileIn  = container.querySelector('[data-profile]');
            const sheetSignedIn  = container.querySelector('[data-sheet-signed-in]');
            const sheetSignedOut = container.querySelector('[data-sheet-signed-out]');

            /* Inline style.display beats author CSS specificity, so this is
               guaranteed to show/hide regardless of any conflicting rules. */
            const show = (el) => { if (el) { el.hidden = false; el.style.removeProperty('display'); } };
            const hide = (el) => { if (el) { el.hidden = true;  el.style.display = 'none'; } };

            if (!user) {
                show(profileOut);
                hide(profileIn);
                hide(sheetSignedIn);
                show(sheetSignedOut);
                return;
            }
            hide(profileOut);
            show(profileIn);
            show(sheetSignedIn);
            hide(sheetSignedOut);

            const display = user.fullName || (user.email ? user.email.split('@')[0] : 'Investor');
            const statusPillClass = user.isAdmin
                ? 'wpn-status-pill wpn-status-pill--admin'
                : (user.isApproved ? 'wpn-status-pill wpn-status-pill--approved' : 'wpn-status-pill wpn-status-pill--pending');
            const statusLabel = user.isAdmin ? 'admin' : (user.isApproved ? 'approved' : 'pending');
            const adminUrl = (user.adminUrl !== undefined && user.adminUrl !== null) ? user.adminUrl : 'admin/';

            setText('[data-profile-line1]', display);
            setText('[data-profile-line2]', user.email || '');
            setHtml('[data-profile-line3]',
                (user.firm ? esc(user.firm) + ' &middot; ' : '') +
                `<span class="${statusPillClass}">${statusLabel}</span>`);

            setAttr('[data-admin-link]', 'href',   adminUrl);
            setAttr('[data-admin-link]', 'hidden', !user.isAdmin);

            setText('[data-sheet-profile-line1]', display);
            setText('[data-sheet-profile-line2]', user.email || '');
            setAttr('[data-sheet-admin]', 'href',   adminUrl);
            setAttr('[data-sheet-admin]', 'hidden', !user.isAdmin);
        }
    };
}

function renderNavHtml() {
    const linksHtml = PRIMARY_LINKS.map((l) => `<a class="wpn-link" href="${esc(l.href)}">${esc(l.label)}</a>`).join('');
    const sheetLinksHtml = PRIMARY_LINKS.map((l) => `<a href="${esc(l.href)}">${esc(l.label)}</a>`).join('');
    const sheetCompanyHtml = COMPANY_LINKS.map((l) => `<a href="${esc(l.href)}">${esc(l.label)}</a>`).join('');
    return `
    <nav class="wpn" aria-label="Portal navigation">
        <div class="wpn-inner">
            <a class="wpn-brand" href="${SITE}/" aria-label="Wavionex home">
                <img src="${SITE}/assets/wavionex-wordmark.png"
                     srcset="${SITE}/assets/wavionex-wordmark@2x.png 2x"
                     width="232" height="40" alt="Wavionex" />
                <span class="wpn-tagline">Computation, Reimagined in Waves</span>
            </a>

            <div class="wpn-primary">
                ${linksHtml}

                <div class="wpn-dropdown" data-dropdown>
                    <button type="button" class="wpn-dropdown-trigger" aria-haspopup="true" aria-expanded="false">
                        Company
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
                        </svg>
                    </button>
                    <div class="wpn-menu" role="menu">
                        ${COMPANY_LINKS.map((l) => `<a href="${esc(l.href)}" role="menuitem">${esc(l.label)}</a>`).join('')}
                    </div>
                </div>
            </div>

            <div class="wpn-right">
                <!-- Signed-out: plain link to the portal sign-in + tooltip. -->
                <div class="wpn-profile-link" data-profile-out>
                    <a href="${SITE}/portals/investor/" class="wpn-profile-trigger" aria-label="Investor Portal Login">
                        <span class="wpn-avatar" aria-hidden="true">
                            <svg viewBox="0 0 24 24" aria-hidden="true">
                                <defs>
                                    <linearGradient id="wavionex-profile-gradient" x1="0" y1="0" x2="24" y2="24" gradientUnits="userSpaceOnUse">
                                        <stop offset="0%" stop-color="#22D3EE"/>
                                        <stop offset="50%" stop-color="#818CF8"/>
                                        <stop offset="100%" stop-color="#6366F1"/>
                                    </linearGradient>
                                </defs>
                                <path fill="url(#wavionex-profile-gradient)" d="M12 4a4 4 0 1 0 0 8 4 4 0 0 0 0-8Zm0 10c-4 0-7 2-7 5v1h14v-1c0-3-3-5-7-5Z"/>
                            </svg>
                        </span>
                    </a>
                    <span class="wpn-profile-tooltip" role="tooltip">Investor Portal Login</span>
                </div>

                <!-- Signed-in: button + dropdown. Inline display:none so it
                     stays hidden until setUser flips it on. -->
                <div class="wpn-profile" data-profile data-dropdown hidden style="display:none;">
                    <button type="button" class="wpn-profile-trigger" aria-haspopup="true" aria-expanded="false" aria-label="Account">
                        <span class="wpn-avatar" aria-hidden="true">
                            <svg viewBox="0 0 24 24" aria-hidden="true">
                                <defs>
                                    <linearGradient id="wavionex-profile-gradient" x1="0" y1="0" x2="24" y2="24" gradientUnits="userSpaceOnUse">
                                        <stop offset="0%" stop-color="#22D3EE"/>
                                        <stop offset="50%" stop-color="#818CF8"/>
                                        <stop offset="100%" stop-color="#6366F1"/>
                                    </linearGradient>
                                </defs>
                                <path fill="url(#wavionex-profile-gradient)" d="M12 4a4 4 0 1 0 0 8 4 4 0 0 0 0-8Zm0 10c-4 0-7 2-7 5v1h14v-1c0-3-3-5-7-5Z"/>
                            </svg>
                        </span>
                    </button>
                    <div class="wpn-profile-menu" role="menu">
                        <div class="wpn-profile-header">
                            <div class="wpn-profile-line1" data-profile-line1></div>
                            <div class="wpn-profile-line2" data-profile-line2></div>
                            <div class="wpn-profile-line3" data-profile-line3></div>
                        </div>
                        <a href="admin/" data-admin-link hidden role="menuitem">Admin portal</a>
                        <button type="button" data-signout role="menuitem">Sign out</button>
                    </div>
                </div>

                <button type="button" class="wpn-mobile-btn" data-mobile-toggle aria-expanded="false" aria-controls="wpn-sheet" aria-label="Open menu">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16" />
                    </svg>
                </button>
            </div>
        </div>

        <div id="wpn-sheet" class="wpn-sheet" data-sheet aria-hidden="true" role="dialog" aria-label="Site menu">
            ${sheetLinksHtml}
            <div class="wpn-sheet-sub">
                <strong style="display:block; padding: 0.6rem 0 0.3rem 0; color: var(--brand-text-muted); font-size: 0.78rem; letter-spacing: 0.06em; text-transform: uppercase;">Company</strong>
                ${sheetCompanyHtml}
            </div>
            <div style="margin-top: 1rem;">
                <strong style="display:block; padding: 0.6rem 0 0.3rem 0; color: var(--brand-text-muted); font-size: 0.78rem; letter-spacing: 0.06em; text-transform: uppercase;">Your account</strong>
                <div data-sheet-signed-in hidden style="display:none;">
                    <div style="padding: 0.4rem 0;">
                        <div class="wpn-profile-line1" data-sheet-profile-line1></div>
                        <div class="wpn-profile-line2" data-sheet-profile-line2></div>
                    </div>
                    <a href="admin/" data-sheet-admin hidden>Admin portal</a>
                    <button type="button" data-signout>Sign out</button>
                </div>
                <div data-sheet-signed-out>
                    <a href="${SITE}/portals/investor/">Sign In</a>
                </div>
            </div>
        </div>
    </nav>`;
}

function wireDropdowns(container) {
    const dropdowns = container.querySelectorAll('[data-dropdown]');
    function closeAll(except) {
        dropdowns.forEach((d) => { if (d !== except) d.setAttribute('data-open', 'false'); });
    }
    dropdowns.forEach((d) => {
        const trigger = d.querySelector('.wpn-dropdown-trigger, .wpn-profile-trigger');
        if (!trigger) return;
        trigger.addEventListener('click', (e) => {
            e.stopPropagation();
            const open = d.getAttribute('data-open') === 'true';
            closeAll(d);
            d.setAttribute('data-open', open ? 'false' : 'true');
            trigger.setAttribute('aria-expanded', open ? 'false' : 'true');
        });
    });
    document.addEventListener('click', () => closeAll(null));
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeAll(null);
    });
}

function wireMobile(container) {
    const btn   = container.querySelector('[data-mobile-toggle]');
    const sheet = container.querySelector('[data-sheet]');
    if (!btn || !sheet) return;
    btn.addEventListener('click', () => {
        const open = sheet.getAttribute('data-open') === 'true';
        sheet.setAttribute('data-open', open ? 'false' : 'true');
        sheet.setAttribute('aria-hidden', open ? 'true' : 'false');
        btn.setAttribute('aria-expanded', open ? 'false' : 'true');
    });
    /* Close the sheet on link click so navigation feels immediate. */
    sheet.addEventListener('click', (e) => {
        if (e.target instanceof HTMLAnchorElement || (e.target instanceof HTMLElement && e.target.tagName === 'BUTTON' && !e.target.hasAttribute('data-signout'))) {
            sheet.setAttribute('data-open', 'false');
            sheet.setAttribute('aria-hidden', 'true');
            btn.setAttribute('aria-expanded', 'false');
        }
    });
}

function wireSignout(container, onSignOut) {
    if (typeof onSignOut !== 'function') return;
    container.querySelectorAll('[data-signout]').forEach((btn) => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            onSignOut();
        });
    });
}

function esc(s) {
    return String(s == null ? '' : s)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
