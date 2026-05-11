/* Wavionex Investor Portal — PoC 05 (Firebase).
   Vanilla browser app talks directly to Firebase Auth + Firestore.
   Data security lives in firestore.rules, not in this file. */

import { initializeApp } from 'https://www.gstatic.com/firebasejs/10.13.2/firebase-app.js';
import {
    getAuth,
    createUserWithEmailAndPassword,
    signInWithEmailAndPassword,
    signOut,
    onAuthStateChanged,
    sendEmailVerification,
    sendPasswordResetEmail,
    updateProfile
} from 'https://www.gstatic.com/firebasejs/10.13.2/firebase-auth.js';
import {
    getFirestore,
    doc,
    setDoc,
    serverTimestamp,
    collection,
    getDocs,
    query,
    orderBy
} from 'https://www.gstatic.com/firebasejs/10.13.2/firebase-firestore.js';

let app = null;
let auth = null;
let db = null;
let configOk = false;

try {
    const cfg = await import('./config.js');
    if (cfg && cfg.FIREBASE_CONFIG && !cfg.FIREBASE_CONFIG.apiKey.includes('YOUR-')) {
        app = initializeApp(cfg.FIREBASE_CONFIG);
        auth = getAuth(app);
        db = getFirestore(app);
        configOk = true;
    }
} catch (_err) {
    /* config.js missing */
}

const $ = (id) => document.getElementById(id);
const views = ['view-signin', 'view-signup', 'view-dashboard'];

function show(v) {
    for (const id of views) $(id).hidden = id !== v;
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function err(el, msg) { el.textContent = msg; el.hidden = !msg; }
function ok(el, msg)  { el.textContent = msg; el.hidden = !msg; }

if (!configOk) {
    $('config-warning').hidden = false;
    show('view-signin');
    document.querySelectorAll('button[type="submit"], #btn-signout, #link-reset, #btn-resend').forEach((b) => (b.disabled = true));
} else {
    bootstrap();
}

function bootstrap() {
    onAuthStateChanged(auth, (user) => {
        if (user) renderDashboard(user);
        else show('view-signin');
    });

    document.querySelectorAll('[data-view]').forEach((btn) => {
        btn.addEventListener('click', () => show('view-' + btn.dataset.view));
    });

    $('form-signin').addEventListener('submit', onSignin);
    $('form-signup').addEventListener('submit', onSignup);
    $('btn-signout').addEventListener('click', () => signOut(auth));
    $('link-reset').addEventListener('click', onReset);
    $('btn-resend').addEventListener('click', onResend);
}

async function onSignin(e) {
    e.preventDefault();
    err($('signin-error'), '');
    const email = $('signin-email').value.trim().toLowerCase();
    const password = $('signin-password').value;
    const btn = e.submitter;
    btn.disabled = true;
    try {
        await signInWithEmailAndPassword(auth, email, password);
    } catch (ex) {
        err($('signin-error'), prettyFirebaseError(ex));
    } finally {
        btn.disabled = false;
    }
}

async function onSignup(e) {
    e.preventDefault();
    err($('signup-error'), '');
    ok($('signup-ok'), '');
    const email = $('signup-email').value.trim().toLowerCase();
    const password = $('signup-password').value;
    const fullName = $('signup-name').value.trim();
    const firm = $('signup-firm').value.trim();
    const btn = e.submitter;
    btn.disabled = true;
    try {
        const cred = await createUserWithEmailAndPassword(auth, email, password);
        await updateProfile(cred.user, { displayName: fullName });
        await setDoc(doc(db, 'profiles', cred.user.uid), {
            email,
            full_name: fullName,
            firm: firm || null,
            created_at: serverTimestamp(),
            updated_at: serverTimestamp()
        });
        await sendEmailVerification(cred.user, { url: window.location.href });
        ok($('signup-ok'), 'Account created — check your inbox for a verification email.');
    } catch (ex) {
        err($('signup-error'), prettyFirebaseError(ex));
    } finally {
        btn.disabled = false;
    }
}

async function onReset() {
    const email = $('signin-email').value.trim().toLowerCase();
    if (!email) {
        err($('signin-error'), 'Type your email above first, then click "Forgot password".');
        return;
    }
    try {
        await sendPasswordResetEmail(auth, email, { url: window.location.href });
        err($('signin-error'), 'Reset email sent. Check your inbox.');
    } catch (ex) {
        err($('signin-error'), prettyFirebaseError(ex));
    }
}

async function onResend() {
    if (!auth.currentUser) return;
    try {
        await sendEmailVerification(auth.currentUser, { url: window.location.href });
        $('dash-subtitle').textContent = 'Verification email re-sent.';
    } catch (ex) {
        $('dash-subtitle').textContent = prettyFirebaseError(ex);
    }
}

async function renderDashboard(user) {
    show('view-dashboard');
    $('dash-name').textContent = user.displayName || user.email.split('@')[0];
    $('dash-email').textContent = user.email;
    $('dash-verified').textContent = user.emailVerified ? 'yes' : 'no';
    $('dash-created').textContent = user.metadata.creationTime
        ? new Date(user.metadata.creationTime).toLocaleDateString() : '—';

    /* Touch the profile doc so it exists even if signup wrote was skipped. */
    try {
        await setDoc(
            doc(db, 'profiles', user.uid),
            { email: user.email, full_name: user.displayName || null, updated_at: serverTimestamp() },
            { merge: true }
        );
    } catch (_) { /* allowed to fail if rules are stricter than expected */ }

    let firm = '—';
    try {
        const snap = await getDocs(query(collection(db, 'profiles')));
        snap.forEach((d) => { if (d.id === user.uid) firm = d.data().firm || '—'; });
    } catch (_) {}
    $('dash-firm').textContent = firm;

    if (!user.emailVerified) {
        $('dash-subtitle').textContent = 'You are signed in. Verify your email to unlock deal data.';
        $('dash-deals').innerHTML = '<p class="portal-muted">Deals are hidden until your email is verified. Use "Resend verification email" below.</p>';
        return;
    }
    $('dash-subtitle').textContent = 'You are signed in. Firestore-gated investor data is loaded below.';

    const grid = $('dash-deals');
    grid.innerHTML = '';
    try {
        const snap = await getDocs(query(collection(db, 'deals'), orderBy('target_close', 'asc')));
        if (snap.empty) {
            grid.innerHTML = '<p class="portal-muted">No deals yet — seed via the Firestore console.</p>';
            return;
        }
        snap.forEach((d) => {
            const v = d.data();
            const el = document.createElement('article');
            el.className = 'portal-deal';
            const close = v.target_close && v.target_close.toDate
                ? v.target_close.toDate().toLocaleDateString()
                : (v.target_close || '—');
            el.innerHTML = `
                <span class="meta">${escapeHtml(v.stage || '')}</span>
                <h4>${escapeHtml(v.name || '')}</h4>
                <p class="portal-muted" style="margin: 0.25rem 0 0 0;">${escapeHtml(v.headline || '')}</p>
                <p class="portal-muted" style="margin: 0.5rem 0 0 0; font-size: 0.78rem;">Target close ${escapeHtml(close)}</p>`;
            grid.appendChild(el);
        });
    } catch (ex) {
        grid.innerHTML = `<p class="portal-alert portal-alert--error">Couldn’t load deals: ${escapeHtml(prettyFirebaseError(ex))}</p>`;
    }
}

function prettyFirebaseError(e) {
    if (!e) return 'Unknown error';
    const code = e.code || '';
    switch (code) {
        case 'auth/email-already-in-use': return 'That email is already registered.';
        case 'auth/invalid-email':        return 'Email looks malformed.';
        case 'auth/weak-password':        return 'Password is too weak — 10+ characters please.';
        case 'auth/invalid-credential':
        case 'auth/wrong-password':
        case 'auth/user-not-found':       return 'Email or password is incorrect.';
        case 'auth/too-many-requests':    return 'Too many attempts — try again later.';
        case 'permission-denied':         return 'Permission denied (check Firestore rules).';
        default: return e.message || String(e);
    }
}

function escapeHtml(s) {
    return String(s == null ? '' : s)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
