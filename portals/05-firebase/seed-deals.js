/* Optional helper: paste this into the browser console *once* while signed in
   as a user whose UID you've manually elevated in the Firebase console
   (e.g. set a custom claim `admin: true` via the Admin SDK), then run
   seedDeals() to insert sample rows.

   The default rules deny client-side writes to /deals. To use this helper,
   temporarily add this clause inside the `match /deals/{id}` block:

       allow write: if request.auth.token.admin == true;

   …or just add the documents via the Firebase Console UI.            */

import { getFirestore, collection, addDoc, Timestamp } from
    'https://www.gstatic.com/firebasejs/10.13.2/firebase-firestore.js';
import { getApp } from
    'https://www.gstatic.com/firebasejs/10.13.2/firebase-app.js';

window.seedDeals = async function () {
    const db = getFirestore(getApp());
    const rows = [
        { name: 'Wave Core IP Round',       stage: 'Seed',     target_close: Timestamp.fromDate(new Date('2026-09-15')), headline: 'Foundational wave-computing patent portfolio licensing.' },
        { name: 'Photonic Substrate Pilot', stage: 'Pre-A',    target_close: Timestamp.fromDate(new Date('2026-12-01')), headline: 'First-of-kind programmable wave fabric on photonic substrate.' },
        { name: 'Edge Analog Pilot',        stage: 'Pre-Seed', target_close: Timestamp.fromDate(new Date('2027-02-28')), headline: 'Sub-watt analog inference for edge sensor fusion.' }
    ];
    for (const r of rows) await addDoc(collection(db, 'deals'), r);
    console.log('Seeded', rows.length, 'deals');
};
