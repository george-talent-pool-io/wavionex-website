/* Quantum Echo — full-window WebGL background, lifted from the marketing
   site so the portal pages share the same animated backdrop.

   Requires THREE.js to be loaded before this script. Boots a full-window
   canvas inside #canvas-container if it exists. Respects
   `prefers-reduced-motion`. */

(function () {
    if (typeof THREE === 'undefined') return;
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

    var QUANTUM_ECHO = { freq: 8.0, decay: 0.3, amp: 0.6 };

    function initQuantumEcho(mountEl, opts) {
        opts = opts || {};
        var fullWindow = !!opts.fullWindow;
        var heroMatch = opts.heroMatch === true;
        var techSectionSubtle = opts.techSectionSubtle === true;
        var sectionWireOnly = !fullWindow && opts.embedStyle === 'section-wire' && !heroMatch;
        var lightSurface = !fullWindow && opts.lightSurface === true;
        var pointerRoot = opts.pointerRoot || mountEl;
        var noSeed = !!opts.noSeed;
        var segments = opts.segments != null ? opts.segments : 128;
        var rippleConfig = opts.rippleConfig || QUANTUM_ECHO;
        if (mountEl.querySelector('canvas')) return;

        var scene = new THREE.Scene();
        var camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
        var renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: sectionWireOnly,
            powerPreference: fullWindow ? 'low-power' : 'high-performance'
        });
        renderer.domElement.style.display = 'block';
        renderer.domElement.style.verticalAlign = 'top';
        if (!fullWindow) {
            renderer.domElement.style.touchAction = 'none';
            renderer.domElement.style.width = '100%';
            renderer.domElement.style.height = '100%';
        }
        if (sectionWireOnly) {
            renderer.setClearColor(0x000000, 0);
        }
        mountEl.appendChild(renderer.domElement);
        camera.position.z = 4;

        var geometry = new THREE.PlaneGeometry(15, 10, segments, segments);
        var wireOpacity = fullWindow ? 0.15 : 0.1;
        var solidOpacity = fullWindow ? 0.5 : 0.82;
        if (heroMatch && !fullWindow) {
            if (techSectionSubtle) {
                wireOpacity = 0.085;
                solidOpacity = 0.28;
            } else {
                wireOpacity = 0.15;
                solidOpacity = 0.5;
            }
        } else if (sectionWireOnly) {
            wireOpacity = lightSurface ? 0.16 : 0.13;
            solidOpacity = 0;
        }
        var idleMotion = fullWindow
            ? 0.1
            : techSectionSubtle
                ? 0.055
                : (heroMatch ? 0.1 : 0.025);
        var material = new THREE.MeshPhongMaterial({
            color: 0x6366f1,
            wireframe: true,
            transparent: true,
            opacity: wireOpacity,
            shininess: techSectionSubtle && !fullWindow ? 22 : (fullWindow || heroMatch) ? 100 : 10
        });
        var solidMaterial = new THREE.MeshPhongMaterial({
            color: 0x0f172a,
            side: THREE.DoubleSide,
            flatShading: true,
            transparent: true,
            opacity: solidOpacity
        });
        var plane = new THREE.Mesh(geometry, material);
        var solidPlane = new THREE.Mesh(geometry, solidMaterial);
        scene.add(plane);
        scene.add(solidPlane);
        if (sectionWireOnly) solidPlane.visible = false;

        var useHeroLighting = fullWindow || heroMatch;
        if (techSectionSubtle && !fullWindow) {
            var pl = new THREE.PointLight(0xffffff, 0.38);
            pl.position.set(0.6, -0.4, 5);
            scene.add(pl);
            scene.add(new THREE.AmbientLight(0x3a3a46));
        } else {
            var light = new THREE.PointLight(0xffffff, useHeroLighting ? 1 : 0.35);
            light.position.set(useHeroLighting ? 0 : -3, useHeroLighting ? 0 : 2, useHeroLighting ? 5 : 3);
            scene.add(light);
            scene.add(new THREE.AmbientLight(useHeroLighting ? 0x404040 : 0x2a2a34));
        }

        var ripples = [];
        var MAX_RIPPLES = 15;
        var introPulseStart = null;
        var introPulseDurationMs = 2700;
        var introPulseStepMs = 198;
        var introPulseStep = -1;

        function startIntroPulse() {
            if (!fullWindow || introPulseStart !== null) return;
            introPulseStart = Date.now();
            introPulseStep = -1;
        }

        if (fullWindow) {
            /* The marketing site fires this event after its hero intro
               animation finishes. Portal pages have no such event, so the
               intro pulse just never starts — fine, idle motion is enough. */
            window.addEventListener('wavionex:hero-intro-complete', startIntroPulse, { once: true });
        }

        function createRipple(x, y) {
            ripples.push({ x: x, y: y, age: 0, config: rippleConfig });
            if (ripples.length > MAX_RIPPLES) ripples.shift();
        }

        function clientToFieldXY(clientX, clientY) {
            if (fullWindow) {
                var vw = window.innerWidth;
                var vh = window.innerHeight;
                if (vw < 1 || vh < 1) return null;
                return { x: (clientX / vw) * 12 - 6, y: -(clientY / vh) * 8 + 4 };
            }
            var rect = mountEl.getBoundingClientRect();
            if (rect.width < 1 || rect.height < 1) return null;
            var nx = (clientX - rect.left) / rect.width;
            var ny = (clientY - rect.top) / rect.height;
            return { x: nx * 12 - 6, y: -ny * 8 + 4 };
        }

        function maybeRippleFromClient(clientX, clientY) {
            var p = clientToFieldXY(clientX, clientY);
            if (!p) return;
            if (ripples.length === 0 || Math.hypot(p.x - ripples[ripples.length - 1].x, p.y - ripples[ripples.length - 1].y) > 0.8) {
                createRipple(p.x, p.y);
            }
        }

        function currentPixelRatio() {
            if (fullWindow) {
                var w = window.innerWidth;
                return w < 520 ? 1.35 : Math.min(window.devicePixelRatio, 2);
            }
            var cw = mountEl.clientWidth || 1;
            return cw < 360 ? 1 : cw < 520 ? 1.25 : Math.min(window.devicePixelRatio, 1.75);
        }

        function updateSize() {
            var w, h;
            if (fullWindow) {
                w = window.innerWidth;
                h = window.innerHeight;
            } else {
                var rect = mountEl.getBoundingClientRect();
                w = Math.max(2, Math.floor(rect.width));
                h = Math.max(2, Math.floor(rect.height));
            }
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
            renderer.setPixelRatio(currentPixelRatio());
            renderer.setSize(w, h, true);
        }

        function onPointerMove(e) { maybeRippleFromClient(e.clientX, e.clientY); }

        if (fullWindow) {
            window.addEventListener('mousemove', onPointerMove);
        } else {
            pointerRoot.addEventListener('mousemove', onPointerMove);
            pointerRoot.addEventListener('mouseenter', function (e) { maybeRippleFromClient(e.clientX, e.clientY); });
        }

        var lastTouchRipple = 0;
        function onTouch(clientX, clientY, throttleMs) {
            if (throttleMs) {
                var now = Date.now();
                if (now - lastTouchRipple < throttleMs) return;
                lastTouchRipple = now;
            }
            maybeRippleFromClient(clientX, clientY);
        }

        if (fullWindow) {
            window.addEventListener('touchstart', function (e) {
                if (!e.touches.length) return;
                var t = e.touches[0];
                onTouch(t.clientX, t.clientY, 0);
            }, { passive: true });
            window.addEventListener('touchmove', function (e) {
                if (!e.touches.length) return;
                var t = e.touches[0];
                onTouch(t.clientX, t.clientY, 50);
            }, { passive: true });
        } else {
            pointerRoot.addEventListener('touchstart', function (e) {
                if (!e.touches.length) return;
                var t = e.touches[0];
                onTouch(t.clientX, t.clientY, 0);
            }, { passive: true });
            pointerRoot.addEventListener('touchmove', function (e) {
                if (!e.touches.length) return;
                var t = e.touches[0];
                onTouch(t.clientX, t.clientY, 50);
            }, { passive: true });
        }

        window.addEventListener('resize', updateSize);
        if (!fullWindow && typeof ResizeObserver !== 'undefined') {
            new ResizeObserver(updateSize).observe(mountEl);
        }

        function animate() {
            requestAnimationFrame(animate);
            var now = Date.now();
            var time = now * 0.001;
            if (fullWindow && introPulseStart !== null) {
                var introElapsed = now - introPulseStart;
                if (introElapsed <= introPulseDurationMs) {
                    var nextStep = Math.floor(introElapsed / introPulseStepMs);
                    var totalSteps = Math.max(1, Math.floor(introPulseDurationMs / introPulseStepMs));
                    while (introPulseStep < nextStep) {
                        introPulseStep += 1;
                        var u = introPulseStep / totalSteps;
                        var angle = u * Math.PI * 2.6 + 0.45;
                        var radius = 0.35 + u * 3.9;
                        createRipple(Math.cos(angle) * radius, Math.sin(angle * 1.15) * radius * 0.62);
                    }
                }
            }
            ripples.forEach(function (r) { r.age += 0.12; });
            ripples = ripples.filter(function (r) { return r.age < 15; });

            var pos = geometry.attributes.position.array;
            for (var i = 0; i < pos.length; i += 3) {
                var px = pos[i];
                var py = pos[i + 1];
                var z = 0;
                ripples.forEach(function (r) {
                    var dist = Math.hypot(px - r.x, py - r.y);
                    var wave = Math.sin(dist * r.config.freq - r.age * 5);
                    var decay = Math.exp(-dist * 0.4) * Math.exp(-r.age * r.config.decay);
                    z += wave * r.config.amp * decay;
                });
                z += Math.sin(px * 0.3 + time) * idleMotion + Math.cos(py * 0.3 + time) * idleMotion;
                pos[i + 2] = z;
            }
            geometry.attributes.position.needsUpdate = true;
            geometry.computeVertexNormals();
            renderer.render(scene, camera);
        }

        updateSize();
        if (!fullWindow && !noSeed) {
            createRipple(0, 0);
            createRipple(-2.2, 1.1);
            createRipple(2.4, -1.3);
        }
        /* Without the marketing site's intro event, fire a small startup pulse so
           the portal page doesn't load to a perfectly still field. */
        if (fullWindow) {
            setTimeout(startIntroPulse, 100);
        }
        animate();
    }

    var bg = document.getElementById('canvas-container');
    if (bg) initQuantumEcho(bg, { fullWindow: true });
})();
