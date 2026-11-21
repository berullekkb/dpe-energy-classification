/* Site DPE Hauts-de-France : navigation et onglets interactifs.
 * Toutes les données viennent de site/data/*.json, produits par
 * scripts/export_site_data.py à partir des résultats du dépôt. */
(function () {
  "use strict";

  var CLASSES = ["Performant", "Passable", "Energivore"];
  var COLORS = { Performant: "#2f7d4f", Passable: "#e0a526", Energivore: "#c9452d" };
  var PANELS = ["simulateur", "poids", "fuite", "carte", "benchmark"];
  var cache = {};
  var ready = {};

  function $(id) { return document.getElementById(id); }
  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) {
      if (k === "text") node.textContent = attrs[k];
      else if (k === "style") node.style.cssText = attrs[k];
      else if (k.slice(0, 2) === "on") node.addEventListener(k.slice(2), attrs[k]);
      else node.setAttribute(k, attrs[k]);
    });
    (children || []).forEach(function (c) { node.appendChild(typeof c === "string" ? document.createTextNode(c) : c); });
    return node;
  }
  function fmt(x, d) {
    return Number(x).toLocaleString("fr-FR", { minimumFractionDigits: d, maximumFractionDigits: d });
  }
  function pct(x, d) { return fmt(100 * x, d === undefined ? 0 : d) + " %"; }
  function load(name) {
    if (!cache[name]) {
      cache[name] = fetch("data/" + name + ".json").then(function (r) {
        if (!r.ok) throw new Error(name + " : " + r.status);
        return r.json();
      });
    }
    return cache[name];
  }
  function svg(tag, attrs) {
    var node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.keys(attrs).forEach(function (k) { node.setAttribute(k, attrs[k]); });
    return node;
  }
  function fail(target, err) {
    target.textContent = "Données indisponibles (" + err.message + "). Ouvrez le site via un serveur web.";
  }

  /* ---------------------------------------------------------- navigation */
  function show() {
    var id = (location.hash || "#simulateur").slice(1);
    if (PANELS.indexOf(id) < 0) id = "simulateur";
    PANELS.forEach(function (p) { $(p).hidden = p !== id; });
    document.querySelectorAll(".tabs a").forEach(function (a) {
      if (a.getAttribute("href") === "#" + id) a.setAttribute("aria-current", "page");
      else a.removeAttribute("aria-current");
    });
    if (!ready[id]) { ready[id] = true; INIT[id](); }
    if (id === "carte" && drawMap) drawMap();
  }

  /* ---------------------------------------------------------- simulateur */
  function initSimulator() {
    Promise.all([load("simulator"), load("model")]).then(function (res) {
      var sim = res[0], model = res[1];
      var state = { index: 0, values: {}, origin: {} };
      var controls = {};

      sim.fields.forEach(function (f) {
        if (f.group === "fixed") return;
        var id = "f-" + f.name, input;
        if (f.kind === "cat" || f.kind === "ord") {
          input = el("select", { id: id });
          f.options.forEach(function (o) { input.appendChild(el("option", { value: String(o.value), text: o.label })); });
        } else {
          input = el("input", { id: id, type: "number", min: f.min, max: f.max, step: f.step, inputmode: "decimal" });
        }
        input.addEventListener("input", function () {
          var v = input.value;
          state.values[f.name] = f.kind === "cat" ? v : Number(v);
          update();
        });
        var wrap = el("label", { class: "field", for: id }, [f.label, input]);
        controls[f.name] = { input: input, wrap: wrap, field: f };
        $(f.group === "main" ? "sim-main" : "sim-advanced").appendChild(wrap);
      });
      $("sim-fixed").textContent = sim.fixed_note;
      $("sim-model").textContent = "Modèle : " + sim.model_label + " (F1 macro " + fmt(sim.validation_f1_macro, 3) +
        " en validation), calculé dans votre navigateur. Les prédictions sont identiques à celles du modèle Python " +
        "(test de parité sur 500 logements). Le Random Forest retenu pour l'étude est trop volumineux pour un navigateur.";

      function setTemplate(i) {
        state.index = i;
        state.origin = Object.assign({}, sim.templates[i].features);
        state.values = Object.assign({}, state.origin);
        Object.keys(controls).forEach(function (name) {
          var c = controls[name], v = state.values[name];
          if (c.field.kind === "ord" && !c.field.options.some(function (o) { return o.value === v; })) {
            c.input.appendChild(el("option", { value: String(v), text: "Code " + v }));
          }
          c.input.value = String(v);
        });
        update();
      }

      function modelRow() {
        var row = {};
        sim.fields.forEach(function (f) {
          var v = state.values[f.name];
          row[f.name] = f.transform === "log1p" ? Math.log1p(Math.max(0, Number(v) || 0)) : v;
        });
        return row;
      }

      function update() {
        var proba = window.DPEPredict.predictProba(model, modelRow());
        var best = CLASSES.reduce(function (a, b) { return proba[a] >= proba[b] ? a : b; });
        $("sim-pred").textContent = best;
        $("sim-pred").style.color = COLORS[best];
        var bars = $("sim-bars");
        bars.textContent = "";
        CLASSES.forEach(function (c) {
          bars.appendChild(el("div", { class: "bar-row" }, [
            el("span", { text: c }),
            el("div", { class: "track" }, [el("div", { class: "fill", style: "width:" + (100 * proba[c]).toFixed(1) + "%;background:" + COLORS[c] })]),
            el("span", { class: "num", text: pct(proba[c]) })
          ]));
        });
        var changed = 0;
        Object.keys(controls).forEach(function (name) {
          var diff = String(state.values[name]) !== String(state.origin[name]);
          controls[name].wrap.classList.toggle("changed", diff);
          if (diff) changed += 1;
        });
        var t = sim.templates[state.index];
        $("sim-truth").textContent = changed === 0
          ? "Logement réel de validation : son DPE le classe «\u00a0" + t.true + "\u00a0»."
          : changed + " caractéristique" + (changed > 1 ? "s modifiées" : " modifiée") +
            " par rapport au logement réel de départ (classé «\u00a0" + t.true + "\u00a0» par son DPE).";
      }

      $("sim-random").addEventListener("click", function () {
        var i;
        do { i = Math.floor(Math.random() * sim.templates.length); } while (i === state.index && sim.templates.length > 1);
        setTemplate(i);
      });
      $("sim-reset").addEventListener("click", function () { setTemplate(state.index); });
      setTemplate(0);
    }).catch(function (e) { fail($("sim-pred"), e); });
  }

  /* ---------------------------------------------------------- poids */
  function initWeights() {
    load("sweep").then(function (data) {
      var pts = data.points, W = 600, H = 300, L = 44, B = 34, T = 12, R = 12;
      var x = function (w) { return L + (w - 1) / 24 * (W - L - R); };
      var y = function (v) { return T + (1 - (v - 0.6) / 0.35) * (H - T - B); };
      var s = svg("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": "Rappel, précision et F1 macro selon le poids de la classe Performant", style: "width:100%;height:auto" });
      [0.6, 0.7, 0.8, 0.9].forEach(function (v) {
        s.appendChild(svg("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), stroke: "#d5dadf" }));
        var t = svg("text", { x: L - 8, y: y(v) + 4, "text-anchor": "end", "font-size": 12, fill: "#4a5560" });
        t.textContent = fmt(v, 1); s.appendChild(t);
      });
      [1, 5, 10, 15, 20, 25].forEach(function (w) {
        var t = svg("text", { x: x(w), y: H - 12, "text-anchor": "middle", "font-size": 12, fill: "#4a5560" });
        t.textContent = w; s.appendChild(t);
      });
      var lbl = svg("text", { x: (W + L) / 2, y: H, "text-anchor": "middle", "font-size": 12, fill: "#4a5560" });
      lbl.textContent = "Poids de la classe Performant"; s.appendChild(lbl);
      [["recall", COLORS.Performant, ""], ["precision", COLORS.Energivore, "6 4"], ["f1_macro", "#1f2a33", "2 3"]].forEach(function (d) {
        s.appendChild(svg("polyline", { points: pts.map(function (p) { return x(p.w) + "," + y(p[d[0]]); }).join(" "), fill: "none", stroke: d[1], "stroke-width": 2.5, "stroke-dasharray": d[2] }));
      });
      var cursor = svg("line", { y1: T, y2: H - B, stroke: "#1f2a33" });
      var dots = ["recall", "precision", "f1_macro"].map(function (k, i) {
        return svg("circle", { r: 5, fill: [COLORS.Performant, COLORS.Energivore, "#1f2a33"][i] });
      });
      s.appendChild(cursor); dots.forEach(function (d) { s.appendChild(d); });
      $("w-chart").appendChild(s);

      var best = pts.reduce(function (a, b) { return b.f1_macro > a.f1_macro ? b : a; });
      var chosen = pts[data.chosen_weight - 1];
      $("w-note").textContent = "Random Forest réentraîné pour chaque poids, mesures sur la validation (11 130 logements). " +
        "Optuna avait retenu un poids de " + data.chosen_weight + " (F1 macro " + fmt(chosen.f1_macro, 3) + ") ; le maximum du balayage est à " +
        best.w + " (" + fmt(best.f1_macro, 3) + ") : l'écart reste dans la variabilité de mesure.";

      var range = $("w-range");
      range.value = data.chosen_weight;
      function update() {
        var p = pts[Number(range.value) - 1];
        $("w-out").textContent = p.w;
        $("w-rec").textContent = fmt(p.recall, 2);
        $("w-pre").textContent = fmt(p.precision, 2);
        $("w-f1").textContent = fmt(p.f1_macro, 3);
        cursor.setAttribute("x1", x(p.w)); cursor.setAttribute("x2", x(p.w));
        ["recall", "precision", "f1_macro"].forEach(function (k, i) {
          dots[i].setAttribute("cx", x(p.w)); dots[i].setAttribute("cy", y(p[k]));
        });
      }
      range.addEventListener("input", update);
      update();
    }).catch(function (e) { fail($("w-note"), e); });
  }

  /* ---------------------------------------------------------- fuite */
  function initLeak() {
    Promise.all([load("replay"), load("meta")]).then(function (res) {
      var rep = res[0], meta = res[1];
      var state = { model: rep.models.findIndex(function (m) { return m.id === "random_forest"; }), strat: true };
      if (state.model < 0) state.model = 0;
      $("f-intro").textContent = pct(meta.share_rows_duplicated_profiles) + " des " + fmt(meta.n_rows, 0) +
        " logements partagent exactement leurs " + meta.n_features + " caractéristiques avec un autre. La façon de découper les données change le score obtenu.";
      var stats = $("f-stats");
      [["Lignes à profil dupliqué", pct(meta.share_rows_duplicated_profiles)],
       ["Profils déjà vus (découpage stratifié)", pct(rep.share_seen_profiles, 1)],
       ["Profils aux étiquettes contradictoires", fmt(meta.contradictory_profiles, 0)]].forEach(function (s) {
        stats.appendChild(el("div", { class: "stat" }, [el("span", { text: s[0] }), el("strong", { text: s[1] })]));
      });
      var buttons = rep.models.map(function (m, i) {
        var b = el("button", { type: "button", class: "pill", text: m.label, onclick: function () { state.model = i; update(); } });
        $("f-models").appendChild(b);
        return b;
      });
      $("f-strat").addEventListener("click", function () { state.strat = true; update(); });
      $("f-group").addEventListener("click", function () { state.strat = false; update(); });

      function update() {
        var m = rep.models[state.model];
        buttons.forEach(function (b, i) { b.setAttribute("aria-pressed", String(i === state.model)); });
        $("f-strat").setAttribute("aria-pressed", String(state.strat));
        $("f-group").setAttribute("aria-pressed", String(!state.strat));
        var bars = state.strat
          ? [["Profils déjà vus", m.seen, COLORS.Energivore], ["Profils nouveaux", m.unseen, COLORS.Performant], ["Score affiché", m.stratified, "#1f2a33"]]
          : [["Score affiché", m.group, "#1f2a33"]];
        var W = 560, H = 260, base = 220, sc = function (v) { return (v - 0.6) / 0.3 * 180; };
        var s = svg("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": "F1 macro selon le découpage", style: "width:100%;height:auto" });
        s.appendChild(svg("line", { x1: 20, x2: W - 20, y1: base, y2: base, stroke: "#d5dadf" }));
        var slot = (W - 40) / 3;
        bars.forEach(function (b, i) {
          var cx = 20 + slot * (i + 0.5) + (state.strat ? 0 : slot), h = Math.max(2, sc(b[1]));
          s.appendChild(svg("rect", { x: cx - 45, y: base - h, width: 90, height: h, rx: 6, fill: b[2] }));
          var v = svg("text", { x: cx, y: base - h - 8, "text-anchor": "middle", "font-size": 15, "font-weight": 600, fill: "#1f2a33" });
          v.textContent = fmt(b[1], 3); s.appendChild(v);
          var t = svg("text", { x: cx, y: base + 22, "text-anchor": "middle", "font-size": 13, fill: "#4a5560" });
          t.textContent = b[0]; s.appendChild(t);
        });
        $("f-chart").textContent = "";
        $("f-chart").appendChild(s);
        $("f-note").textContent = state.strat
          ? pct(rep.share_seen_profiles, 1) + " des logements évalués ont un jumeau exact dans l'entraînement. Sur eux, le " + m.label +
            " atteint " + fmt(m.seen, 3) + " contre " + fmt(m.unseen, 3) + " sur les profils nouveaux : le score affiché (" + fmt(m.stratified, 3) + ") est gonflé."
          : "Aucun profil n'est partagé entre entraînement et évaluation : " + fmt(m.group, 3) + " mesure la vraie capacité à généraliser (écart de " +
            fmt(m.stratified - m.group, 3) + " avec le découpage stratifié).";
      }
      update();
    }).catch(function (e) { fail($("f-intro"), e); });
  }

  /* ---------------------------------------------------------- carte */
  var drawMap = null;
  function initMap() {
    load("tsne").then(function (data) {
      var canvas = $("tsne-canvas"), tip = $("tsne-tip"), view = "true", screen = [];
      var cols = data.hover_columns;
      var names = { type_batiment: "Bâtiment", periode_construction_enc: "Période", type_energie_principale_chauffage: "Chauffage", qualite_isolation_enveloppe_enc: "Isolation" };
      function label(col, idx) {
        var raw = data.dictionaries[col][idx];
        return (data.value_labels[col] && data.value_labels[col][raw]) || raw;
      }
      drawMap = function () {
        var rect = canvas.getBoundingClientRect();
        if (!rect.width) return;
        var dpr = window.devicePixelRatio || 1;
        canvas.width = rect.width * dpr; canvas.height = rect.height * dpr;
        var ctx = canvas.getContext("2d");
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, rect.width, rect.height);
        var pad = 12, n = 0;
        screen = [];
        var order = [1, 2, 0];
        order.forEach(function (cls) {
          data.points.forEach(function (p, i) {
            var shown = view === "err" ? p[2] !== p[3] : true;
            var c = view === "pred" ? p[3] : p[2];
            if (!shown || c !== cls) return;
            var px = pad + p[0] * (rect.width - 2 * pad), py = pad + p[1] * (rect.height - 2 * pad);
            ctx.beginPath();
            ctx.arc(px, py, view === "err" ? 3.4 : 2.4, 0, 2 * Math.PI);
            ctx.fillStyle = COLORS[data.labels[c]];
            ctx.globalAlpha = 0.78;
            ctx.fill();
            if (view === "err") { ctx.globalAlpha = 1; ctx.strokeStyle = "#1f2a33"; ctx.lineWidth = 0.8; ctx.stroke(); }
            screen.push([px, py, i]);
            n += 1;
          });
        });
        ctx.globalAlpha = 1;
        $("tsne-count").textContent = fmt(n, 0) + " logements affichés" + (view === "err" ? " (erreurs du modèle retenu)" : "");
      };
      document.querySelectorAll("#carte [data-view]").forEach(function (b) {
        b.addEventListener("click", function () {
          view = b.getAttribute("data-view");
          document.querySelectorAll("#carte [data-view]").forEach(function (o) { o.setAttribute("aria-pressed", String(o === b)); });
          drawMap();
        });
      });
      canvas.addEventListener("mousemove", function (e) {
        var r = canvas.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top, best = null, bd = 64;
        screen.forEach(function (s) { var d = (s[0] - mx) * (s[0] - mx) + (s[1] - my) * (s[1] - my); if (d < bd) { bd = d; best = s; } });
        if (!best) { tip.hidden = true; return; }
        var p = data.points[best[2]];
        tip.textContent = "";
        tip.appendChild(el("strong", { text: "Réel : " + data.labels[p[2]] + " · prédit : " + data.labels[p[3]] }));
        cols.forEach(function (c, j) { tip.appendChild(el("div", { text: names[c] + " : " + label(c, p[4 + j]) })); });
        tip.hidden = false;
        var left = Math.min(best[0] + 14, r.width - 270);
        tip.style.left = Math.max(0, left) + "px";
        tip.style.top = Math.max(0, best[1] - 20) + "px";
      });
      canvas.addEventListener("mouseleave", function () { tip.hidden = true; });
      window.addEventListener("resize", function () { if (!$("carte").hidden) drawMap(); });
      drawMap();
    }).catch(function (e) { fail($("tsne-count"), e); });
  }

  /* ---------------------------------------------------------- benchmark */
  function initBench() {
    load("benchmark").then(function (data) {
      var selected = data.models.findIndex(function (m) { return m.id === data.selected; });
      var rows = data.models.map(function (m, i) {
        var width = Math.max(2, (m.f1_macro - 0.6) / 0.2 * 100);
        var name = el("span", {}, [m.label]);
        if (m.origin === "new") name.appendChild(el("span", { class: "tag", text: "nouveau" }));
        if (m.id === data.selected) name.appendChild(el("span", { class: "tag", text: "retenu" }));
        var b = el("button", { type: "button", class: "model-row", onclick: function () { selected = i; update(); } }, [
          name,
          el("div", { class: "track" }, [el("div", { class: "fill", style: "width:" + width + "%;background:#1f2a33" })]),
          el("span", { class: "num", text: fmt(m.f1_macro, 3) })
        ]);
        $("b-list").appendChild(b);
        return b;
      });
      function update() {
        var m = data.models[selected];
        rows.forEach(function (r, i) { r.setAttribute("aria-pressed", String(i === selected)); });
        $("b-title").textContent = m.label;
        var mx = $("b-matrix");
        mx.textContent = "";
        mx.appendChild(el("span"));
        CLASSES.forEach(function (c) { mx.appendChild(el("span", { class: "head", text: "→ " + c })); });
        m.confusion.forEach(function (row, i) {
          var tot = row.reduce(function (a, b) { return a + b; }, 0);
          mx.appendChild(el("span", { class: "muted", style: "align-self:center", text: CLASSES[i] }));
          row.forEach(function (v) {
            var f = v / tot;
            mx.appendChild(el("div", { class: "cell", style: "background:rgba(31,42,51," + (0.06 + 0.85 * f).toFixed(2) + ");color:" + (f > 0.45 ? "#fff" : "#1f2a33") }, [
              fmt(v, 0), el("br"), el("span", { class: "small", text: pct(f) })
            ]));
          });
        });
        var t = $("b-table");
        t.textContent = "";
        t.appendChild(el("tr", {}, ["Classe", "Précision", "Rappel", "F1"].map(function (h) { return el("th", { scope: "col", text: h }); })));
        CLASSES.forEach(function (c) {
          var pc = m.per_class[c];
          t.appendChild(el("tr", {}, [el("td", { text: c }), el("td", { text: fmt(pc.precision, 3) }), el("td", { text: fmt(pc.recall, 3) }), el("td", { text: fmt(pc.f1, 3) })]));
        });
        t.appendChild(el("tr", {}, [el("td", { text: "Global" }), el("td", { text: "acc. " + fmt(m.accuracy, 3) }), el("td", { text: "équil. " + fmt(m.balanced_accuracy, 3) }), el("td", { text: "macro " + fmt(m.f1_macro, 3) })]));
      }
      update();
      var test = data.test;
      [["F1 macro", fmt(test.metrics.f1_macro, 3)], ["Balanced accuracy", fmt(test.metrics.balanced_accuracy, 3)],
       ["Accuracy", fmt(test.metrics.accuracy, 3)], ["Baseline initiale (F1 macro)", fmt(test.baseline_f1_macro, 3)]].forEach(function (s) {
        $("b-test").appendChild(el("div", { class: "stat" }, [el("span", { text: s[0] }), el("strong", { text: s[1] })]));
      });
      $("b-test-note").textContent = test.label + " choisi sur la validation, réentraîné sur entraînement + validation, puis évalué une seule fois sur 10 876 logements jamais utilisés pour choisir. " +
        "Rappel par classe : Performant " + fmt(test.per_class.Performant.recall, 2) + ", Passable " + fmt(test.per_class.Passable.recall, 2) +
        ", Energivore " + fmt(test.per_class.Energivore.recall, 2) + ". La baseline est une régression logistique évaluée sur le même test.";
    }).catch(function (e) { fail($("b-title"), e); });
  }

  var INIT = { simulateur: initSimulator, poids: initWeights, fuite: initLeak, carte: initMap, benchmark: initBench };
  window.addEventListener("hashchange", show);
  show();
})();
