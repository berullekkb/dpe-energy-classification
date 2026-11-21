/*
 * Prédiction LightGBM dans le navigateur, sans dépendance.
 *
 * Reproduit le pipeline Python (scripts/export_site_data.py) :
 *   1. encodage : variables qualitatives -> codes de l'encodage cible appris
 *      sur le train (modalité inconnue -> moyenne globale), variables
 *      numériques inchangées ;
 *   2. parcours des arbres : à chaque nœud, valeur <= seuil -> gauche
 *      (les seuils sont stockés une fois dans model.thresholds) ;
 *   3. score brut par classe (arbre i -> classe i % nombre de classes),
 *      puis softmax.
 * tests/js/parity.test.mjs vérifie l'égalité avec Python sur 500 logements.
 */
(function (root) {
  "use strict";

  function encodeRow(model, row) {
    return model.features.map(function (f) {
      var value = row[f.name];
      if (f.kind === "cat") {
        var code = f.map[String(value)];
        return code === undefined ? f.prior : code;
      }
      return Number(value);
    });
  }

  function treeValue(tree, x, thresholds) {
    if (tree.f.length === 0) return tree.v[0];
    var node = 0;
    while (node >= 0) {
      node = x[tree.f[node]] <= thresholds[tree.t[node]] ? tree.l[node] : tree.r[node];
    }
    return tree.v[-node - 1];
  }

  function predictProba(model, row) {
    var x = encodeRow(model, row);
    var k = model.num_class;
    var raw = new Array(k).fill(0);
    for (var i = 0; i < model.trees.length; i++) {
      raw[i % k] += treeValue(model.trees[i], x, model.thresholds);
    }
    var max = Math.max.apply(null, raw);
    var exps = raw.map(function (r) { return Math.exp(r - max); });
    var total = exps.reduce(function (a, b) { return a + b; }, 0);
    var out = {};
    model.classes.forEach(function (c, j) { out[c] = exps[j] / total; });
    return out;
  }

  root.DPEPredict = { encodeRow: encodeRow, predictProba: predictProba };
})(typeof globalThis !== "undefined" ? globalThis : this);
