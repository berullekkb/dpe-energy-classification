// Parité navigateur / Python du simulateur.
// Usage : node tests/js/parity.test.mjs
// Échoue si une probabilité diffère de plus de 1e-9 ou si une classe prédite change.
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
await import(pathToFileURL(join(root, "docs", "assets", "predict.js")).href);

const model = JSON.parse(readFileSync(join(root, "docs", "data", "model.json"), "utf8"));
const parity = JSON.parse(readFileSync(join(root, "docs", "data", "parity.json"), "utf8"));
const { predictProba } = globalThis.DPEPredict;

let maxDiff = 0;
let mismatches = 0;
parity.rows.forEach((row, i) => {
  const proba = predictProba(model, row);
  const js = parity.classes.map((c) => proba[c]);
  const py = parity.expected[i];
  js.forEach((p, j) => { maxDiff = Math.max(maxDiff, Math.abs(p - py[j])); });
  const argmax = (a) => a.indexOf(Math.max(...a));
  if (argmax(js) !== argmax(py)) mismatches += 1;
});

console.log(`${parity.rows.length} logements, écart maximal ${maxDiff.toExponential(2)}, classes différentes : ${mismatches}`);
if (maxDiff > 1e-9 || mismatches > 0) {
  console.error("ÉCHEC : le navigateur ne reproduit pas les prédictions Python.");
  process.exit(1);
}
console.log("OK : prédictions identiques.");

// Les logements réels proposés par le simulateur, convertis comme dans
// l'interface (volume en litres -> log1p), doivent donner la prédiction Python.
const sim = JSON.parse(readFileSync(join(root, "docs", "data", "simulator.json"), "utf8"));
const toModel = (values) => {
  const row = {};
  sim.fields.forEach((f) => {
    const v = values[f.name];
    row[f.name] = f.transform === "log1p" ? Math.log1p(Math.max(0, Number(v) || 0)) : v;
  });
  return row;
};
let templateMismatch = 0;
sim.templates.forEach((t) => {
  const proba = predictProba(model, toModel(t.features));
  const best = Object.keys(proba).reduce((a, b) => (proba[a] >= proba[b] ? a : b));
  if (best !== t.predicted) templateMismatch += 1;
});
console.log(`${sim.templates.length} logements du simulateur, prédictions différentes de Python : ${templateMismatch}`);
if (templateMismatch > 0) {
  console.error("ÉCHEC : la conversion de l'interface modifie les prédictions.");
  process.exit(1);
}
