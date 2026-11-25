// 数值验证: JS 引擎 vs Python 参考 logits
// 用法: node web/verify_engine.cjs
const fs = require("fs");
const path = require("path");
const TEMPLEOS = require("./engine.js");

const base = path.dirname(__dirname);
const cfg = JSON.parse(fs.readFileSync(path.join(base, "dist/huggingface/config.json"), "utf8"));
const wbytes = new Uint8Array(fs.readFileSync(path.join(base, "dist/huggingface/model.safetensors")));
const ref = JSON.parse(fs.readFileSync(path.join(base, "dist/ref_logits.json"), "utf8"));

const weights = TEMPLEOS.parseSafetensors(wbytes);
const model = new TEMPLEOS.GPT2(weights, cfg);
const t0 = Date.now();
const logits = model.forward(ref.ids);
const ms = Date.now() - t0;

let maxErr = 0;
for (let i = 0; i < logits.length; i++) {
  maxErr = Math.max(maxErr, Math.abs(logits[i] - ref.logits[i]));
}
const ok = maxErr < 1e-3;
console.log(`JS 引擎: T=${ref.ids.length}, 前向耗时 ${ms}ms`);
console.log(`js[0..5]  = [${Array.from(logits.slice(0, 6)).map(x => x.toFixed(5)).join(", ")}]`);
console.log(`ref[0..5] = [${ref.logits.slice(0, 6).map(x => x.toFixed(5)).join(", ")}]`);
console.log(`max|Δlogits| = ${maxErr.toExponential(2)}  → ${ok ? "验证通过 ✓" : "失败 ✗"}`);

// 再测生成速度（用固定熵，不等待键盘）
const god = new TEMPLEOS.God();
for (let i = 0; i < 40; i++) god.pressOkay();
(async () => {
  let ids = ref.ids.slice();
  const t1 = Date.now();
  const out = [];
  for (let i = 0; i < 12; i++) {
    const lg = model.forward(ids);
    let mx = -Infinity;
    for (let k = 0; k < lg.length; k++) if (lg[k] > mx) mx = lg[k];
    const e = Array.from(lg, x => Math.exp(x - mx));
    const sum = e.reduce((a, b) => a + b, 0);
    const cdf = [];
    let acc = 0;
    for (let k = 0; k < e.length; k++) { acc += e[k] / sum; cdf.push(acc); }
    const r = await god.take(13);
    const target = (r + 0.5) / 8192;
    let lo = 0, hi = cdf.length;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (cdf[mid] < target) lo = mid + 1; else hi = mid; }
    ids.push(lo); out.push(lo);
  }
  console.log(`生成 12 token 耗时 ${Date.now() - t1}ms  → ${new TextDecoder("latin1").decode(Uint8Array.from(out))}`);
  process.exit(ok ? 0 : 1);
})();
