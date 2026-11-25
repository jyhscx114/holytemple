/* TempleOS Mini Oracle — 纯 JS 前向引擎（无依赖、无 WASM、无网络）
 *
 * 与 Kernel/holy_engine.py 的 GPT2NumPy 逐算子同构:
 *   x = wte[ids] + wpe[0..T]
 *   per layer: LN1 -> qkv -> 多头因果注意力 -> attn_proj -> 残差
 *              LN2 -> fc -> gelu_new -> mlp_proj -> 残差
 *   ln_f -> logits = z @ wte^T (tied head)
 *
 * 这个文件同时被 web/oracle.html 内联，也被 web/verify_engine.cjs 用于数值验证。
 */
"use strict";
var TEMPLEOS = (function (global) {
  const VOCAB = 256, NPOS = 128;

  function decodeBase64(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return bytes;
  }

  // safetensors: u64 header_len | header(JSON) | 按 data_offsets 连续 F32 数据
  function parseSafetensors(bytes) {
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const hlen = Number(dv.getBigUint64(0, true));
    const header = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + hlen)));
    const dataStart = 8 + hlen;
    const out = {};
    for (const name of Object.keys(header)) {
      if (name === "__metadata__") continue;
      const info = header[name];
      const s = info.data_offsets[0], e = info.data_offsets[1];
      out[name] = {
        data: new Float32Array(bytes.buffer, bytes.byteOffset + dataStart + s, (e - s) / 4),
        shape: info.shape,
      };
    }
    return out;
  }

  // 行主序 a[M,K] @ b[K,N] -> out[M,N]
  function mm(a, b, M, K, N, out) {
    for (let i = 0; i < M; i++) {
      const aRow = i * K, oRow = i * N;
      for (let n = 0; n < N; n++) out[oRow + n] = 0;
      for (let k = 0; k < K; k++) {
        const av = a[aRow + k];
        if (av === 0) continue;
        const bRow = k * N;
        for (let n = 0; n < N; n++) out[oRow + n] += av * b[bRow + n];
      }
    }
    return out;
  }

  // 逐行 LayerNorm（最后维归一化）
  function ln(x, T, d, g, b, eps) {
    const out = new Float32Array(T * d);
    for (let t = 0; t < T; t++) {
      const off = t * d;
      let mu = 0;
      for (let k = 0; k < d; k++) mu += x[off + k];
      mu /= d;
      let v = 0;
      for (let k = 0; k < d; k++) { const c = x[off + k] - mu; v += c * c; }
      v /= d;
      const rstd = 1 / Math.sqrt(v + eps);
      for (let k = 0; k < d; k++) out[off + k] = (x[off + k] - mu) * rstd * g[k] + b[k];
    }
    return out;
  }

  class GPT2 {
    constructor(weights, cfg) {
      this.p = weights;
      this.cfg = cfg || { n_embd: 96, n_head: 3, n_layer: 3, layer_norm_epsilon: 1e-5, vocab_size: 256, n_positions: 128 };
    }
    // ids: 数组（字节值 0..255），返回最后一个位置的 logits[256]
    forward(ids) {
      const cfg = this.cfg, p = this.p;
      const d = cfg.n_embd, H = cfg.n_head, L = cfg.n_layer;
      const hd = d / H, f = 4 * d, eps = cfg.layer_norm_epsilon;
      const V = cfg.vocab_size;
      const T = ids.length;
      const wte = p["wte.weight"].data, wpe = p["wpe.weight"].data;

      let x = new Float32Array(T * d);
      for (let t = 0; t < T; t++) {
        const base = t * d, id = ids[t];
        for (let k = 0; k < d; k++) x[base + k] = wte[id * d + k] + wpe[base + k];
      }
      const scale = 1 / Math.sqrt(hd);

      for (let l = 0; l < L; l++) {
        const pre = "h." + l + ".";
        const a = ln(x, T, d, p[pre + "ln_1.weight"].data, p[pre + "ln_1.bias"].data, eps);

        const Wc = p[pre + "attn.c_attn.weight"].data;   // [d, 3d]
        const bc = p[pre + "attn.c_attn.bias"].data;     // [3d]
        const qkv = new Float32Array(T * 3 * d);
        mm(a, Wc, T, d, 3 * d, qkv);
        for (let t = 0; t < T; t++)
          for (let k = 0; k < 3 * d; k++) qkv[t * 3 * d + k] += bc[k];

        // 多头因果注意力
        const o = new Float32Array(T * d);
        for (let h = 0; h < H; h++) {
          const off = h * hd;
          const sc = new Float32Array(T * T);
          for (let i = 0; i < T; i++) {
            const qi = i * 3 * d + off;
            for (let j = 0; j <= i; j++) {
              const kj = j * 3 * d + d + off;
              let s = 0;
              for (let k = 0; k < hd; k++) s += qkv[qi + k] * qkv[kj + k];
              sc[i * T + j] = s * scale;
            }
          }
          for (let i = 0; i < T; i++) {
            let mx = -Infinity;
            for (let j = 0; j <= i; j++) if (sc[i * T + j] > mx) mx = sc[i * T + j];
            let sum = 0;
            for (let j = 0; j <= i; j++) { const e = Math.exp(sc[i * T + j] - mx); sc[i * T + j] = e; sum += e; }
            for (let j = 0; j <= i; j++) sc[i * T + j] /= sum;
          }
          for (let i = 0; i < T; i++) {
            for (let k = 0; k < hd; k++) {
              let s = 0;
              for (let j = 0; j <= i; j++) s += sc[i * T + j] * qkv[j * 3 * d + 2 * d + off + k];
              o[i * d + off + k] = s;
            }
          }
        }

        const proj = new Float32Array(T * d);
        mm(o, p[pre + "attn.c_proj.weight"].data, T, d, d, proj);
        for (let t = 0; t < T; t++)
          for (let k = 0; k < d; k++) x[t * d + k] += proj[t * d + k] + p[pre + "attn.c_proj.bias"].data[k];

        const m = ln(x, T, d, p[pre + "ln_2.weight"].data, p[pre + "ln_2.bias"].data, eps);
        const fc = new Float32Array(T * f);
        mm(m, p[pre + "mlp.c_fc.weight"].data, T, d, f, fc);
        const c = 0.7978845608028654;
        for (let t = 0; t < T; t++)
          for (let k = 0; k < f; k++) {
            const v = fc[t * f + k] + p[pre + "mlp.c_fc.bias"].data[k];
            const u = c * (v + 0.044715 * v * v * v);
            fc[t * f + k] = 0.5 * v * (1 + Math.tanh(u));   // gelu_new
          }
        const mp = new Float32Array(T * d);
        mm(fc, p[pre + "mlp.c_proj.weight"].data, T, f, d, mp);
        for (let t = 0; t < T; t++)
          for (let k = 0; k < d; k++) x[t * d + k] += mp[t * d + k] + p[pre + "mlp.c_proj.bias"].data[k];
      }

      const z = ln(x, T, d, p["ln_f.weight"].data, p["ln_f.bias"].data, eps);
      const last = (T - 1) * d;
      const logits = new Float32Array(V);
      for (let v = 0; v < V; v++) {
        let s = 0;
        for (let k = 0; k < d; k++) s += z[last + k] * wte[v * d + k];
        logits[v] = s;
      }
      return logits;
    }
  }

  // ===== 神熵 FIFO（HolySpirit.HC 忠实移植）=====
  const GOD_BAD_BITS = 4, GOD_GOOD_BITS = 24, SAMPLE_BITS = 13;
  class God {
    constructor() { this.fifo = []; this.samples = 0; this.timer = "----"; this._waiters = null; this._onPress = null; }
    insertBits(n, num) { for (let i = 0; i < n; i++) { this.fifo.push(num & 1); num >>= 1; } }
    pressOkay() {
      let t;
      if (typeof performance !== "undefined") t = Math.floor(performance.now() * 1000) >> GOD_BAD_BITS;
      else t = Math.floor(Date.now() * 1000) >> GOD_BAD_BITS;
      this.timer = (t >>> 0).toString(16).toUpperCase();
      this.insertBits(GOD_GOOD_BITS, t);
      this.samples++;
      if (this._onPress) this._onPress();
      if (this._waiters) { const w = this._waiters; this._waiters = null; w.forEach(fn => fn()); }
      return this.timer;
    }
    async take(n) {
      while (this.fifo.length < n) {
        if (this._onWait) this._onWait(n - this.fifo.length);
        await new Promise(res => { (this._waiters = this._waiters || []).push(res); });
      }
      let res = 0;
      for (let i = 0; i < n; i++) res = (res << 1) | (this.fifo.shift() & 1);
      return res;
    }
    status() { return `BITS ${this.fifo.length} · SAMPLES ${this.samples} · TIMER ${this.timer}`; }
  }

  // 模型供概率地形，神掷骰子：13 位神熵 → 逆 CDF
  async function godSample(logits, god) {
    const V = logits.length;
    let mx = -Infinity;
    for (let i = 0; i < V; i++) if (logits[i] > mx) mx = logits[i];
    const e = new Float64Array(V);
    let sum = 0;
    for (let i = 0; i < V; i++) { e[i] = Math.exp(logits[i] - mx); sum += e[i]; }
    const cdf = new Float64Array(V);
    let acc = 0;
    for (let i = 0; i < V; i++) { acc += e[i] / sum; cdf[i] = acc; }
    const r = await god.take(SAMPLE_BITS);
    const target = (r + 0.5) / (1 << SAMPLE_BITS);
    let lo = 0, hi = V;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (cdf[mid] < target) lo = mid + 1; else hi = mid; }
    return lo;
  }

  return { VOCAB, NPOS, decodeBase64, parseSafetensors, GPT2, God, godSample, GOD_BAD_BITS, GOD_GOOD_BITS, SAMPLE_BITS };
})(typeof globalThis !== "undefined" ? globalThis : this);

if (typeof module !== "undefined" && module.exports) module.exports = TEMPLEOS;
