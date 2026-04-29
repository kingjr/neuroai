/* code-highlight.js — Tiny Pygments-style highlighter for Python and bash.
   Shared by code-selector.js and code-builder.js so the two interactive
   pages render code blocks identically and stay in lock-step.
   Exposes `window.codeHighlight = { python, bash, escapeHtml }`.        */

(function () {
  if (typeof window === "undefined") return;
  if (window.codeHighlight) return;  // idempotent

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }
  function tok(cls, txt) {
    return '<span class="' + cls + '">' + escapeHtml(txt) + '</span>';
  }

  // Truthy keys for tight inner-loop lookup.
  var PY_KEYWORDS = {
    "import":1,"from":1,"as":1,"def":1,"class":1,"if":1,"elif":1,"else":1,
    "for":1,"in":1,"while":1,"try":1,"except":1,"finally":1,"with":1,
    "return":1,"yield":1,"lambda":1,"and":1,"or":1,"not":1,"is":1,
    "None":1,"True":1,"False":1,"break":1,"continue":1,"pass":1,
    "global":1,"nonlocal":1,"self":1,
  };
  var PY_BUILTINS = {
    "print":1,"len":1,"range":1,"enumerate":1,"zip":1,"map":1,"filter":1,
    "sum":1,"max":1,"min":1,"abs":1,"round":1,"type":1,"isinstance":1,
    "hasattr":1,"getattr":1,"setattr":1,"dict":1,"list":1,"tuple":1,
    "set":1,"str":1,"int":1,"float":1,"bool":1,"open":1,"super":1,
  };

  function python(code) {
    var i = 0, n = code.length, out = "";
    while (i < n) {
      var ch = code[i];
      // Comments.
      if (ch === "#") {
        var s = i; while (i < n && code[i] !== "\n") i++;
        out += tok("highlight-comment", code.slice(s, i)); continue;
      }
      // Triple-quoted strings.
      if ((ch === '"' || ch === "'")
          && code[i + 1] === ch && code[i + 2] === ch) {
        var tq = ch + ch + ch, ts = i; i += 3;
        while (i < n && code.slice(i, i + 3) !== tq) i++;
        if (i < n) i += 3;
        out += tok("highlight-string", code.slice(ts, i)); continue;
      }
      // Single-quoted strings (with escape handling).
      if (ch === '"' || ch === "'") {
        var q = ch, ss = i; i++;
        while (i < n) {
          if (code[i] === "\\") { i += 2; continue; }
          if (code[i] === q) { i++; break; }
          i++;
        }
        out += tok("highlight-string", code.slice(ss, i)); continue;
      }
      // String prefixes (f"…", b'…', r"…", etc.).
      if (/[fFrRbBuU]/.test(ch)
          && i + 1 < n && (code[i + 1] === '"' || code[i + 1] === "'")) {
        var pq = code[i + 1], ps = i; i += 2;
        while (i < n) {
          if (code[i] === "\\") { i += 2; continue; }
          if (code[i] === pq) { i++; break; }
          i++;
        }
        out += tok("highlight-string", code.slice(ps, i)); continue;
      }
      // Identifiers, keywords, builtins, function calls.
      if (/[A-Za-z_]/.test(ch)) {
        var ws = i; i++;
        while (i < n && /[A-Za-z0-9_]/.test(code[i])) i++;
        var w = code.slice(ws, i);
        if (PY_KEYWORDS[w]) { out += tok("highlight-keyword", w); continue; }
        if (PY_BUILTINS[w]) { out += tok("highlight-builtin", w); continue; }
        var j = i; while (j < n && /\s/.test(code[j])) j++;
        if (j < n && code[j] === "(") out += tok("highlight-function", w);
        else out += escapeHtml(w);
        continue;
      }
      // Numeric literals.
      if (/[0-9]/.test(ch) || (ch === "." && i + 1 < n && /[0-9]/.test(code[i + 1]))) {
        var ns = i; i++; while (i < n && /[0-9.]/.test(code[i])) i++;
        out += tok("highlight-number", code.slice(ns, i)); continue;
      }
      out += escapeHtml(ch); i++;
    }
    return out;
  }

  function bash(code) {
    var i = 0, n = code.length, out = "";
    while (i < n) {
      var ch = code[i];
      if (ch === "#") {
        var s = i; while (i < n && code[i] !== "\n") i++;
        out += tok("highlight-comment", code.slice(s, i)); continue;
      }
      if (ch === "'" || ch === '"') {
        var q = ch, ss = i; i++;
        while (i < n && code[i] !== q) i++;
        if (i < n) i++;
        out += tok("highlight-string", code.slice(ss, i)); continue;
      }
      out += escapeHtml(ch); i++;
    }
    return out;
  }

  window.codeHighlight = { python: python, bash: bash, escapeHtml: escapeHtml };
})();
