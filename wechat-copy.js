/*
 * wechat-copy.js — "复制到公众号" / Copy to WeChat
 *
 * Why this exists:
 *   WeChat's official-account rich-text editor (mp.weixin.qq.com) throws away
 *   <style> blocks, class/id attributes and CSS custom properties. It keeps
 *   ONLY inline `style="..."` on each element. So pasting straight from the
 *   rendered page breaks two ways:
 *     1. Code blocks scramble — the Pygments colors live in classes, and the
 *        `white-space: pre` that preserves line breaks lives in style.css.
 *     2. The 米黄色 (cream) background shows up only on the few blocks that
 *        carry their own background; everything else falls back to white,
 *        giving the "片段一片段" patchwork look.
 *
 * What it does:
 *   Clones the article, rewrites every block with inline styles tuned for the
 *   warm/beige (light) theme, inlines each syntax-highlight color, forces
 *   `white-space: pre` on code, and wraps the whole thing in ONE <section>
 *   whose background is continuous cream. Then copies it to the clipboard as
 *   rich text (text/html), so a single ⌘V into WeChat lands clean.
 */
(function () {
  "use strict";

  // ---- Beige ("米黄色" / parchment) palette — matches style.css light theme ----
  var C = {
    paper:    "#f5edda", // page cream
    card:     "#efe5cd", // panels / table head / blockquote
    codeBg:   "#ece1c6", // parchment code block
    ink:      "#353026", // body text
    dim:      "#7c715a", // muted sepia
    rule:     "#e2d6ba", // soft tan border
    green:    "#2f8f57", // accent
    blue:     "#2f6db0", // links / numbers / functions
    codeRed:  "#b3261e", // inline code + keywords
  };

  var MONO = 'Menlo, Consolas, "Courier New", monospace';
  var SANS =
    '-apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", Helvetica, Arial, sans-serif';

  // Map a Pygments token class string -> { color, italic } for the light theme.
  function tokenStyle(cls) {
    if (!cls) return null;
    if (/\b(k|kc|kd|kn|kp|kr|kt|o|ow)\b/.test(cls)) return { color: C.codeRed };
    if (/\b(s|s1|s2|sa|sb|sc|sd|se|sh|si|sx|sr|ss|dl)\b/.test(cls))
      return { color: C.green };
    if (/\b(c|c1|cm|cs|cp|cpf|ch)\b/.test(cls))
      return { color: C.dim, italic: true };
    if (/\b(nf|fm|nc|nn|nd|m|mi|mf|mh|mo|mb|il)\b/.test(cls))
      return { color: C.blue };
    return null; // inherit body ink
  }

  function setStyle(el, css) {
    el.setAttribute("style", css);
  }

  // Make whitespace structural so it survives WeChat's paste sanitizer.
  // WeChat collapses runs of spaces and drops newlines even when the element
  // carries `white-space: pre`. So we turn every "\n" into a real <br> and
  // every space into a non-breaking space ( ). Operates on TEXT NODES
  // only, so inline style attributes on token <span>s are never touched.
  function hardenWhitespace(root) {
    var walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null);
    var nodes = [];
    var t;
    while ((t = walker.nextNode())) nodes.push(t);
    for (var i = 0; i < nodes.length; i++) {
      var tn = nodes[i];
      var text = tn.nodeValue;
      if (text.indexOf("\n") === -1 && text.indexOf(" ") === -1) continue;
      var frag = document.createDocumentFragment();
      var lines = text.split("\n");
      for (var j = 0; j < lines.length; j++) {
        if (j > 0) frag.appendChild(document.createElement("br"));
        if (lines[j].length) {
          frag.appendChild(
            document.createTextNode(lines[j].replace(/ /g, " "))
          );
        }
      }
      tn.parentNode.replaceChild(frag, tn);
    }
  }

  // Convert one Pygments <div class="highlight"><pre>…<code>…</code></pre></div>
  // into a clean, fully-inlined <pre><code> that survives the WeChat paste.
  function buildCodeBlock(highlightEl) {
    // Prefer the inner <code>; fall back to the element itself (a bare <pre>
    // from a fenced block with no language has no <code> child).
    var codeEl = highlightEl.querySelector("code") || highlightEl;
    var src = codeEl.cloneNode(true);

    // Inline every token color, then drop all classes.
    var spans = src.querySelectorAll("*");
    for (var i = 0; i < spans.length; i++) {
      var sp = spans[i];
      var ts = tokenStyle(sp.getAttribute("class"));
      if (ts) {
        setStyle(
          sp,
          "color:" + ts.color + ";" + (ts.italic ? "font-style:italic;" : "")
        );
      } else {
        sp.removeAttribute("style");
      }
      sp.removeAttribute("class");
    }
    // Pygments prepends an empty <span></span>; harmless once classes are gone.

    var pre = document.createElement("pre");
    setStyle(
      pre,
      "background:" + C.codeBg + ";" +
      "color:" + C.ink + ";" +
      "border:1px solid " + C.rule + ";" +
      "border-radius:8px;" +
      "padding:14px 16px;" +
      "margin:16px 0;" +
      "overflow-x:auto;" +
      "font-family:" + MONO + ";" +
      "font-size:14px;" +
      "line-height:1.6;" +
      // The two declarations that actually keep code from collapsing:
      "white-space:pre;" +
      "word-wrap:normal;"
    );

    var code = document.createElement("code");
    setStyle(
      code,
      "background:transparent;color:inherit;font-family:inherit;" +
      "font-size:inherit;white-space:pre;padding:0;"
    );
    // Preserve text + the now-inlined token spans.
    while (src.firstChild) code.appendChild(src.firstChild);
    // Turn newlines/indentation into <br> + non-breaking spaces so WeChat
    // can't collapse them on paste.
    hardenWhitespace(code);
    pre.appendChild(code);
    return pre;
  }

  // ---- Math → image -------------------------------------------------------
  // WeChat strips the CSS/web-fonts MathJax relies on, so rendered formulas
  // scramble on paste. Fix: take each MathJax <svg>, draw it onto a canvas at
  // hi-DPI, and emit a transparent base64 PNG <img> that pastes intact.
  var MATH_SCALE = 3; // raster at 3x for crisp formulas on retina + WeChat zoom

  function rasterizeMath(container) {
    var svg = container.querySelector("svg");
    if (!svg) return Promise.resolve(null);
    var rect = svg.getBoundingClientRect();
    var w = rect.width;
    var h = rect.height;
    if (!w || !h) return Promise.resolve(null);

    var clone = svg.cloneNode(true);
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    clone.setAttribute("xmlns:xlink", "http://www.w3.org/1999/xlink");
    // MathJax paints with `currentColor`; pin it to the WeChat ink color.
    clone.style.color = C.ink;
    // Replace ex-based sizing with explicit pixels so the standalone SVG
    // rasterizes at the size it renders on the page.
    clone.setAttribute("width", w + "px");
    clone.setAttribute("height", h + "px");

    var svgStr = new XMLSerializer().serializeToString(clone);
    var src =
      "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svgStr);

    return new Promise(function (resolve) {
      var img = new Image();
      img.onload = function () {
        try {
          var canvas = document.createElement("canvas");
          canvas.width = Math.ceil(w * MATH_SCALE);
          canvas.height = Math.ceil(h * MATH_SCALE);
          var ctx = canvas.getContext("2d");
          ctx.scale(MATH_SCALE, MATH_SCALE);
          ctx.drawImage(img, 0, 0, w, h);
          resolve({
            url: canvas.toDataURL("image/png"),
            w: w,
            h: h,
            display: container.getAttribute("display") === "true",
            valign: svg.style.verticalAlign || "",
          });
        } catch (e) {
          resolve(null);
        }
      };
      img.onerror = function () { resolve(null); };
      img.src = src;
    });
  }

  function mathImg(res) {
    var img = document.createElement("img");
    img.src = res.url;
    if (res.display) {
      setStyle(
        img,
        "display:block;margin:18px auto;max-width:100%;" +
        "width:" + Math.round(res.w) + "px;height:auto;"
      );
    } else {
      setStyle(
        img,
        "display:inline-block;width:" + Math.round(res.w) + "px;" +
        "height:" + Math.round(res.h) + "px;" +
        "vertical-align:" + (res.valign || "middle") + ";"
      );
    }
    return img;
  }

  // Apply inline styles to a "normal" (non-code) element based on its tag.
  function styleBlock(el) {
    var tag = el.tagName;
    switch (tag) {
      case "H1":
        setStyle(el, "font-size:1.5em;font-weight:700;line-height:1.4;margin:1.4em 0 .6em;color:" + C.ink + ";");
        break;
      case "H2":
        setStyle(el, "font-size:1.3em;font-weight:700;line-height:1.4;margin:1.4em 0 .6em;color:" + C.ink + ";border-bottom:1px solid " + C.rule + ";padding-bottom:.3em;");
        break;
      case "H3":
        setStyle(el, "font-size:1.15em;font-weight:700;line-height:1.4;margin:1.3em 0 .5em;color:" + C.ink + ";");
        break;
      case "H4":
      case "H5":
      case "H6":
        setStyle(el, "font-size:1.05em;font-weight:700;line-height:1.4;margin:1.2em 0 .5em;color:" + C.ink + ";");
        break;
      case "P":
        setStyle(el, "margin:14px 0;color:" + C.ink + ";line-height:1.75;");
        break;
      case "A":
        setStyle(el, "color:" + C.blue + ";text-decoration:none;word-break:break-all;");
        break;
      case "STRONG":
      case "B":
        setStyle(el, "font-weight:700;color:" + C.ink + ";");
        break;
      case "EM":
      case "I":
        setStyle(el, "font-style:italic;");
        break;
      case "BLOCKQUOTE":
        setStyle(el, "border-left:3px solid " + C.green + ";background:" + C.card + ";margin:16px 0;padding:.6em 1em;color:" + C.dim + ";border-radius:0 6px 6px 0;");
        break;
      case "UL":
      case "OL":
        setStyle(el, "padding-left:1.4em;margin:14px 0;color:" + C.ink + ";");
        break;
      case "LI":
        setStyle(el, "margin:6px 0;line-height:1.75;");
        break;
      case "TABLE":
        setStyle(el, "border-collapse:collapse;width:100%;margin:16px 0;font-size:.95em;");
        break;
      case "TH":
        setStyle(el, "border:1px solid " + C.rule + ";padding:6px 10px;text-align:left;background:" + C.card + ";font-weight:700;color:" + C.ink + ";");
        break;
      case "TD":
        setStyle(el, "border:1px solid " + C.rule + ";padding:6px 10px;text-align:left;color:" + C.ink + ";");
        break;
      case "HR":
        setStyle(el, "border:none;border-top:1px solid " + C.rule + ";margin:2em 0;");
        break;
      case "IMG":
        setStyle(el, "max-width:100%;height:auto;border-radius:6px;");
        break;
      case "CODE": // inline code (code blocks are handled separately)
        setStyle(el, "background:" + C.codeBg + ";color:" + C.codeRed + ";font-family:" + MONO + ";font-size:.92em;padding:2px 5px;border-radius:4px;");
        break;
    }
    el.removeAttribute("class");
    el.removeAttribute("id");
  }

  // Build the full WeChat-ready HTML string from the live article content.
  async function buildWeChatHTML() {
    var content = document.querySelector("article .content");
    if (!content) return null;

    // 0) Rasterize math from the LIVE DOM (getBoundingClientRect needs the
    //    elements rendered), then map results into the clone by index.
    var liveMath = Array.prototype.slice.call(
      content.querySelectorAll("mjx-container")
    );
    var mathResults = await Promise.all(liveMath.map(rasterizeMath));

    var work = content.cloneNode(true);

    var cloneMath = work.querySelectorAll("mjx-container");
    var mathImgSet = new Set();
    for (var m = 0; m < cloneMath.length; m++) {
      var res = mathResults[m];
      var node = cloneMath[m];
      if (!res) continue; // leave the original if rasterizing failed
      var picture = mathImg(res);
      mathImgSet.add(picture);
      node.parentNode.replaceChild(picture, node);
    }

    // 1) Replace every code block first (before generic styling touches them).
    var highlights = work.querySelectorAll("div.highlight, pre");
    for (var i = 0; i < highlights.length; i++) {
      var h = highlights[i];
      // Skip a <pre> that's already inside a .highlight we'll handle.
      if (h.tagName === "PRE" && h.closest("div.highlight")) continue;
      var block = buildCodeBlock(h);
      h.parentNode.replaceChild(block, h);
    }

    // 2) Style all remaining elements (skip the code + math images).
    var all = work.querySelectorAll("*");
    for (var j = 0; j < all.length; j++) {
      var el = all[j];
      if (el.tagName === "PRE" || el.closest("pre")) continue;
      if (mathImgSet.has(el)) continue; // keep the sizing we set above
      styleBlock(el);
    }

    // 3) Wrap in ONE continuous cream section.
    var section = document.createElement("section");
    setStyle(
      section,
      "background:" + C.paper + ";" +
      "color:" + C.ink + ";" +
      "font-family:" + SANS + ";" +
      "font-size:16px;line-height:1.75;letter-spacing:.02em;" +
      "padding:20px 18px;" +
      "word-break:break-word;"
    );
    while (work.firstChild) section.appendChild(work.firstChild);

    var box = document.createElement("div");
    box.appendChild(section);
    return box.innerHTML;
  }

  // ---- Clipboard ----------------------------------------------------------
  async function copyRich(html) {
    // Preferred: real rich clipboard write.
    if (navigator.clipboard && window.ClipboardItem) {
      try {
        var item = new ClipboardItem({
          "text/html": new Blob([html], { type: "text/html" }),
          "text/plain": new Blob([stripTags(html)], { type: "text/plain" }),
        });
        await navigator.clipboard.write([item]);
        return true;
      } catch (e) {
        /* fall through */
      }
    }
    // Fallback: select a hidden contenteditable and execCommand('copy').
    try {
      var holder = document.createElement("div");
      holder.contentEditable = "true";
      holder.style.cssText =
        "position:fixed;left:-9999px;top:0;opacity:0;white-space:pre-wrap;";
      holder.innerHTML = html;
      document.body.appendChild(holder);
      var range = document.createRange();
      range.selectNodeContents(holder);
      var sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
      var ok = document.execCommand("copy");
      sel.removeAllRanges();
      document.body.removeChild(holder);
      return ok;
    } catch (e) {
      return false;
    }
  }

  function stripTags(html) {
    var d = document.createElement("div");
    d.innerHTML = html;
    return d.innerText || d.textContent || "";
  }

  // ---- Button -------------------------------------------------------------
  function flash(btn, text, ok) {
    var prev = btn.querySelector(".wx-label").textContent;
    btn.querySelector(".wx-label").textContent = text;
    btn.classList.toggle("wx-ok", ok === true);
    btn.classList.toggle("wx-bad", ok === false);
    setTimeout(function () {
      btn.querySelector(".wx-label").textContent = prev;
      btn.classList.remove("wx-ok", "wx-bad");
    }, 1800);
  }

  function init() {
    if (!document.querySelector("article .content")) return; // posts only
    var nav = document.querySelector(".topnav");

    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "wechat-copy-btn";
    btn.title = "把整篇文章复制为公众号富文本（米黄色背景 + 完整代码格式）";
    btn.innerHTML =
      '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>' +
      '<span class="wx-label">Copy to WeChat</span>';

    btn.addEventListener("click", async function () {
      flash(btn, "处理中…");
      var html = await buildWeChatHTML();
      if (!html) {
        flash(btn, "没有内容", false);
        return;
      }
      var ok = await copyRich(html);
      flash(btn, ok ? "已复制 ✓ 去公众号粘贴" : "复制失败，请重试", ok);
    });

    if (nav) nav.appendChild(btn);
    else {
      btn.classList.add("wechat-copy-floating");
      document.body.appendChild(btn);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
