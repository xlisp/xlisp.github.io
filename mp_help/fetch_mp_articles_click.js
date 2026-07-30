/*
 * fetch_mp_articles_click.js  --  click mode (fallback)
 *
 * Use this only if fetch_mp_articles.js (API mode) does not work.
 *
 * Clicking "next page" on this site RELOADS the whole page, which wipes every
 * variable in the tab. So this script keeps its state in localStorage instead:
 *
 *   run -> scrape current page -> merge into localStorage -> wait 5s ->
 *   click "next page" -> page reloads -> you press Cmd+Enter on the snippet
 *   again -> it picks up exactly where it left off.
 *
 * On the last page it stops by itself and downloads mp_articles.json.
 * Nothing is ever held only in memory, so a reload can never lose data.
 *
 * Pure ASCII on purpose (Chinese literals are \uXXXX escapes).
 */
(function () {
  var DELAY = 5000;                    // ms to wait before clicking next page
  var STORE = 'mp_articles_state';     // localStorage key holding the results
  var NEXT_PAGE = '\u4e0b\u4e00\u9875';   // the pager's "next page" label

  var clean = function (s) {
    return (s || '').replace(/\u00a0/g, ' ').replace(/\s+/g, ' ').trim();
  };
  var num = function (s) { return Number(clean(s).replace(/,/g, '')) || 0; };

  var load = function () {
    try { return JSON.parse(localStorage.getItem(STORE)) || []; } catch (e) { return []; }
  };
  var store = function (items) {
    try { localStorage.setItem(STORE, JSON.stringify(items)); } catch (e) {
      console.error('localStorage write failed', e);
    }
  };

  var download = function (name, text, type) {
    var a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([text], { type: type }));
    a.download = name;
    a.click();
  };

  // ---- pager ---------------------------------------------------------------
  var curPage = function () {
    var el = document.querySelector('.weui-desktop-pagination__num_current');
    return clean(el && el.innerText) || '?';
  };

  var lastPage = function () {
    var nums = document.querySelectorAll('.weui-desktop-pagination__num');
    var max = 0;
    for (var i = 0; i < nums.length; i++) {
      var n = num(nums[i].innerText);
      if (n > max) max = n;
    }
    return max;
  };

  var nextBtn = function () {
    var links = document.querySelectorAll('.weui-desktop-pagination a');
    for (var i = 0; i < links.length; i++) {
      if (clean(links[i].innerText) === NEXT_PAGE) return links[i];
    }
    return null;
  };

  var isDisabled = function (el) {
    return !el ||
      el.className.indexOf('disabled') !== -1 ||
      el.getAttribute('aria-disabled') === 'true';
  };

  // ---- scrape the page that is on screen right now -------------------------
  var grabPage = function () {
    var boxes = document.querySelectorAll('.weui-desktop-mass-appmsg');
    var out = [];
    for (var i = 0; i < boxes.length; i++) {
      var box = boxes[i];
      var a = box.querySelector('a.weui-desktop-mass-appmsg__title[href^="http"]');
      if (!a) continue;

      var card = box.closest('.weui-desktop-mass__content');
      var timeEl = card && card.querySelector('.weui-desktop-mass__time');

      // title: only the direct <span> child, so the "original" / "edited"
      // tags that live inside the same <a> do not leak into the title
      var titleEl = a.querySelector(':scope > span');

      var data = function (sel) {
        var el = box.querySelector(sel + ' .weui-desktop-mass-media__data__inner');
        return num(el && el.innerText);
      };

      out.push({
        page: curPage(),
        time: clean(timeEl && timeEl.innerText),
        title: clean(titleEl ? titleEl.innerText : a.innerText),
        url: a.href,
        read: data('.appmsg-view'),
        like: data('.appmsg-like'),
        share: data('.appmsg-share'),
        rec: data('.appmsg-haokan'),
        comment: data('.appmsg-comment')
      });
    }
    return out;
  };

  // ---- exported helpers (redefined on every run, since reloads wipe them) ---
  window.__mpData = load;

  window.__mpSaveJSON = function () {
    download('mp_articles.json', JSON.stringify(load(), null, 2), 'application/json');
  };

  window.__mpSaveCSV = function () {
    var items = load();
    var head = ['time', 'title', 'url', 'read', 'like', 'share', 'rec', 'comment'];
    var esc = function (v) { return '"' + String(v == null ? '' : v).replace(/"/g, '""') + '"'; };
    var rows = items.map(function (r) {
      return head.map(function (h) { return esc(r[h]); }).join(',');
    });
    // \ufeff = BOM, so Excel opens the UTF-8 file without mojibake
    download('mp_articles.csv', '\ufeff' + [head.join(',')].concat(rows).join('\n'),
      'text/csv;charset=utf-8');
  };

  window.__mpReset = function () {
    localStorage.removeItem(STORE);
    console.log('state cleared, next run starts from scratch');
  };

  // ---- merge this page into the stored set ---------------------------------
  var all = load();
  var seen = {};
  for (var i = 0; i < all.length; i++) seen[all[i].url] = true;

  var added = 0;
  var items = grabPage();
  for (var j = 0; j < items.length; j++) {
    if (seen[items[j].url]) continue;    // dedupe by url
    seen[items[j].url] = true;
    all.push(items[j]);
    added++;
  }
  store(all);

  var cur = curPage();
  var last = lastPage();
  console.log('page ' + cur + '/' + (last || '?') + ': +' + added +
    ', stored ' + all.length + ' articles');

  if (!items.length) {
    console.warn('no articles found on this page - is the list still loading?');
    console.warn('wait a second and run the snippet again.');
    return;
  }

  // ---- finished? -----------------------------------------------------------
  var btn = nextBtn();
  if (isDisabled(btn) || (last && num(cur) >= last)) {
    console.log('last page reached, downloading mp_articles.json (' + all.length + ' articles)');
    window.__mpSaveJSON();
    console.log('run __mpSaveCSV() for csv, __mpReset() before scraping again');
    return;
  }

  // ---- otherwise click next after 5s ---------------------------------------
  console.log('clicking "next page" in ' + (DELAY / 1000) + 's ... the page will RELOAD.');
  console.log('after the reload, press Cmd+Enter on this snippet again to continue.');
  console.log('(data so far is safe in localStorage, download anytime: __mpSaveJSON())');
  setTimeout(function () { btn.click(); }, DELAY);
})();
