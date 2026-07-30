/*
 * fetch_mp_articles.js
 *
 * Scrape article title / link / read count from the WeChat MP admin page
 * "Publish records" (mp.weixin.qq.com -> content & material -> publish records),
 * auto-clicking through every page.
 *
 * Usage: see README.md in this directory.
 * NOTE: this file is intentionally pure ASCII (Chinese literals are written as
 * \uXXXX escapes) so that copy-pasting it into the console can never break on
 * a mangled character.
 */
(async () => {
  var NEXT_PAGE = '\u4e0b\u4e00\u9875'; // the pager's "next page" label
  var NBSP = /\u00a0/g;              // titles are full of &nbsp;

  var sleep = function (ms) { return new Promise(function (r) { setTimeout(r, ms); }); };
  var clean = function (s) { return (s || '').replace(NBSP, ' ').replace(/\s+/g, ' ').trim(); };
  var num = function (s) { return Number(clean(s).replace(/,/g, '')) || 0; };

  // current page number shown in the pager
  var curPage = function () {
    var el = document.querySelector('.weui-desktop-pagination__num_current');
    return clean(el && el.innerText) || '?';
  };

  // the "next page" link in the pager
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

  // scrape every article card on the current page
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

  var all = [];
  var seen = {};
  var MAX_PAGES = 100;

  for (var p = 0; p < MAX_PAGES; p++) {
    var before = curPage();
    var items = grabPage();
    var added = 0;
    for (var j = 0; j < items.length; j++) {
      if (seen[items[j].url]) continue;   // dedupe by url
      seen[items[j].url] = true;
      all.push(items[j]);
      added++;
    }
    console.log('page ' + before + ': +' + added + ', total ' + all.length);

    var btn = nextBtn();
    if (isDisabled(btn)) { console.log('no more pages'); break; }
    btn.click();

    // the list is rendered async (Vue), so wait for the page number to change
    var waited = 0;
    while (curPage() === before && waited < 10000) { await sleep(200); waited += 200; }
    if (curPage() === before) { console.warn('paging timed out, stopping'); break; }
    await sleep(600); // let the new list finish rendering
  }

  window.__articles = all;
  console.table(all);
  console.log('done: ' + all.length + ' articles, saved to window.__articles');

  // export helpers
  window.__downloadCSV = function () {
    var head = ['time', 'title', 'url', 'read', 'like', 'share', 'rec', 'comment'];
    var esc = function (v) { return '"' + String(v == null ? '' : v).replace(/"/g, '""') + '"'; };
    var rows = all.map(function (r) {
      return head.map(function (h) { return esc(r[h]); }).join(',');
    });
    var csv = '\ufeff' + [head.join(',')].concat(rows).join('\n'); // BOM for Excel
    var a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
    a.download = 'mp_articles.csv';
    a.click();
  };

  window.__downloadJSON = function () {
    var a = document.createElement('a');
    a.href = URL.createObjectURL(
      new Blob([JSON.stringify(all, null, 2)], { type: 'application/json' })
    );
    a.download = 'mp_articles.json';
    a.click();
  };

  window.__markdown = function () {
    return all.map(function (r) {
      return '- [' + r.title + '](' + r.url + ') - ' + r.read;
    }).join('\n');
  };

  console.log('__downloadCSV() / __downloadJSON() / __markdown()');
})();
