/*
 * fetch_mp_articles.js  --  API mode (recommended)
 *
 * Scrape title / link / read count of every article from the WeChat MP admin
 * page "publish records" (cgi-bin/appmsgpublish).
 *
 * It never clicks the pager, so the page never reloads and nothing is lost:
 * it re-requests the very URL you are on with a different `begin` offset plus
 * `f=json&ajax=1`, waiting 5s between requests.
 *
 * Every page is checkpointed into localStorage, and the whole result set is
 * downloaded as mp_articles.json when finished.
 *
 * Usage: see README.md. This file is pure ASCII on purpose (Chinese literals
 * are \uXXXX escapes) so a bad copy-paste can never produce a SyntaxError.
 */
(async () => {
  var DELAY = 5000;        // ms between requests
  var COUNT = 10;          // articles per request (the site's own page size)
  var MAX_PAGES = 100;     // safety fuse
  var STORE = 'mp_articles_state';

  var sleep = function (ms) { return new Promise(function (r) { setTimeout(r, ms); }); };
  var clean = function (s) {
    return (s || '').replace(/\u00a0/g, ' ').replace(/\s+/g, ' ').trim();
  };

  var download = function (name, text, type) {
    var a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([text], { type: type }));
    a.download = name;
    a.click();
  };

  var saveJSON = function (items) {
    download('mp_articles.json', JSON.stringify(items, null, 2), 'application/json');
  };

  var saveCSV = function (items) {
    var head = ['time', 'title', 'url', 'read', 'like', 'share', 'comment'];
    var esc = function (v) { return '"' + String(v == null ? '' : v).replace(/"/g, '""') + '"'; };
    var rows = items.map(function (r) {
      return head.map(function (h) { return esc(r[h]); }).join(',');
    });
    // \ufeff = BOM, so Excel opens the UTF-8 file without mojibake
    download('mp_articles.csv', '\ufeff' + [head.join(',')].concat(rows).join('\n'),
      'text/csv;charset=utf-8');
  };

  // ---- build the ajax url out of the url we are standing on -----------------
  if (location.pathname.indexOf('appmsgpublish') === -1) {
    console.warn('you are not on the publish-records page (cgi-bin/appmsgpublish).');
    console.warn('open it first, then run this snippet again.');
    return;
  }

  var ajaxURL = function (begin) {
    var u = new URL(location.href);
    u.searchParams.set('sub', 'list');
    u.searchParams.set('sub_action', 'list_ex');
    u.searchParams.set('begin', String(begin));
    u.searchParams.set('count', String(COUNT));
    u.searchParams.set('f', 'json');
    u.searchParams.set('ajax', '1');
    return u.toString();
  };

  // ---- flatten one publish_list entry into article rows ---------------------
  var loggedKeys = false;

  var rowsFrom = function (entry) {
    var info;
    try {
      info = typeof entry.publish_info === 'string'
        ? JSON.parse(entry.publish_info)
        : (entry.publish_info || entry);
    } catch (e) { return []; }

    var msgs = info.appmsgex || info.appmsg_info || [];
    var sentAt = info.sent_info && info.sent_info.time;

    if (!loggedKeys && msgs.length) {
      loggedKeys = true;
      console.log('fields available per article:', Object.keys(msgs[0]).join(', '));
    }

    return msgs.map(function (m) {
      return {
        time: sentAt ? new Date(sentAt * 1000).toLocaleString('zh-CN') : '',
        title: clean(m.title),
        url: m.link || '',
        read: m.read_num != null ? m.read_num : (m.read_user_num || 0),
        like: m.like_num != null ? m.like_num : (m.old_like_num || 0),
        share: m.share_num != null ? m.share_num : 0,
        comment: m.comment_num != null ? m.comment_num : 0,
        digest: clean(m.digest),
        cover: m.cover || ''
      };
    });
  };

  // ---- main loop ------------------------------------------------------------
  var all = [];
  var seen = {};
  var total = null;

  for (var p = 0; p < MAX_PAGES; p++) {
    var begin = p * COUNT;
    var res, text, json;

    try {
      res = await fetch(ajaxURL(begin), {
        credentials: 'include',
        headers: { 'X-Requested-With': 'XMLHttpRequest' }
      });
      text = await res.text();
      json = JSON.parse(text);
    } catch (e) {
      console.error('request/parse failed at begin=' + begin, e);
      console.error('first 300 chars of the response:', String(text).slice(0, 300));
      break;
    }

    if (json.base_resp && json.base_resp.ret !== 0) {
      console.error('server returned ret=' + json.base_resp.ret + ' (' +
        (json.base_resp.err_msg || '') + '). session expired? re-login and retry.');
      break;
    }

    var page;
    try {
      page = typeof json.publish_page === 'string'
        ? JSON.parse(json.publish_page)
        : json.publish_page;
    } catch (e) {
      console.error('cannot parse publish_page', e);
      break;
    }
    if (!page) {
      console.error('no publish_page in response; top-level keys:', Object.keys(json).join(', '));
      break;
    }

    if (total == null && page.total_count != null) {
      total = page.total_count;
      console.log('total articles reported by the server: ' + total);
    }

    var list = page.publish_list || [];
    var added = 0;
    for (var i = 0; i < list.length; i++) {
      var rows = rowsFrom(list[i]);
      for (var k = 0; k < rows.length; k++) {
        if (!rows[k].url || seen[rows[k].url]) continue;   // dedupe by url
        seen[rows[k].url] = true;
        all.push(rows[k]);
        added++;
      }
    }

    // checkpoint after every page, so a crash never loses what was scraped
    try { localStorage.setItem(STORE, JSON.stringify(all)); } catch (e) {}

    console.log('begin=' + begin + ': +' + added + ', total ' + all.length);

    if (!list.length) { console.log('empty page, done'); break; }
    if (total != null && begin + COUNT >= total) { console.log('reached total, done'); break; }

    await sleep(DELAY);   // be gentle: 5s between requests
  }

  window.__articles = all;
  console.table(all);
  console.log('done: ' + all.length + ' articles');

  window.__saveJSON = function () { saveJSON(all); };
  window.__saveCSV = function () { saveCSV(all); };
  window.__markdown = function () {
    return all.map(function (r) {
      return '- [' + r.title + '](' + r.url + ') - ' + r.read;
    }).join('\n');
  };

  if (all.length) saveJSON(all);   // auto-download
  console.log('re-download with __saveJSON() / __saveCSV(), or __markdown()');
})();
