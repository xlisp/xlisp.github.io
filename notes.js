/* Per-article outline-comments, stored in DataScript, persisted to IndexedDB.
 * Requires `datascript` to be loaded as a global before this script runs. */
(function () {
  if (typeof datascript === 'undefined') {
    console.warn('[notes] datascript global not found; comments disabled');
    return;
  }
  const ds = datascript;

  // ---------- Constants ----------
  const ARTICLE_ID = (location.pathname.split('/').pop() || 'index')
    .replace(/\.html?$/i, '') || 'index';
  const ARTICLE_TITLE = (() => {
    const h1 = document.querySelector('article .post-header h1');
    if (h1 && h1.textContent.trim()) return h1.textContent.trim();
    return (document.title || ARTICLE_ID).split('·')[0].trim();
  })();

  const IDB_NAME = 'xlisp_notes';
  const IDB_STORE = 'comments';
  const IDB_VERSION = 1;

  const SCHEMA = {
    'comment/parent': { ':db/valueType': ':db.type/ref' }
  };

  // ---------- IndexedDB helpers ----------
  function openIDB() {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open(IDB_NAME, IDB_VERSION);
      req.onupgradeneeded = e => {
        const db = e.target.result;
        if (!db.objectStoreNames.contains(IDB_STORE)) {
          const store = db.createObjectStore(IDB_STORE, { keyPath: 'id' });
          store.createIndex('article', 'article', { unique: false });
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }
  function idbGetAll(idb) {
    return new Promise((resolve, reject) => {
      const tx = idb.transaction(IDB_STORE, 'readonly');
      const req = tx.objectStore(IDB_STORE).getAll();
      req.onsuccess = () => resolve(req.result || []);
      req.onerror = () => reject(req.error);
    });
  }
  function idbPut(idb, item) {
    return new Promise((resolve, reject) => {
      const tx = idb.transaction(IDB_STORE, 'readwrite');
      tx.objectStore(IDB_STORE).put(item);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }
  function idbDel(idb, id) {
    return new Promise((resolve, reject) => {
      const tx = idb.transaction(IDB_STORE, 'readwrite');
      tx.objectStore(IDB_STORE).delete(id);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }

  // ---------- DataScript helpers ----------
  function pv(obj, attr) {
    if (obj == null) return undefined;
    if (obj[':' + attr] !== undefined) return obj[':' + attr];
    return obj[attr];
  }
  function refId(v) {
    if (v == null) return null;
    if (typeof v === 'number') return v;
    return pv(v, 'db/id') ?? null;
  }
  // ID counter — kept small so DataScript's eid validation accepts it.
  // We assign these as :db/id so IDB rows and DataScript eids stay in sync.
  let _idCounter = 0;
  function newId() { return ++_idCounter; }

  let idb = null;
  let conn = null;
  let activeReplyId = null;

  async function loadFromIDB() {
    const all = await idbGetAll(idb);
    // Counter must clear every existing IDB id (across all articles) so we don't
    // collide on idbPut, and every existing DataScript eid for this article.
    if (all.length) {
      _idCounter = Math.max(_idCounter, ...all.map(c => Number(c.id) || 0));
    }
    const mine = all.filter(c => c.article === ARTICLE_ID);
    if (!mine.length) return;
    mine.sort((a, b) => (a.parentId == null ? 0 : 1) - (b.parentId == null ? 0 : 1));
    const tx = mine.map(c => {
      const m = {
        ':db/id': c.id,
        'comment/article': c.article,
        'comment/text': c.text || '',
        'comment/selected-text': c.selected || '',
        'comment/created-at': c.createdAt || 0
      };
      if (c.parentId != null) m['comment/parent'] = c.parentId;
      return m;
    });
    try {
      ds.transact(conn, tx);
    } catch (e) {
      console.error('[notes] transact (load) failed', e, tx);
      throw e;
    }
  }

  function listComments() {
    const rows = ds.q(
      '[:find ?e :in $ ?a :where [?e :comment/article ?a]]',
      ds.db(conn), ARTICLE_ID
    );
    return rows.map(r => r[0]).map(id => {
      const e = ds.pull(ds.db(conn), '[*]', id);
      return {
        id,
        text: pv(e, 'comment/text') || '',
        selected: pv(e, 'comment/selected-text') || '',
        createdAt: pv(e, 'comment/created-at') || 0,
        parentId: refId(pv(e, 'comment/parent'))
      };
    });
  }

  async function addComment({ selected, text, parentId }) {
    const id = newId();
    const createdAt = Date.now();
    const item = {
      id, article: ARTICLE_ID,
      text: text || '',
      selected: selected || '',
      createdAt,
      parentId: parentId || null
    };
    const tx = [{
      ':db/id': id,
      'comment/article': ARTICLE_ID,
      'comment/text': item.text,
      'comment/selected-text': item.selected,
      'comment/created-at': createdAt
    }];
    if (parentId) tx[0]['comment/parent'] = parentId;
    try {
      ds.transact(conn, tx);
    } catch (e) {
      console.error('[notes] transact (add) failed', e, tx);
      // Roll back the local counter so it doesn't drift past actual data.
      if (id === _idCounter) _idCounter -= 1;
      toast('Save failed — see console', true);
      return;
    }
    try {
      await idbPut(idb, item);
    } catch (e) {
      console.error('[notes] idbPut failed', e, item);
      toast('Storage write failed', true);
      return;
    }
    render();
  }

  function collectSubtree(rootId, all) {
    const out = [rootId];
    const stack = [rootId];
    while (stack.length) {
      const cur = stack.pop();
      for (const c of all) {
        if (c.parentId === cur) {
          out.push(c.id);
          stack.push(c.id);
        }
      }
    }
    return out;
  }

  async function deleteComment(id) {
    const all = listComments();
    const ids = collectSubtree(id, all);
    ds.transact(conn, ids.map(i => [':db.fn/retractEntity', i]));
    for (const i of ids) await idbDel(idb, i);
    render();
  }

  // ---------- Tree / export ----------
  function buildTree(all) {
    const byParent = new Map();
    for (const c of all) {
      const p = c.parentId || 0;
      if (!byParent.has(p)) byParent.set(p, []);
      byParent.get(p).push(c);
    }
    function attach(pid) {
      const arr = (byParent.get(pid) || []).slice();
      arr.sort((a, b) => a.createdAt - b.createdAt);
      return arr.map(c => ({ ...c, children: attach(c.id) }));
    }
    return attach(0);
  }

  function nodeToMarkdown(node, depth) {
    const ind = '  '.repeat(depth);
    let md = '';
    if (node.selected) {
      md += `${ind}- > "${node.selected.replace(/\s*\n\s*/g, ' ').trim()}"\n`;
      md += `${ind}  - ${node.text.replace(/\s*\n\s*/g, ' ').trim()}\n`;
    } else {
      md += `${ind}- ${node.text.replace(/\s*\n\s*/g, ' ').trim()}\n`;
    }
    for (const c of node.children || []) md += nodeToMarkdown(c, depth + 1);
    return md;
  }
  function exportMarkdown() {
    const tree = buildTree(listComments());
    let md = `# ${ARTICLE_TITLE}\n\n_Article: ${ARTICLE_ID}_\n\n`;
    if (!tree.length) md += '_(no notes)_\n';
    else for (const n of tree) md += nodeToMarkdown(n, 0);
    return md;
  }
  function findInTree(tree, id) {
    for (const n of tree) {
      if (n.id === id) return n;
      const f = findInTree(n.children || [], id);
      if (f) return f;
    }
    return null;
  }

  // ---------- DOM ----------
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
  }
  function $(id) { return document.getElementById(id); }

  function buildUI() {
    document.body.insertAdjacentHTML('beforeend', `
<button class="notes-fab" id="notes-fab" aria-label="Open notes" title="Outline notes">
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
    <polyline points="14 2 14 8 20 8"/>
    <line x1="9" y1="13" x2="15" y2="13"/>
    <line x1="9" y1="17" x2="13" y2="17"/>
  </svg>
  <span class="notes-fab-label">Notes</span>
  <span class="notes-count" id="notes-count" hidden>0</span>
</button>

<div class="notes-panel" id="notes-panel" role="dialog" aria-hidden="true" aria-labelledby="notes-panel-title">
  <header class="notes-panel-head">
    <div class="notes-panel-title-wrap">
      <span class="notes-icon" aria-hidden="true">📝</span>
      <h3 id="notes-panel-title">Outline notes</h3>
    </div>
    <div class="notes-panel-actions">
      <button class="notes-icon-btn" id="notes-export" title="Export as markdown" aria-label="Export">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
      </button>
      <button class="notes-icon-btn" id="notes-copy-all" title="Copy all as markdown" aria-label="Copy all">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
      </button>
      <button class="notes-icon-btn" id="notes-close" title="Close" aria-label="Close">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
      </button>
    </div>
  </header>
  <div class="notes-tree" id="notes-tree"></div>
</div>

<div class="notes-toolbar" id="notes-tb" hidden>
  <button class="notes-tb-btn" id="notes-tb-comment" type="button">
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
    Comment
  </button>
</div>

<div class="notes-composer" id="notes-composer" hidden>
  <div class="notes-composer-quote" id="notes-composer-quote"></div>
  <textarea id="notes-composer-text" rows="3" placeholder="Your note…"></textarea>
  <div class="notes-composer-actions">
    <button class="notes-btn-secondary" id="notes-composer-cancel" type="button">Cancel</button>
    <button class="notes-btn-primary" id="notes-composer-save" type="button">Save</button>
  </div>
</div>
    `);
  }

  // ---------- Rendering ----------
  function renderNode(node) {
    const li = document.createElement('li');
    li.className = 'notes-node';
    li.dataset.id = String(node.id);

    const inner = document.createElement('div');
    inner.className = 'notes-node-body';
    inner.innerHTML = `
      ${node.selected ? `<div class="notes-quote" title="Selected from article">"${escapeHtml(node.selected)}"</div>` : ''}
      <div class="notes-text">${escapeHtml(node.text)}</div>
      <div class="notes-actions">
        <button data-act="reply" type="button" title="Reply with a sub-note">↳ Reply</button>
        <button data-act="copy" type="button" title="Copy this note + replies">⎘ Copy</button>
        <button data-act="delete" type="button" title="Delete this note + replies">🗑 Delete</button>
      </div>
    `;
    li.appendChild(inner);

    if (activeReplyId === node.id) {
      const composer = document.createElement('div');
      composer.className = 'notes-reply-composer';
      composer.innerHTML = `
        <textarea rows="2" placeholder="Reply…"></textarea>
        <div class="notes-composer-actions">
          <button class="notes-btn-secondary" data-act="reply-cancel" type="button">Cancel</button>
          <button class="notes-btn-primary" data-act="reply-save" type="button">Save</button>
        </div>
      `;
      li.appendChild(composer);
      setTimeout(() => {
        const ta = composer.querySelector('textarea');
        if (ta) ta.focus();
      }, 30);
    }

    if (node.children && node.children.length) {
      const ul = document.createElement('ul');
      ul.className = 'notes-children';
      for (const c of node.children) ul.appendChild(renderNode(c));
      li.appendChild(ul);
    }
    return li;
  }

  function render() {
    const all = listComments();
    const tree = buildTree(all);
    const container = $('notes-tree');
    container.innerHTML = '';

    // Article title is the root of the outline.
    const root = document.createElement('div');
    root.className = 'notes-root-block';
    root.innerHTML = `
      <div class="notes-root-title">
        <span class="notes-root-icon" aria-hidden="true">📄</span>
        <span class="notes-root-text">${escapeHtml(ARTICLE_TITLE)}</span>
      </div>
    `;
    container.appendChild(root);

    if (!tree.length) {
      const hint = document.createElement('p');
      hint.className = 'notes-empty';
      hint.innerHTML = 'Select any text in the article — a <b>💬 Comment</b> button appears below the selection. Each note becomes a child of this title.';
      root.appendChild(hint);
    } else {
      const ul = document.createElement('ul');
      ul.className = 'notes-children notes-root';
      for (const n of tree) ul.appendChild(renderNode(n));
      root.appendChild(ul);
    }

    const countEl = $('notes-count');
    if (all.length) {
      countEl.hidden = false;
      countEl.textContent = String(all.length);
    } else {
      countEl.hidden = true;
    }
  }

  // ---------- Selection toolbar ----------
  function showToolbar(rect) {
    const tb = $('notes-tb');
    tb.hidden = false;
    const top = window.scrollY + rect.top - tb.offsetHeight - 8;
    const left = window.scrollX + rect.left + rect.width / 2 - tb.offsetWidth / 2;
    tb.style.top = Math.max(8, top) + 'px';
    tb.style.left = Math.max(8, left) + 'px';
  }
  function hideToolbar() {
    const tb = $('notes-tb');
    if (tb) tb.hidden = true;
  }

  // ---------- Inline composer (anchored to selection) ----------
  let pendingSelection = null; // { text }

  function positionComposer(rect) {
    const cmp = $('notes-composer');
    // Show first so we can read its size.
    cmp.style.visibility = 'hidden';
    cmp.hidden = false;
    const w = cmp.offsetWidth || 340;
    const h = cmp.offsetHeight || 160;
    const margin = 12;
    const vw = window.innerWidth;
    const vh = window.innerHeight;

    // Prefer below the selection; flip above if not enough room.
    let top = rect.bottom + 10;
    if (top + h > vh - margin && rect.top - 10 - h > margin) {
      top = rect.top - 10 - h;
    }
    let left = rect.left;
    if (left + w > vw - margin) left = vw - margin - w;
    if (left < margin) left = margin;

    cmp.style.top = (window.scrollY + top) + 'px';
    cmp.style.left = (window.scrollX + left) + 'px';
    cmp.style.visibility = '';
  }

  function openComposer(text, rect) {
    pendingSelection = { text };
    $('notes-composer-quote').textContent = '"' + text + '"';
    $('notes-composer-quote').hidden = !text;
    $('notes-composer-text').value = '';
    positionComposer(rect);
    setTimeout(() => $('notes-composer-text').focus(), 30);
  }
  function closeComposer() {
    $('notes-composer').hidden = true;
    pendingSelection = null;
  }
  function composerOpen() { return !$('notes-composer').hidden; }

  // ---------- Panel ----------
  function openPanel() {
    $('notes-panel').classList.add('open');
    $('notes-panel').setAttribute('aria-hidden', 'false');
    $('notes-fab').classList.add('hidden');
  }
  function closePanel() {
    $('notes-panel').classList.remove('open');
    $('notes-panel').setAttribute('aria-hidden', 'true');
    $('notes-fab').classList.remove('hidden');
  }

  // ---------- Toast / clipboard / download ----------
  function toast(msg, bad) {
    const el = document.createElement('div');
    el.className = 'notes-toast' + (bad ? ' bad' : '');
    el.textContent = msg;
    document.body.appendChild(el);
    requestAnimationFrame(() => el.classList.add('show'));
    setTimeout(() => {
      el.classList.remove('show');
      setTimeout(() => el.remove(), 220);
    }, 1500);
  }
  async function copyText(s) {
    try {
      await navigator.clipboard.writeText(s);
      toast('Copied to clipboard');
    } catch (_) {
      const ta = document.createElement('textarea');
      ta.value = s;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      try { document.execCommand('copy'); toast('Copied to clipboard'); }
      catch (_) { toast('Copy failed', true); }
      finally { ta.remove(); }
    }
  }
  function downloadFile(filename, content, mime) {
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1500);
  }

  // ---------- Init ----------
  async function init() {
    buildUI();

    try {
      idb = await openIDB();
    } catch (e) {
      console.error('[notes] IndexedDB unavailable', e);
      $('notes-fab').remove();
      return;
    }

    conn = ds.create_conn(SCHEMA);

    try {
      await loadFromIDB();
    } catch (e) {
      console.error('[notes] load failed', e);
    }
    render();

    // Panel open/close
    $('notes-fab').addEventListener('click', openPanel);
    $('notes-close').addEventListener('click', closePanel);

    // Export / copy-all
    $('notes-export').addEventListener('click', () => {
      downloadFile(`notes-${ARTICLE_ID}.md`, exportMarkdown(), 'text/markdown;charset=utf-8');
    });
    $('notes-copy-all').addEventListener('click', () => copyText(exportMarkdown()));

    // Tree click delegation
    $('notes-tree').addEventListener('click', e => {
      const btn = e.target.closest('button[data-act]');
      if (!btn) return;
      const node = btn.closest('.notes-node');
      if (!node) return;
      const id = Number(node.dataset.id);
      const act = btn.dataset.act;
      if (act === 'reply') {
        activeReplyId = (activeReplyId === id) ? null : id;
        render();
      } else if (act === 'reply-cancel') {
        activeReplyId = null;
        render();
      } else if (act === 'reply-save') {
        const ta = btn.closest('.notes-reply-composer').querySelector('textarea');
        const txt = ta.value.trim();
        activeReplyId = null;
        if (txt) addComment({ text: txt, parentId: id });
        else render();
      } else if (act === 'copy') {
        const tree = buildTree(listComments());
        const n = findInTree(tree, id);
        if (n) copyText(nodeToMarkdown(n, 0).trimEnd());
      } else if (act === 'delete') {
        if (confirm('Delete this note and all its replies?')) deleteComment(id);
      }
    });
    // Cmd/Ctrl+Enter inside reply composer saves
    $('notes-tree').addEventListener('keydown', e => {
      const ta = e.target.closest('.notes-reply-composer textarea');
      if (!ta) return;
      if (e.key === 'Escape') {
        activeReplyId = null;
        render();
      } else if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        const node = ta.closest('.notes-node');
        if (!node) return;
        const id = Number(node.dataset.id);
        const txt = ta.value.trim();
        activeReplyId = null;
        if (txt) addComment({ text: txt, parentId: id });
        else render();
      }
    });

    // Inline composer save/cancel
    $('notes-composer-cancel').addEventListener('click', closeComposer);
    $('notes-composer-save').addEventListener('click', () => {
      if (!pendingSelection) { closeComposer(); return; }
      const text = $('notes-composer-text').value.trim();
      const sel = pendingSelection.text;
      closeComposer();
      if (text) addComment({ selected: sel, text });
    });
    $('notes-composer-text').addEventListener('keydown', e => {
      if (e.key === 'Escape') closeComposer();
      else if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') $('notes-composer-save').click();
    });

    // Selection toolbar wiring
    const article = document.querySelector('article .content');
    if (article) {
      let pendingShow = null;
      document.addEventListener('selectionchange', () => {
        clearTimeout(pendingShow);
        pendingShow = setTimeout(() => {
          // Don't disturb composer-open state.
          if (composerOpen()) return;
          const sel = window.getSelection();
          if (!sel || sel.isCollapsed || !sel.toString().trim()) {
            hideToolbar();
            return;
          }
          const range = sel.getRangeAt(0);
          if (!article.contains(range.commonAncestorContainer)) {
            hideToolbar();
            return;
          }
          const rect = range.getBoundingClientRect();
          if (!rect.width && !rect.height) { hideToolbar(); return; }
          showToolbar(rect);
        }, 30);
      });

      // Don't drop the selection when clicking the toolbar.
      $('notes-tb').addEventListener('mousedown', e => e.preventDefault());
      $('notes-tb-comment').addEventListener('click', () => {
        const sel = window.getSelection();
        const text = sel ? sel.toString().trim() : '';
        if (!text) return;
        const rect = sel.getRangeAt(0).getBoundingClientRect();
        hideToolbar();
        openComposer(text, rect);
      });

      // Click outside the composer (or article) closes it.
      document.addEventListener('mousedown', e => {
        if (!composerOpen()) return;
        if (e.target.closest('#notes-composer')) return;
        // Clicking elsewhere cancels the in-flight composer.
        closeComposer();
      });

      window.addEventListener('scroll', hideToolbar, { passive: true });
      window.addEventListener('resize', () => {
        hideToolbar();
        if (composerOpen()) closeComposer();
      }, { passive: true });
    }

    // Esc closes panel
    document.addEventListener('keydown', e => {
      if (e.key !== 'Escape') return;
      if (composerOpen()) { closeComposer(); return; }
      if ($('notes-panel').classList.contains('open')) closePanel();
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
