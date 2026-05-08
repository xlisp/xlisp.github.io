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
  const ARTICLE_TITLE = (document.title || ARTICLE_ID).split('·')[0].trim();

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
  // pull results may use ':comment/text' or 'comment/text' — normalize
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
  function newId() {
    // Fits in safe integer range for the foreseeable future.
    return Date.now() * 1000 + Math.floor(Math.random() * 999);
  }

  let idb = null;
  let conn = null;

  async function loadFromIDB() {
    const all = await idbGetAll(idb);
    const mine = all.filter(c => c.article === ARTICLE_ID);
    if (!mine.length) return;
    // Order parents before children so refs resolve.
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
    ds.transact(conn, tx);
  }

  function listComments() {
    const rows = ds.q(
      '[:find ?e :in $ ?a :where [?e :comment/article ?a]]',
      ds.db(conn), ARTICLE_ID
    );
    const ids = rows.map(r => r[0]);
    return ids.map(id => {
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
    ds.transact(conn, tx);
    await idbPut(idb, item);
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
    let md = `# Notes — ${ARTICLE_TITLE}\n\n_Article: ${ARTICLE_ID}_\n\n`;
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
  function el(html) {
    const t = document.createElement('template');
    t.innerHTML = html.trim();
    return t.content.firstChild;
  }
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

<div class="notes-modal-overlay" id="notes-modal" hidden>
  <div class="notes-modal" role="dialog" aria-labelledby="notes-modal-title">
    <h4 id="notes-modal-title">Add comment</h4>
    <div class="notes-modal-quote" id="notes-modal-quote" hidden></div>
    <textarea id="notes-modal-text" rows="4" placeholder="Your note…"></textarea>
    <div class="notes-modal-actions">
      <button class="notes-btn-secondary" id="notes-modal-cancel" type="button">Cancel</button>
      <button class="notes-btn-primary" id="notes-modal-save" type="button">Save</button>
    </div>
  </div>
</div>
    `);
  }

  // ---------- Rendering ----------
  function renderEmpty() {
    return '<p class="notes-empty">Select text in the article and tap <b>💬 Comment</b> to start an outline note.</p>';
  }
  function renderNode(node) {
    const li = document.createElement('li');
    li.className = 'notes-node';
    li.dataset.id = String(node.id);
    li.innerHTML = `
      ${node.selected ? `<div class="notes-quote" title="Selected from article">"${escapeHtml(node.selected)}"</div>` : ''}
      <div class="notes-text">${escapeHtml(node.text)}</div>
      <div class="notes-actions">
        <button data-act="reply" type="button" title="Reply with a sub-note">↳ Reply</button>
        <button data-act="copy" type="button" title="Copy this note + replies">⎘ Copy</button>
        <button data-act="delete" type="button" title="Delete this note + replies">🗑 Delete</button>
      </div>
    `;
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
    if (!tree.length) {
      container.innerHTML = renderEmpty();
    } else {
      const ul = document.createElement('ul');
      ul.className = 'notes-children notes-root';
      for (const n of tree) ul.appendChild(renderNode(n));
      container.appendChild(ul);
    }
    const countEl = $('notes-count');
    if (all.length) {
      countEl.hidden = false;
      countEl.textContent = String(all.length);
    } else {
      countEl.hidden = true;
    }
  }

  // ---------- Toolbar ----------
  function showToolbar(rect) {
    const tb = $('notes-tb');
    tb.hidden = false;
    // Position above selection, centered horizontally.
    const top = window.scrollY + rect.top - tb.offsetHeight - 8;
    const left = window.scrollX + rect.left + rect.width / 2 - tb.offsetWidth / 2;
    tb.style.top = Math.max(8, top) + 'px';
    tb.style.left = Math.max(8, left) + 'px';
  }
  function hideToolbar() {
    const tb = $('notes-tb');
    if (tb) tb.hidden = true;
  }

  // ---------- Modal ----------
  let modalCallback = null;
  function openModal({ title, quote, onSave }) {
    $('notes-modal-title').textContent = title;
    const qe = $('notes-modal-quote');
    if (quote) {
      qe.hidden = false;
      qe.textContent = '"' + quote + '"';
    } else {
      qe.hidden = true;
      qe.textContent = '';
    }
    $('notes-modal-text').value = '';
    $('notes-modal').hidden = false;
    setTimeout(() => $('notes-modal-text').focus(), 30);
    modalCallback = onSave;
  }
  function closeModal() {
    $('notes-modal').hidden = true;
    modalCallback = null;
  }

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

  // ---------- Toast ----------
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
      // Fallback
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

    // Export
    $('notes-export').addEventListener('click', () => {
      const md = exportMarkdown();
      downloadFile(`notes-${ARTICLE_ID}.md`, md, 'text/markdown;charset=utf-8');
    });
    $('notes-copy-all').addEventListener('click', () => {
      copyText(exportMarkdown());
    });

    // Tree click delegation
    $('notes-tree').addEventListener('click', e => {
      const btn = e.target.closest('button[data-act]');
      if (!btn) return;
      const node = btn.closest('.notes-node');
      if (!node) return;
      const id = Number(node.dataset.id);
      const act = btn.dataset.act;
      if (act === 'reply') {
        openModal({
          title: 'Reply',
          quote: '',
          onSave: text => { if (text && text.trim()) addComment({ text: text.trim(), parentId: id }); }
        });
      } else if (act === 'copy') {
        const tree = buildTree(listComments());
        const n = findInTree(tree, id);
        if (n) copyText(nodeToMarkdown(n, 0).trimEnd());
      } else if (act === 'delete') {
        if (confirm('Delete this note and all its replies?')) deleteComment(id);
      }
    });

    // Modal
    $('notes-modal-cancel').addEventListener('click', closeModal);
    $('notes-modal-save').addEventListener('click', () => {
      const text = $('notes-modal-text').value;
      const cb = modalCallback;
      closeModal();
      if (cb) cb(text);
    });
    $('notes-modal').addEventListener('click', e => {
      if (e.target.id === 'notes-modal') closeModal();
    });
    $('notes-modal-text').addEventListener('keydown', e => {
      if (e.key === 'Escape') closeModal();
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') $('notes-modal-save').click();
    });

    // Selection toolbar
    const article = document.querySelector('article .content');
    if (article) {
      let pendingShow = null;
      document.addEventListener('selectionchange', () => {
        clearTimeout(pendingShow);
        pendingShow = setTimeout(() => {
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
      // Don't lose selection when clicking the toolbar.
      $('notes-tb').addEventListener('mousedown', e => e.preventDefault());
      $('notes-tb-comment').addEventListener('click', () => {
        const sel = window.getSelection();
        const text = sel ? sel.toString().trim() : '';
        if (!text) return;
        hideToolbar();
        openPanel();
        openModal({
          title: 'New comment',
          quote: text,
          onSave: note => { if (note && note.trim()) addComment({ selected: text, text: note.trim() }); }
        });
        if (sel) sel.removeAllRanges();
      });
      window.addEventListener('scroll', hideToolbar, { passive: true });
      window.addEventListener('resize', hideToolbar, { passive: true });
    }

    // Esc closes panel/modal
    document.addEventListener('keydown', e => {
      if (e.key !== 'Escape') return;
      if (!$('notes-modal').hidden) { closeModal(); return; }
      if ($('notes-panel').classList.contains('open')) closePanel();
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
