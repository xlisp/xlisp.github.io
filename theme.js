/* Day / night theme toggle.
   The initial theme is applied by a tiny inline <head> script (no flash);
   this file just builds the floating toggle button and handles clicks. */
(function () {
  var KEY = 'theme';
  var root = document.documentElement;

  function current() {
    return root.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
  }
  function apply(theme) {
    if (theme === 'light') root.setAttribute('data-theme', 'light');
    else root.removeAttribute('data-theme');
    try { localStorage.setItem(KEY, theme); } catch (e) {}
    if (btn) {
      var toLight = current() === 'dark';
      btn.setAttribute('aria-label', toLight ? '切换到白天模式' : '切换到夜间模式');
      btn.setAttribute('title', toLight ? '白天模式' : '夜间模式');
      btn.setAttribute('aria-pressed', current() === 'light' ? 'true' : 'false');
    }
  }

  var btn = document.createElement('button');
  btn.className = 'theme-toggle';
  btn.type = 'button';
  // Sun shown in dark mode (click → daytime); moon shown in light mode.
  btn.innerHTML =
    '<svg class="icon-sun" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
      '<circle cx="12" cy="12" r="4"/>' +
      '<path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/>' +
    '</svg>' +
    '<svg class="icon-moon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
      '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>' +
    '</svg>';

  btn.addEventListener('click', function () {
    apply(current() === 'light' ? 'dark' : 'light');
  });

  document.addEventListener('DOMContentLoaded', function () {
    document.body.appendChild(btn);
    apply(current());
  });
  if (document.readyState !== 'loading') {
    document.body.appendChild(btn);
    apply(current());
  }
})();
