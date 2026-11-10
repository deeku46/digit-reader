/* shared day/night toggle, remembers the portfolio's choice */
(function () {
  var root = document.documentElement, saved = null;
  try { saved = localStorage.getItem('theme'); } catch (e) {}
  if (saved === 'dark') root.setAttribute('data-theme', 'dark');
  window.addEventListener('DOMContentLoaded', function () {
    var b = document.getElementById('theme'); if (!b) return;
    var sync = function () { b.textContent = root.getAttribute('data-theme') === 'dark' ? 'day' : 'night'; };
    sync();
    b.addEventListener('click', function () {
      var dark = root.getAttribute('data-theme') !== 'dark';
      if (dark) root.setAttribute('data-theme', 'dark'); else root.removeAttribute('data-theme');
      try { localStorage.setItem('theme', dark ? 'dark' : 'light'); } catch (e) {}
      sync(); window.dispatchEvent(new Event('themechange'));
    });
  });
})();
