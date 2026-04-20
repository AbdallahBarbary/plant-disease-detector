/* PlantScan AI — main.js */

// ── RESET ALL STATE ON LOAD ───────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', function() {
  ['dzPreview','fileChip','errToast','resultsPanel','btnSpinner'].forEach(function(id) {
    var el = document.getElementById(id);
    if (el) { el.hidden = true; el.classList.remove('visible'); }
  });
  var dzIdle = document.getElementById('dzIdle');
  if (dzIdle) dzIdle.hidden = false;
  var uploadPanel = document.getElementById('uploadPanel');
  if (uploadPanel) uploadPanel.hidden = false;
  var btn = document.getElementById('btnAnalyze');
  if (btn) btn.disabled = true;
  var lbl = document.getElementById('btnLabel');
  if (lbl) lbl.textContent = 'Analyze Plant';
  var img = document.getElementById('previewImg');
  if (img) img.src = '';
  var arr = document.getElementById('btnArrow');
  if (arr) arr.style.display = '';
});

// ── PARTICLES ─────────────────────────────────────────────────────────────
(function() {
  var canvas = document.getElementById('particleCanvas');
  if (!canvas) return;
  var ctx = canvas.getContext('2d');
  var W, H, particles = [];
  function resize() { W = canvas.width = window.innerWidth; H = canvas.height = window.innerHeight; }
  function P() { this.reset = function() { this.x = Math.random()*W; this.y = Math.random()*H; this.r = Math.random()*1.2+.3; this.vx = (Math.random()-.5)*.25; this.vy = (Math.random()-.5)*.25; this.a = Math.random()*.4+.05; }; this.reset(); }
  function init() { resize(); particles = Array.from({length:80}, function() { return new P(); }); }
  function draw() { ctx.clearRect(0,0,W,H); particles.forEach(function(p) { ctx.beginPath(); ctx.arc(p.x,p.y,p.r,0,Math.PI*2); ctx.fillStyle='rgba(110,231,183,'+p.a+')'; ctx.fill(); p.x+=p.vx; p.y+=p.vy; if(p.x<0||p.x>W||p.y<0||p.y>H) p.reset(); }); requestAnimationFrame(draw); }
  window.addEventListener('resize', resize);
  init(); draw();
})();

// ── SPOTLIGHT ─────────────────────────────────────────────────────────────
var spotlight = document.getElementById('spotlight');
if (spotlight) {
  document.addEventListener('mousemove', function(e) {
    spotlight.style.setProperty('--mx', (e.clientX/window.innerWidth*100).toFixed(1)+'%');
    spotlight.style.setProperty('--my', (e.clientY/window.innerHeight*100).toFixed(1)+'%');
  });
}

// ── REFS ──────────────────────────────────────────────────────────────────
var dropZone    = document.getElementById('dropZone');
var fileInput   = document.getElementById('fileInput');
var dzIdle      = document.getElementById('dzIdle');
var dzPreview   = document.getElementById('dzPreview');
var previewImg  = document.getElementById('previewImg');
var btnClear    = document.getElementById('btnClear');
var fileChip    = document.getElementById('fileChip');
var chipName    = document.getElementById('chipName');
var chipSize    = document.getElementById('chipSize');
var btnAnalyze  = document.getElementById('btnAnalyze');
var btnLabel    = document.getElementById('btnLabel');
var btnSpinner  = document.getElementById('btnSpinner');
var btnArrow    = document.getElementById('btnArrow');
var uploadPanel = document.getElementById('uploadPanel');
var resultsPanel= document.getElementById('resultsPanel');
var resultsGrid = document.getElementById('resultsGrid');
var btnAgain    = document.getElementById('btnAgain');
var errToast    = document.getElementById('errToast');
var errMsg      = document.getElementById('errMsg');

var selectedFile = null;

// ── DROP ZONE ─────────────────────────────────────────────────────────────
dropZone.addEventListener('click', function() { fileInput.click(); });
dropZone.addEventListener('keydown', function(e) { if(e.key==='Enter'||e.key===' ') fileInput.click(); });
dropZone.addEventListener('dragover', function(e) { e.preventDefault(); dropZone.classList.add('over'); });
dropZone.addEventListener('dragleave', function() { dropZone.classList.remove('over'); });
dropZone.addEventListener('drop', function(e) { e.preventDefault(); dropZone.classList.remove('over'); if(e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]); });
fileInput.addEventListener('change', function() { if(fileInput.files[0]) handleFile(fileInput.files[0]); });

// ── HANDLE FILE ───────────────────────────────────────────────────────────
function handleFile(file) {
  if (!['image/png','image/jpeg','image/webp'].includes(file.type)) return showError('Please upload a PNG, JPG, or WEBP image.');
  if (file.size > 10*1024*1024) return showError('File too large — max 10 MB.');
  selectedFile = file;
  var reader = new FileReader();
  reader.onload = function(ev) {
    previewImg.src = ev.target.result;
    dzIdle.hidden  = true;
    dzPreview.hidden = false;
    dzPreview.classList.add('visible');
    fileChip.hidden = false;
    fileChip.classList.add('visible');
    chipName.textContent = file.name.length > 34 ? file.name.slice(0,32)+'…' : file.name;
    chipSize.textContent = fmtBytes(file.size);
    btnAnalyze.disabled  = false;
    hideError();
  };
  reader.readAsDataURL(file);
}

function fmtBytes(b) {
  if (b < 1024) return b+' B';
  if (b < 1048576) return (b/1024).toFixed(1)+' KB';
  return (b/1048576).toFixed(1)+' MB';
}

// ── REMOVE ────────────────────────────────────────────────────────────────
btnClear.addEventListener('click', function(e) { e.preventDefault(); e.stopPropagation(); resetUpload(); });

function resetUpload() {
  selectedFile = null; fileInput.value = ''; previewImg.src = '';
  dzPreview.hidden = true; dzPreview.classList.remove('visible');
  dzIdle.hidden = false;
  fileChip.hidden = true; fileChip.classList.remove('visible');
  btnAnalyze.disabled = true;
  setLoading(false);
  hideError();
}

// ── ANALYZE ───────────────────────────────────────────────────────────────
btnAnalyze.addEventListener('click', function() {
  if (!selectedFile) return;
  setLoading(true); hideError();
  var form = new FormData();
  form.append('file', selectedFile);
  fetch('/predict', {method:'POST', body:form})
    .then(function(res) { return res.json().then(function(data) { return {res:res, data:data}; }); })
    .then(function(obj) {
      setLoading(false);
      if (!obj.res.ok || obj.data.error) { showError(obj.data.error || 'Something went wrong.'); return; }
      renderResults(obj.data.predictions);
      uploadPanel.hidden  = true;
      resultsPanel.hidden = false;
      resultsPanel.scrollIntoView({behavior:'smooth', block:'start'});
    })
    .catch(function() { setLoading(false); showError('Network error — make sure the server is running.'); });
});

// ── RESULTS ───────────────────────────────────────────────────────────────
function renderResults(preds) {
  var ranks = ['Top Prediction','2nd Match','3rd Match'];
  var cls   = ['top','second','third'];
  resultsGrid.innerHTML = preds.map(function(p,i) {
    return '<div class="res-card '+cls[i]+'">'+
      '<div class="res-top-row">'+
        '<div class="res-left">'+
          '<div class="res-rank">'+ranks[i]+'</div>'+
          '<div class="res-name">'+p.display+'</div>'+
        '</div>'+
        '<div class="res-pct">'+p.confidence.toFixed(1)+'%</div>'+
      '</div>'+
      '<div class="res-bar-track"><div class="res-bar-fill" style="width:'+Math.min(p.confidence,100)+'%"></div></div>'+
      '<span class="res-badge badge-'+p.severity+'">'+sev(p.severity)+'</span>'+
      '<p class="res-tip">💡 '+p.tip+'</p>'+
    '</div>';
  }).join('');
}

function sev(s) {
  var m = {none:'✓ Healthy', moderate:'⚠ Moderate Risk', high:'✕ High Risk', unknown:'? Unknown'};
  return m[s] || s;
}

// ── SCAN AGAIN ────────────────────────────────────────────────────────────
btnAgain.addEventListener('click', function() {
  resultsPanel.hidden = true;
  uploadPanel.hidden  = false;
  resultsGrid.innerHTML = '';
  resetUpload();
  window.scrollTo({top:0, behavior:'smooth'});
});

// ── UTILS ─────────────────────────────────────────────────────────────────
function setLoading(on) {
  btnAnalyze.disabled  = on;
  btnLabel.textContent = on ? 'Analyzing...' : 'Analyze Plant';
  if (on) { btnSpinner.hidden=false; btnSpinner.classList.add('visible'); btnArrow.style.display='none'; }
  else    { btnSpinner.hidden=true;  btnSpinner.classList.remove('visible'); btnArrow.style.display=''; }
}
function showError(msg) { errMsg.textContent=msg; errToast.hidden=false; errToast.classList.add('visible'); }
function hideError()    { errToast.hidden=true; errToast.classList.remove('visible'); }
