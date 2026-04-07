/* ═══════════════════════════════════════════════════════════
   SfM Pothole Analyzer  –  app.js
   Handles: drag-drop upload, fetch /analyze, progress anim,
            results display, animated canvas background.
   ═══════════════════════════════════════════════════════════ */

'use strict';

// ── Canvas particle background ────────────────────────────────────────────────
(function initCanvas () {
  const canvas = document.getElementById('bg-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  let W, H, particles;

  function resize () {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  function mkParticle () {
    return {
      x : Math.random() * W,
      y : Math.random() * H,
      r : Math.random() * 1.6 + 0.4,
      vx: (Math.random() - 0.5) * 0.35,
      vy: (Math.random() - 0.5) * 0.35,
      a : Math.random(),
    };
  }

  function init () {
    resize();
    particles = Array.from({ length: 160 }, mkParticle);
  }

  function draw () {
    ctx.clearRect(0, 0, W, H);
    particles.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(108,99,255,${p.a * 0.7})`;
      ctx.fill();

      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0 || p.x > W) p.vx *= -1;
      if (p.y < 0 || p.y > H) p.vy *= -1;
    });
    requestAnimationFrame(draw);
  }

  window.addEventListener('resize', resize);
  init();
  draw();
})();

// ── DOM references ────────────────────────────────────────────────────────────
const dropZone      = document.getElementById('drop-zone');
const videoInput    = document.getElementById('video-input');
const dropContent   = document.getElementById('drop-content');
const dropSelected  = document.getElementById('drop-selected');
const selectedName  = document.getElementById('selected-name');
const selectedSize  = document.getElementById('selected-size');
const analyzeBtn    = document.getElementById('analyze-btn');
const progressWrap  = document.getElementById('progress-wrap');
const progressBar   = document.getElementById('progress-bar');
const progressLabel = document.getElementById('progress-label');
const resultsSection= document.getElementById('results-section');
const reanalyzeBtn  = document.getElementById('reanalyze-btn');
const sessionBadge  = document.getElementById('session-badge');

// Pipeline step elements
const STEPS = [1,2,3,4,5,6].map(n => document.getElementById(`step-${n}`));

// Metric elements
const mCount    = document.getElementById('m-count');
const mWidth    = document.getElementById('m-width');
const mDepth    = document.getElementById('m-depth');
const mArea     = document.getElementById('m-area');
const mVolume   = document.getElementById('m-volume');
const mSeverity = document.getElementById('m-severity');
const mCost     = document.getElementById('m-cost');
const mConf     = document.getElementById('m-conf');

// Image/download elements
const imgDetection = document.getElementById('img-detection');
const imgMatches   = document.getElementById('img-matches');
const imgDepth     = document.getElementById('img-depth');
const imgCloud     = document.getElementById('img-cloud');
const dlDetection  = document.getElementById('dl-detection');
const dlMatches    = document.getElementById('dl-matches');
const dlDepth      = document.getElementById('dl-depth');
const dlCloud      = document.getElementById('dl-cloud');

// ── State ─────────────────────────────────────────────────────────────────────
let selectedFile = null;

// ── Drag & drop ───────────────────────────────────────────────────────────────
dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
['dragleave', 'dragend'].forEach(ev =>
  dropZone.addEventListener(ev, () => dropZone.classList.remove('drag-over'))
);
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('video/')) setFile(file);
  else showToast('Please drop a video file.');
});

// Click-to-browse
videoInput.addEventListener('change', () => {
  if (videoInput.files[0]) setFile(videoInput.files[0]);
});

dropZone.addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ' ') videoInput.click();
});

function setFile (file) {
  selectedFile = file;
  dropContent.classList.add('hidden');
  dropSelected.classList.remove('hidden');
  selectedName.textContent = file.name;
  selectedSize.textContent = formatBytes(file.size);
  analyzeBtn.disabled = false;
}

function formatBytes (bytes) {
  if (bytes < 1024)       return bytes + ' B';
  if (bytes < 1048576)    return (bytes / 1024).toFixed(1)    + ' KB';
  return (bytes / 1048576).toFixed(1) + ' MB';
}

// ── Analyze ───────────────────────────────────────────────────────────────────
analyzeBtn.addEventListener('click', runAnalysis);

async function runAnalysis () {
  if (!selectedFile) return;

  // UI → loading state
  analyzeBtn.classList.add('loading');
  analyzeBtn.querySelector('.btn-text').textContent = 'Analyzing…';
  analyzeBtn.disabled = true;
  progressWrap.classList.remove('hidden');
  resultsSection.classList.add('hidden');
  resetSteps();

  const form = new FormData();
  form.append('video', selectedFile);

  // Simulated progress stages
  const stages = [
    { pct: 12, label: '🎞  Extracting frames…',                 step: 0 },
    { pct: 28, label: '🧠  Running YOLOv8n detection…',          step: 1 },
    { pct: 48, label: '📍  SIFT keypoint matching…',             step: 2 },
    { pct: 65, label: '📐  Computing Essential Matrix (SfM)…',   step: 3 },
    { pct: 80, label: '📦  Triangulating 3D point cloud…',       step: 4 },
    { pct: 92, label: '📏  Estimating width & depth…',           step: 5 },
  ];

  let stageIdx = 0;
  const stageInterval = setInterval(() => {
    if (stageIdx < stages.length) {
      const s = stages[stageIdx];
      setProgress(s.pct, s.label);
      activateStep(s.step);
      stageIdx++;
    }
  }, 1400);

  try {
    const resp = await fetch('/analyze', { method: 'POST', body: form });
    clearInterval(stageInterval);

    const data = await resp.json();

    if (!resp.ok || data.error) {
      throw new Error(data.error || `Server error ${resp.status}`);
    }

    // Final progress
    setProgress(100, '✅  Analysis complete!');
    STEPS.forEach(s => { s.classList.remove('active'); s.classList.add('done'); });

    await delay(600);
    displayResults(data);

  } catch (err) {
    clearInterval(stageInterval);
    setProgress(0, `❌  Error: ${err.message}`);
    progressBar.style.background = 'var(--c-danger)';
    showToast(`Error: ${err.message}`, 'error');
    console.error(err);
  } finally {
    analyzeBtn.classList.remove('loading');
    analyzeBtn.querySelector('.btn-text').textContent = 'Analyze Video';
    analyzeBtn.disabled = false;
  }
}

function setProgress (pct, label) {
  progressBar.style.width = pct + '%';
  progressLabel.textContent = label;
}

function activateStep (idx) {
  if (idx > 0) {
    STEPS[idx - 1].classList.remove('active');
    STEPS[idx - 1].classList.add('done');
  }
  if (STEPS[idx]) STEPS[idx].classList.add('active');
}

function resetSteps () {
  STEPS.forEach(s => { s.classList.remove('active','done'); });
  progressBar.style.background = '';
  progressBar.style.width = '0%';
}

// ── Display results ───────────────────────────────────────────────────────────
function displayResults (data) {
  const m = data.metrics || {};

  // Session badge
  sessionBadge.textContent = `Session: ${data.session_id}`;

  // Metrics – animated count-up
  countUp(mCount,    m.num_potholes  ?? 0,   0);
  countUp(mWidth,    m.width_cm      ?? 0,   1, ' cm');
  countUp(mDepth,    m.depth_cm      ?? 0,   1, ' cm');
  countUp(mArea,     m.area_cm2      ?? 0,   1, ' cm²');
  countUp(mVolume,   m.volume_cm3    ?? 0,   1, ' cm³');
  countUp(mCost,     m.repair_cost_usd ?? 0, 2, ' $');
  mSeverity.textContent = m.severity   || '–';
  mConf.textContent     = m.confidence || '–';

  // Images & downloads
  setImg(imgDetection, dlDetection, data.detection_image);
  setImg(imgMatches,   dlMatches,   data.match_image);
  setImg(imgDepth,     dlDepth,     data.depth_map);
  setImg(imgCloud,     dlCloud,     data.pointcloud);

  // Show section
  resultsSection.classList.remove('hidden');
  setTimeout(() => resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
}

function setImg (imgEl, dlEl, url) {
  if (!url) return;
  const ts = '?t=' + Date.now();
  imgEl.src = url + ts;
  dlEl.href = url;
}

// ── Counter animation ─────────────────────────────────────────────────────────
function countUp (el, target, decimals = 0, suffix = '') {
  const duration = 1200;
  const start    = performance.now();
  function frame (now) {
    const t = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - t, 3);
    el.textContent = (target * eased).toFixed(decimals) + suffix;
    if (t < 1) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

// ── Re-analyze ────────────────────────────────────────────────────────────────
reanalyzeBtn.addEventListener('click', () => {
  selectedFile = null;
  videoInput.value = '';
  dropContent.classList.remove('hidden');
  dropSelected.classList.add('hidden');
  analyzeBtn.disabled = true;
  progressWrap.classList.add('hidden');
  resultsSection.classList.add('hidden');
  resetSteps();
  window.scrollTo({ top: 0, behavior: 'smooth' });
});

// ── Toast notification ────────────────────────────────────────────────────────
function showToast (msg, type = 'info') {
  const t = document.createElement('div');
  t.textContent = msg;
  Object.assign(t.style, {
    position:     'fixed',
    bottom:       '24px',
    right:        '24px',
    zIndex:       '9999',
    padding:      '14px 22px',
    borderRadius: '12px',
    fontSize:     '0.88rem',
    fontWeight:   '600',
    color:        '#fff',
    background:   type === 'error' ? 'rgba(255,76,76,0.9)' : 'rgba(108,99,255,0.9)',
    boxShadow:    '0 8px 32px rgba(0,0,0,0.5)',
    backdropFilter: 'blur(12px)',
    transition:   'opacity 0.4s',
    maxWidth:     '380px',
  });
  document.body.appendChild(t);
  setTimeout(() => { t.style.opacity = '0'; setTimeout(() => t.remove(), 400); }, 4000);
}

// ── Utility ───────────────────────────────────────────────────────────────────
const delay = ms => new Promise(r => setTimeout(r, ms));
