/* =============================================
   ECG Analyser — ecg.js
   Mirrors the decision logic from your notebook
   ============================================= */

'use strict';

// ── DOM refs ─────────────────────────────────
const hrInput      = document.getElementById('hr');
const spo2Input    = document.getElementById('spo2');
const rrInput      = document.getElementById('rr');
const ecgTextarea  = document.getElementById('ecg-signal');
const signalMeta   = document.getElementById('signal-meta');
const signalError  = document.getElementById('signal-error');
const analyzeBtn   = document.getElementById('analyze-btn');
const btnText      = document.getElementById('btn-text');
const btnLoader    = document.getElementById('btn-loader');
const resultCard   = document.getElementById('result-card');
const waveformSvg  = document.getElementById('waveform-svg');
const resultBadge  = document.getElementById('result-badge');
const resultStatus = document.getElementById('result-status');
const resultSub    = document.getElementById('result-sub');
const resultDetail = document.getElementById('result-details');
const demoBtn      = document.getElementById('demo-btn');

// ── Live signal counter ───────────────────────
ecgTextarea.addEventListener('input', updateSignalCount);

function updateSignalCount() {
  const vals = parseSignal();
  signalMeta.textContent = vals.length + ' / 300 values entered';
  signalMeta.style.color = vals.length >= 300 ? '#1D9E75' : '';
  signalError.classList.remove('show');
}

// ── Parse ECG textarea ────────────────────────
function parseSignal() {
  return ecgTextarea.value
    .split(',')
    .map(s => parseFloat(s.trim()))
    .filter(n => !isNaN(n));
}

// ── Infer ECG class from signal statistics ────
// Mirrors your CNN's 3-class output:
//   0 = Normal, 1 = Arrhythmia, 2 = Dangerous
function inferECGClass(signal) {
  if (!signal || signal.length < 10) return 0;

  const n    = signal.length;
  const mean = signal.reduce((a, b) => a + b, 0) / n;
  const variance = signal.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / n;
  const std  = Math.sqrt(variance);
  const max  = Math.max(...signal);
  const min  = Math.min(...signal);
  const range = max - min;

  // Count zero-crossings (irregular rhythm indicator)
  let crossings = 0;
  for (let i = 1; i < signal.length; i++) {
    if ((signal[i] - mean) * (signal[i - 1] - mean) < 0) crossings++;
  }
  const crossRate = crossings / signal.length;

  if (std > 0.6 && range > 2.5) return 2;          // dangerous pattern
  if (std > 0.3 || range > 1.5 || crossRate > 0.4) return 1; // arrhythmia
  return 0;                                           // normal
}

// ── Vitals risk level ─────────────────────────
function vitalsRisk(hr, spo2, rr) {
  if (spo2 < 88 || rr > 28 || hr > 140) return 2;
  if (spo2 < 94 || rr > 22 || hr > 110 || hr < 50) return 1;
  return 0;
}

// ── Final decision — exact logic from notebook ─
function finalDecision(ecgPred, hr, spo2, rr) {
  const vitalsPred = vitalsRisk(hr, spo2, rr);

  if (ecgPred === 2)
    return { text: 'Dangerous Arrhythmia',       level: 'danger',  sub: 'High-variance ECG pattern detected. Immediate evaluation recommended.' };
  if (ecgPred === 1 && vitalsPred === 2)
    return { text: 'High Risk Cardiac Condition', level: 'danger',  sub: 'Arrhythmia combined with critical vitals.' };
  if (ecgPred === 1)
    return { text: 'Cardiac Arrhythmia',          level: 'warning', sub: 'Irregular ECG pattern detected. Monitor closely.' };
  if (spo2 < 90)
    return { text: 'Hypoxia',                    level: 'danger',  sub: 'Oxygen saturation critically low. Immediate oxygen therapy needed.' };
  if (rr > 25 || rr < 8)
    return { text: 'Respiratory Distress',        level: 'warning', sub: 'Respiratory rate is outside the safe range (' + rr + ' br/min).' };
  if (vitalsPred === 2)
    return { text: 'Critical Health Condition',   level: 'danger',  sub: 'Multiple vitals outside safe range.' };
  if (vitalsPred === 1)
    return { text: 'General Health Risk',         level: 'warning', sub: 'One or more vitals slightly abnormal. Further assessment recommended.' };
  if (hr < 50)
    return { text: 'Bradycardia',                 level: 'warning', sub: 'Heart rate is below the normal threshold (' + hr + ' bpm).' };
  if (hr > 120)
    return { text: 'Tachycardia',                 level: 'warning', sub: 'Heart rate is above the normal threshold (' + hr + ' bpm).' };

  return { text: 'Normal Condition',              level: 'normal',  sub: 'All parameters are within expected ranges.' };
}

// ── Draw waveform ─────────────────────────────
function drawWaveform(signal, level) {
  const W = 600, H = 60, PAD = 4;
  const slice = signal.slice(0, 300);
  const mn = Math.min(...slice);
  const mx = Math.max(...slice);
  const range = mx - mn || 1;

  const pts = slice.map((v, i) => {
    const x = (i / (slice.length - 1)) * W;
    const y = H - PAD - ((v - mn) / range) * (H - PAD * 2);
    return x.toFixed(2) + ',' + y.toFixed(2);
  }).join(' ');

  const colorMap = { normal: '#1D9E75', warning: '#EF9F27', danger: '#E24B4A' };
  const stroke   = colorMap[level] || colorMap.normal;

  waveformSvg.innerHTML =
    '<polyline points="' + pts + '" fill="none" stroke="' + stroke +
    '" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>';
}

// ── Render result card ────────────────────────
function renderResult(result, ecgPred, hr, spo2, rr) {
  // Card background level
  resultCard.className = 'result-card level-' + result.level;

  // Badge
  resultBadge.textContent = result.level.toUpperCase();
  resultBadge.className   = 'result-badge badge-' + result.level;

  // Status text
  resultStatus.textContent = result.text;
  resultStatus.className   = 'result-status status-' + result.level;

  // Sub text
  resultSub.textContent = result.sub;

  // Detail pills
  const ecgLabels  = ['Normal (0)', 'Arrhythmia (1)', 'Dangerous (2)'];
  const hrStatus   = hr < 50 ? 'Bradycardia' : hr > 120 ? 'Tachycardia' : 'Normal';
  const spo2Status = spo2 < 90 ? 'Critical' : spo2 < 94 ? 'Low' : 'Normal';
  const rrStatus   = rr < 8 ? 'Very low' : rr > 25 ? 'Elevated' : 'Normal';

  const pills = [
    { label: 'ECG class',     value: ecgLabels[ecgPred], cls: ecgPred === 0 ? 'ok' : ecgPred === 1 ? 'warn' : 'bad' },
    { label: 'Heart rate',    value: hr + ' bpm — ' + hrStatus,   cls: hr < 50 || hr > 120 ? 'warn' : 'ok' },
    { label: 'SpO2',          value: spo2 + '% — ' + spo2Status,  cls: spo2 < 90 ? 'bad' : spo2 < 94 ? 'warn' : 'ok' },
    { label: 'Resp. rate',    value: rr + ' br/min — ' + rrStatus, cls: rr < 8 || rr > 25 ? 'warn' : 'ok' },
  ];

  resultDetail.innerHTML = pills.map(p =>
    '<div class="detail-pill pill-' + p.cls + '">' +
      '<span class="pill-label">' + p.label + '</span>' +
      '<span class="pill-value">' + p.value + '</span>' +
    '</div>'
  ).join('');
}

// ── Main analysis ─────────────────────────────
analyzeBtn.addEventListener('click', function () {
  const hr   = parseFloat(hrInput.value);
  const spo2 = parseFloat(spo2Input.value);
  const rr   = parseFloat(rrInput.value);
  const signal = parseSignal();

  // Validation
  if (signal.length < 10) {
    signalError.textContent = 'Please enter at least 10 numeric ECG values (300 recommended).';
    signalError.classList.add('show');
    return;
  }
  if (isNaN(hr) || isNaN(spo2) || isNaN(rr)) {
    alert('Please fill in all three vital signs.');
    return;
  }

  // Show loading state
  btnText.classList.add('hidden');
  btnLoader.classList.remove('hidden');
  analyzeBtn.disabled = true;

  // Small delay so UI updates before heavy computation
  setTimeout(function () {
    const ecgPred = inferECGClass(signal);
    const result  = finalDecision(ecgPred, hr, spo2, rr);

    drawWaveform(signal, result.level);
    renderResult(result, ecgPred, hr, spo2, rr);

    resultCard.classList.remove('hidden');
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    btnText.classList.remove('hidden');
    btnLoader.classList.add('hidden');
    analyzeBtn.disabled = false;
  }, 250);
});

// ── Demo signal loader ────────────────────────
demoBtn.addEventListener('click', function () {
  const demo = generateDemoECG(300);
  ecgTextarea.value = demo.join(', ');
  updateSignalCount();
});

// Generates a realistic-looking synthetic ECG waveform
function generateDemoECG(n) {
  const signal = [];
  const fs = 360; // sampling rate used in MIT-BIH

  for (let i = 0; i < n; i++) {
    const t = i / fs;
    // P wave
    let v  = 0.1 * Math.exp(-Math.pow((t % 0.8 - 0.1), 2) / 0.002);
    // QRS complex
    v += -0.1 * Math.exp(-Math.pow((t % 0.8 - 0.22), 2) / 0.0005);
    v +=  1.0 * Math.exp(-Math.pow((t % 0.8 - 0.25), 2) / 0.0003);
    v += -0.2 * Math.exp(-Math.pow((t % 0.8 - 0.28), 2) / 0.0005);
    // T wave
    v += 0.2  * Math.exp(-Math.pow((t % 0.8 - 0.40), 2) / 0.004);
    // Noise
    v += (Math.random() - 0.5) * 0.02;
    signal.push(parseFloat(v.toFixed(4)));
  }
  return signal;
}

// ── Input validation on vitals ────────────────
[hrInput, spo2Input, rrInput].forEach(function (el) {
  el.addEventListener('change', function () {
    const min = parseFloat(el.min);
    const max = parseFloat(el.max);
    const val = parseFloat(el.value);
    if (val < min) el.value = min;
    if (val > max) el.value = max;
  });
});
