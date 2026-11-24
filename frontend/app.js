/* ============================================================
   WheelValue — Vanilla JS Application
   ============================================================ */

"use strict";

// ─── Currency formatter (en-IN, INR, no decimals) ────────────
const INR = new Intl.NumberFormat("en-IN", {
  style: "currency",
  currency: "INR",
  maximumFractionDigits: 0,
});

function inrFormat(value) {
  return INR.format(value);
}

function toLakhs(value) {
  return (value / 1e5).toFixed(2) + " Lakhs";
}

// ─── Shared state ─────────────────────────────────────────────
let metadata = null;
let analyticsCharts = {};

// ─── Tab navigation ───────────────────────────────────────────
function initTabs() {
  const buttons = document.querySelectorAll(".tab-btn");
  const views = document.querySelectorAll(".tab-view");

  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.tab;

      buttons.forEach((b) => {
        b.classList.toggle("active", b === btn);
        b.setAttribute("aria-selected", b === btn ? "true" : "false");
      });

      views.forEach((v) => {
        const isTarget = v.id === `tab-${target}`;
        v.classList.toggle("active", isTarget);
        v.hidden = !isTarget;
      });

      if (target === "analytics") loadAnalytics();
      if (target === "performance") renderPerformance();
    });
  });
}

// ─── Load metadata from API ───────────────────────────────────
async function loadMetadata() {
  try {
    const res = await fetch("/api/metadata");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    metadata = await res.json();
    populateForm(metadata);
    renderPerformance(); // pre-load performance data
  } catch (err) {
    console.error("Failed to load metadata:", err);
  }
}

// ─── Populate form selects and input attributes ───────────────
function populateForm(meta) {
  // Dropdowns
  const selectFields = {
    "f-brand": "brand",
    "f-fuel": "fuel",
    "f-seller_type": "seller_type",
    "f-transmission": "transmission",
    "f-owner": "owner",
  };

  for (const [id, key] of Object.entries(selectFields)) {
    const sel = document.getElementById(id);
    if (!sel) continue;
    const options = meta.categorical_options[key] || [];
    options.forEach((opt) => {
      const el = document.createElement("option");
      el.value = opt;
      el.textContent = opt;
      sel.appendChild(el);
    });
  }

  // Numeric inputs: set placeholder (median) and min/max
  const ranges = meta.numeric_ranges;
  const numericMap = {
    "f-year": "year",
    "f-km_driven": "km_driven",
    "f-mileage": "mileage",
    "f-engine": "engine",
    "f-max_power": "max_power",
    "f-seats": "seats",
  };

  for (const [id, key] of Object.entries(numericMap)) {
    const inp = document.getElementById(id);
    if (!inp || !ranges[key]) continue;
    const r = ranges[key];
    if (r.min !== undefined) inp.min = r.min;
    if (r.max !== undefined) inp.max = r.max;
    inp.placeholder = `Median: ${r.median}`;
  }

  // Also populate analytics fuel dropdown
  const aFuel = document.getElementById("a-fuel");
  if (aFuel) {
    (meta.categorical_options.fuel || []).forEach((f) => {
      const el = document.createElement("option");
      el.value = f;
      el.textContent = f;
      aFuel.appendChild(el);
    });
  }
}

// ─── Prediction form ──────────────────────────────────────────
function initPredictorForm() {
  const form = document.getElementById("car-form");
  const btn = document.getElementById("predict-btn");
  const resetBtn = document.getElementById("reset-btn");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearErrors();

    // Gather payload
    const payload = gatherPayload(form);

    setLoading(true);
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (res.status === 422) {
        const body = await res.json();
        const errs =
          body.detail?.errors ||
          (Array.isArray(body.detail)
            ? body.detail.map((d) => d.msg)
            : [String(body.detail)]);
        showErrors(errs);
        return;
      }

      if (!res.ok) {
        showErrors([`Server error: ${res.status}`]);
        return;
      }

      const data = await res.json();
      renderPrediction(data);
    } catch (err) {
      showErrors(["Network error — ensure the server is running."]);
      console.error(err);
    } finally {
      setLoading(false);
    }
  });

  resetBtn.addEventListener("click", () => {
    form.reset();
    clearErrors();
    document.getElementById("result-empty").hidden = false;
    document.getElementById("result-populated").hidden = true;
  });
}

function gatherPayload(form) {
  const data = new FormData(form);
  const payload = {};
  for (const [k, v] of data.entries()) {
    // Numeric fields
    if (
      ["year", "km_driven", "mileage", "engine", "max_power", "seats"].includes(k)
    ) {
      payload[k] = v === "" ? null : Number(v);
    } else {
      payload[k] = v;
    }
  }
  return payload;
}

function setLoading(loading) {
  const btn = document.getElementById("predict-btn");
  const text = btn.querySelector(".btn-text");
  const spinner = btn.querySelector(".btn-spinner");
  btn.disabled = loading;
  text.textContent = loading ? "Estimating…" : "Estimate Resale Price";
  spinner.hidden = !loading;
}

function clearErrors() {
  const box = document.getElementById("validation-errors");
  box.hidden = true;
  document.getElementById("error-list").innerHTML = "";
}

function showErrors(errors) {
  const box = document.getElementById("validation-errors");
  const list = document.getElementById("error-list");
  list.innerHTML = errors.map((e) => `<li>${e}</li>`).join("");
  box.hidden = false;
  box.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ─── Render prediction result ─────────────────────────────────
function renderPrediction(data) {
  document.getElementById("result-empty").hidden = true;
  document.getElementById("result-populated").hidden = false;

  const price = data.predicted_price_inr;
  const mae = data.validation_mae;

  document.getElementById("hero-price").textContent = inrFormat(price);
  document.getElementById("hero-lakhs").textContent = toLakhs(price);
  document.getElementById("error-range-pill").textContent =
    `Expected range: ${inrFormat(data.error_range_low)} – ${inrFormat(data.error_range_high)}`;
  document.getElementById("mae-value").textContent = mae.toLocaleString("en-IN", {
    maximumFractionDigits: 0,
  });

  renderSubmittedSummary(data.submitted_details);
  renderShapChart(data.shap_contributions);
  renderShapTakeaway(data.shap_contributions);
}

function renderSubmittedSummary(details) {
  const container = document.getElementById("submitted-pills");
  const fields = [
    ["Year", details.year],
    ["Car Age", `${details.car_age} yr`],
    ["KM Driven", details.km_driven?.toLocaleString("en-IN")],
    ["Fuel", details.fuel],
    ["Transmission", details.transmission],
    ["Power", `${details.max_power} bhp`],
    ["Engine", `${details.engine} cc`],
    ["Seats", details.seats],
    ["Owner", details.owner],
    ["Seller", details.seller_type],
  ];
  container.innerHTML = fields
    .filter(([, v]) => v !== null && v !== undefined)
    .map(([k, v]) => `<span class="spec-pill"><strong>${k}:</strong> ${v}</span>`)
    .join("");
}

// ─── SHAP horizontal diverging bar chart (pure HTML/CSS) ─────
function renderShapChart(contributions) {
  const container = document.getElementById("shap-chart");
  container.innerHTML = "";

  const top = contributions.slice(0, 8);
  if (!top.length) return;

  const maxAbs = Math.max(...top.map((c) => Math.abs(c.impact)));
  const BAR_MAX_PCT = 48; // each side occupies up to 48% of track width

  top.forEach((c) => {
    const pct = maxAbs > 0 ? (Math.abs(c.impact) / maxAbs) * BAR_MAX_PCT : 0;
    const isPos = c.impact >= 0;
    const valStr = c.impact.toFixed(3);

    const row = document.createElement("div");
    row.className = "shap-bar-row";

    const label = document.createElement("div");
    label.className = "shap-feat-label";
    label.title = c.label;
    label.textContent = c.label;

    const track = document.createElement("div");
    track.className = "shap-bar-track";

    const center = document.createElement("div");
    center.className = "shap-bar-center";

    const fill = document.createElement("div");
    fill.className = `shap-bar-fill ${isPos ? "positive" : "negative"}`;
    fill.style.width = `${pct}%`;

    const val = document.createElement("span");
    val.className = "shap-bar-value";
    val.textContent = (isPos ? "+" : "") + valStr;

    fill.appendChild(val);
    track.appendChild(center);
    track.appendChild(fill);
    row.appendChild(label);
    row.appendChild(track);
    container.appendChild(row);
  });
}

function renderShapTakeaway(contributions) {
  const list = document.getElementById("shap-takeaway");
  const top3 = contributions.slice(0, 3);
  list.innerHTML = top3
    .map((c) => {
      const dir = c.impact >= 0 ? "raised" : "lowered";
      return `<li><strong>${c.label}</strong> ${dir} the estimate (${
        c.impact >= 0 ? "+" : ""
      }${c.impact.toFixed(3)} log-price units).</li>`;
    })
    .join("");
}

// ─── Market Analytics ─────────────────────────────────────────
function initAnalyticsFilters() {
  const fuelSel = document.getElementById("a-fuel");
  const transSel = document.getElementById("a-transmission");
  if (fuelSel) fuelSel.addEventListener("change", loadAnalytics);
  if (transSel) transSel.addEventListener("change", loadAnalytics);
}

async function loadAnalytics() {
  const fuel = document.getElementById("a-fuel")?.value || "All";
  const transmission = document.getElementById("a-transmission")?.value || "All";

  const params = new URLSearchParams({ fuel, transmission });
  try {
    const res = await fetch(`/api/analytics?${params}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderAnalytics(data);
  } catch (err) {
    console.error("Analytics fetch failed:", err);
  }
}

function renderAnalytics(data) {
  document.getElementById("record-count").textContent =
    data.record_count.toLocaleString("en-IN");

  renderBarChart(
    "chart-brands",
    data.brand_avg_prices.map((d) => d.brand),
    data.brand_avg_prices.map((d) => d.avg_price_lakhs),
    "Avg Price (₹ Lakhs)",
    "horizontal"
  );

  renderBarChart(
    "chart-fuel",
    data.fuel_avg_prices.map((d) => d.fuel),
    data.fuel_avg_prices.map((d) => d.avg_price_lakhs),
    "Avg Price (₹ Lakhs)",
    "vertical"
  );

  renderLineChart(
    "chart-year",
    data.year_price_trend.map((d) => d.year),
    data.year_price_trend.map((d) => d.avg_price_lakhs),
    "Avg Price (₹ Lakhs)"
  );
}

const PALETTE = [
  "#2563eb","#3b82f6","#60a5fa","#93c5fd","#1d4ed8",
  "#7c3aed","#8b5cf6","#a78bfa","#4f46e5","#6366f1",
  "#0ea5e9","#38bdf8","#06b6d4","#22d3ee","#0891b2",
];

function getOrCreateChart(canvasId, type, chartData, chartOptions) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  if (analyticsCharts[canvasId]) {
    analyticsCharts[canvasId].destroy();
    delete analyticsCharts[canvasId];
  }

  analyticsCharts[canvasId] = new Chart(canvas, {
    type,
    data: chartData,
    options: chartOptions,
  });
}

function commonOptions(title, isHorizontal) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    indexAxis: isHorizontal ? "y" : "x",
    plugins: {
      legend: { display: false },
      title: { display: false },
      tooltip: {
        callbacks: {
          label: (ctx) => ` ₹${ctx.parsed[isHorizontal ? "x" : "y"].toFixed(2)} Lakhs`,
        },
      },
    },
    scales: {
      x: {
        grid: { color: "#e2e8f0" },
        ticks: { color: "#64748b", font: { size: 11 } },
      },
      y: {
        grid: { color: "#e2e8f0" },
        ticks: { color: "#64748b", font: { size: 11 } },
      },
    },
  };
}

function renderBarChart(canvasId, labels, values, yLabel, orientation) {
  const isHorizontal = orientation === "horizontal";
  const bgColors = labels.map((_, i) => PALETTE[i % PALETTE.length]);

  getOrCreateChart(
    canvasId,
    "bar",
    {
      labels,
      datasets: [
        {
          label: yLabel,
          data: values,
          backgroundColor: bgColors,
          borderRadius: 4,
          borderSkipped: false,
        },
      ],
    },
    commonOptions(yLabel, isHorizontal)
  );
}

function renderLineChart(canvasId, labels, values, yLabel) {
  getOrCreateChart(
    canvasId,
    "line",
    {
      labels,
      datasets: [
        {
          label: yLabel,
          data: values,
          borderColor: "#2563eb",
          backgroundColor: "rgba(37,99,235,.1)",
          fill: true,
          tension: 0.3,
          pointRadius: 3,
          pointBackgroundColor: "#2563eb",
        },
      ],
    },
    {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => ` ₹${ctx.parsed.y.toFixed(2)} Lakhs`,
          },
        },
      },
      scales: {
        x: { grid: { color: "#e2e8f0" }, ticks: { color: "#64748b", font: { size: 11 } } },
        y: { grid: { color: "#e2e8f0" }, ticks: { color: "#64748b", font: { size: 11 } } },
      },
    }
  );
}

// ─── Model Performance view ───────────────────────────────────
function renderPerformance() {
  if (!metadata) return;

  const metrics = metadata.model_metrics;
  const fi = metadata.feature_importance;

  // KPI cards
  document.getElementById("kpi-winner").textContent = metrics.winner || "—";
  document.getElementById("kpi-mae").textContent = inrFormat(metrics.test.mae);
  document.getElementById("kpi-rmse").textContent = inrFormat(metrics.test.rmse);
  document.getElementById("kpi-r2").textContent = metrics.test.r2.toFixed(4);

  // CV table
  const tbody = document.getElementById("cv-tbody");
  if (tbody && metrics.comparison) {
    tbody.innerHTML = metrics.comparison
      .map((row) => {
        const isWinner = row.model === metrics.winner;
        return `<tr class="${isWinner ? "winner-row" : ""}">
          <td>${row.model}${isWinner ? " 🏆" : ""}</td>
          <td>${inrFormat(row.val_mae)}</td>
          <td>${inrFormat(row.val_rmse)}</td>
          <td>${row.val_r2.toFixed(4)}</td>
          <td>${inrFormat(row.train_mae)}</td>
          <td>${inrFormat(row.mae_gap)}</td>
        </tr>`;
      })
      .join("");
  }

  // Feature importance CSS bar chart
  renderImportanceChart(fi);
}

function renderImportanceChart(fi) {
  const container = document.getElementById("importance-chart");
  if (!container || !fi) return;

  const entries = Object.entries(fi).sort((a, b) => b[1] - a[1]);
  const maxVal = entries[0]?.[1] || 1;

  container.innerHTML = entries
    .map(([feat, val]) => {
      const pct = (val / maxVal) * 100;
      const label = feat.replace(/_/g, " ");
      const labelCap = label.charAt(0).toUpperCase() + label.slice(1);
      return `<div class="imp-row">
        <div class="imp-label" title="${labelCap}">${labelCap}</div>
        <div class="imp-bar-track">
          <div class="imp-bar-fill" style="width:${pct}%"></div>
        </div>
        <div class="imp-value">${inrFormat(val)}</div>
      </div>`;
    })
    .join("");
}

// ─── Boot ─────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initPredictorForm();
  initAnalyticsFilters();
  loadMetadata();
});
