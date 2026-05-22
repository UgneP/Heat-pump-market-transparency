const state = {
  data: null,
  rows: [],
  sortKey: "Price",
  sortDirection: "asc",
};

const eur = new Intl.NumberFormat("en-IE", {
  style: "currency",
  currency: "EUR",
  maximumFractionDigits: 0,
});

function formatEur(value) {
  return eur.format(value).replace(/,/g, "'");
}

function numberValue(id) {
  const value = document.getElementById(id).value;
  return value === "" ? null : Number(value);
}

function selectValue(id) {
  const value = document.getElementById(id).value;
  return value === "Not specified" ? null : value;
}

function setOptions(id, values, includeAny = true) {
  const select = document.getElementById(id);
  select.innerHTML = "";
  if (includeAny) {
    select.append(new Option("Not specified", "Not specified"));
  }
  values.forEach((value) => select.append(new Option(value, value)));
}

function setInputLimits(id, limits) {
  const input = document.getElementById(id);
  input.min = limits[0];
  input.max = limits[1];
  input.placeholder = `${limits[0]}-${limits[1]}`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll('"', "&quot;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function selectedMultiValues(id) {
  return [...document.querySelectorAll(`#${id} input[type="checkbox"]:checked`)].map((input) => input.value);
}

function renderMultiSelect(id, optionsId, values, allLabel) {
  const details = document.getElementById(id);
  const menu = document.getElementById(optionsId);
  const summary = details.querySelector("summary");
  menu.innerHTML = values.map((value) => `
    <label>
      <input type="checkbox" value="${escapeHtml(value)}">
      <span>${escapeHtml(value)}</span>
    </label>
  `).join("");
  const updateLabel = () => {
    const selected = selectedMultiValues(id);
    if (!selected.length) {
      summary.textContent = allLabel;
    } else if (selected.length === 1) {
      summary.textContent = selected[0];
    } else {
      summary.textContent = `${selected.length} selected`;
    }
  };
  menu.addEventListener("change", () => {
    updateLabel();
    renderDashboard();
  });
  details.addEventListener("toggle", () => {
    if (details.open) {
      document.querySelectorAll(".multi-select").forEach((other) => {
        if (other !== details) other.removeAttribute("open");
      });
    }
  });
  updateLabel();
}

function setRangeLimits(minId, maxId, limits, step) {
  const minInput = document.getElementById(minId);
  const maxInput = document.getElementById(maxId);
  [minInput, maxInput].forEach((input) => {
    input.min = limits[0];
    input.max = limits[1];
    input.step = step;
  });
  minInput.value = limits[0];
  maxInput.value = limits[1];
}

function metricCard(label, value) {
  return `<div class="metric-card"><div class="metric-label">${label}</div><div class="metric-value">${value}</div></div>`;
}

function websiteLabel(value) {
  if (!value) return "";
  try {
    const host = new URL(value).hostname.replace(/^www\./, "");
    return host.split(".").slice(-2).join(".");
  } catch {
    return String(value).replace(/^https?:\/\//, "").replace(/^www\./, "").split("/")[0];
  }
}

function activateTab(name) {
  document.querySelectorAll(".tab").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.tab === name);
  });
  document.querySelectorAll(".panel").forEach((panel) => {
    panel.classList.toggle("is-active", panel.id === name);
  });
}

function predictionMatches(row, inputs) {
  const [power, storage, noise, standby, config, tank, refrigerant] = row;
  const tolerances = { power: 0.26, storage: 16, noise: 1.1, standby: 2.6 };
  if (inputs.config && config !== inputs.config) return false;
  if (inputs.tank && tank !== inputs.tank) return false;
  if (inputs.refrigerant && refrigerant !== inputs.refrigerant) return false;
  if (inputs.power !== null && Math.abs(power - inputs.power) > tolerances.power) return false;
  if (inputs.storage !== null && Math.abs(storage - inputs.storage) > tolerances.storage) return false;
  if (inputs.noise !== null && Math.abs(noise - inputs.noise) > tolerances.noise) return false;
  if (inputs.standby !== null && Math.abs(standby - inputs.standby) > tolerances.standby) return false;
  return true;
}

function closestPredictions(inputs) {
  const numeric = [
    ["power", 0],
    ["storage", 1],
    ["noise", 2],
    ["standby", 3],
  ].filter(([key]) => inputs[key] !== null);

  return state.data.predictionRows
    .filter((row) => {
      if (inputs.config && row[4] !== inputs.config) return false;
      if (inputs.tank && row[5] !== inputs.tank) return false;
      if (inputs.refrigerant && row[6] !== inputs.refrigerant) return false;
      return true;
    })
    .map((row) => {
      const distance = numeric.reduce((total, [key, index]) => {
        const [min, max] = state.data.limits[key];
        return total + Math.abs(row[index] - inputs[key]) / Math.max(max - min, 1);
      }, 0);
      return { row, distance };
    })
    .sort((a, b) => a.distance - b.distance)
    .slice(0, 500)
    .map((item) => item.row);
}

function checkPrice(event) {
  if (event) event.preventDefault();

  const price = numberValue("offer-price") || 0;
  const inputs = {
    power: numberValue("power"),
    storage: numberValue("storage"),
    config: selectValue("config"),
    tank: selectValue("tank"),
    refrigerant: selectValue("refrigerant"),
    noise: numberValue("noise"),
    standby: numberValue("standby"),
  };

  let matches = state.data.predictionRows.filter((row) => predictionMatches(row, inputs));
  if (!matches.length) {
    matches = closestPredictions(inputs);
  }

  let rawLow = Infinity;
  let rawHigh = -Infinity;
  matches.forEach((row) => {
    const prediction = row[7];
    if (prediction < rawLow) rawLow = prediction;
    if (prediction > rawHigh) rawHigh = prediction;
  });
  const low = rawLow * (1 - state.data.expectedPriceBand);
  const high = rawHigh * (1 + state.data.expectedPriceBand);

  let label = "In expected range";
  let color = "var(--green)";
  let message = "This offer is within the model-based expected range for the entered characteristics.";

  if (price < low) {
    label = "Cheaper than expected - check details";
    color = "var(--gray)";
    message = "Congratulations on a cheaper price, but check whether everything is included and whether the manufacturer or supplier is reliable.";
  } else if (price > high) {
    label = "Higher than expected";
    color = "var(--red)";
    message = "This offer is above the expected range; check whether extra services, warranty, installation scope, or availability explain the premium.";
  }

  const missing = Object.entries(inputs)
    .filter(([, value]) => value === null)
    .map(([key]) => ({
      power: "Rated power",
      storage: "Storage size",
      config: "Configuration",
      tank: "Tank configuration",
      refrigerant: "Refrigerant type",
      noise: "Noise",
      standby: "Standby power",
    })[key]);

  const missingNote = missing.length
    ? ` Range is wider because these inputs were not specified: ${missing.join(", ")}.`
    : "";

  document.getElementById("result").innerHTML = `
    <span class="label-pill" style="background:${color}">${label}</span>
    <div class="metric-label">Expected price range</div>
    <div class="range-value">${formatEur(low)} - ${formatEur(high)}</div>
    <p class="small-note">Your offer: ${formatEur(price)}</p>
    <p class="small-note">${message}${missingNote}</p>
  `;
}

function safeCheckPrice(event) {
  try {
    checkPrice(event);
  } catch (error) {
    console.error(error);
    document.getElementById("result").innerHTML = `
      <span class="label-pill" style="background:var(--red)">Price check unavailable</span>
      <p class="small-note">${error.message}</p>
    `;
  }
}

function uniqueRows(key) {
  return [...new Set(state.rows.map((row) => row[key]).filter(Boolean))].sort();
}

function filteredRows() {
  const query = document.getElementById("table-search").value.toLowerCase();
  const manufacturers = selectedMultiValues("manufacturer-filter");
  const configurations = selectedMultiValues("configuration-filter");
  const refrigerant = document.getElementById("refrigerant-filter").value;
  const priceMin = numberValue("price-min-filter");
  const priceMax = numberValue("price-max-filter");
  const powerMin = numberValue("power-min-filter");
  const powerMax = numberValue("power-max-filter");

  return state.rows.filter((row) => {
    if (manufacturers.length && !manufacturers.includes(row["Manufacturer display"])) return false;
    if (configurations.length && !configurations.includes(row.Configuration)) return false;
    if (refrigerant !== "All refrigerants" && row.Refrigerant !== refrigerant) return false;
    if (priceMin !== null && row.Price < priceMin) return false;
    if (priceMax !== null && row.Price > priceMax) return false;
    if (powerMin !== null && row["Rated Power low T [kW]"] < powerMin) return false;
    if (powerMax !== null && row["Rated Power low T [kW]"] > powerMax) return false;
    if (!query) return true;
    return [
      row["Manufacturer display"],
      row.Titel,
      row.Configuration,
      row.Refrigerant,
      row.Website,
    ].join(" ").toLowerCase().includes(query);
  });
}

function syncRange(minId, maxId, labelId, formatter) {
  const minInput = document.getElementById(minId);
  const maxInput = document.getElementById(maxId);
  let min = Number(minInput.value);
  let max = Number(maxInput.value);
  if (min > max) {
    if (document.activeElement === minInput) {
      max = min;
      maxInput.value = max;
    } else {
      min = max;
      minInput.value = min;
    }
  }
  document.getElementById(labelId).textContent = `${formatter(min)} - ${formatter(max)}`;
  const minLimit = Number(minInput.min);
  const maxLimit = Number(minInput.max);
  const span = Math.max(maxLimit - minLimit, 1);
  const start = ((min - minLimit) / span) * 100;
  const end = ((max - minLimit) / span) * 100;
  minInput.closest(".range-stack").style.setProperty("--range-start", `${start}%`);
  minInput.closest(".range-stack").style.setProperty("--range-end", `${end}%`);
}

function syncDashboardRanges() {
  syncRange("price-min-filter", "price-max-filter", "price-range-label", formatEur);
  syncRange("power-min-filter", "power-max-filter", "power-range-label", (value) => `${value.toFixed(1)} kW`);
}

function renderDashboard() {
  const rows = filteredRows().sort((a, b) => {
    const av = a[state.sortKey];
    const bv = b[state.sortKey];
    const direction = state.sortDirection === "asc" ? 1 : -1;
    if (typeof av === "number" && typeof bv === "number") return (av - bv) * direction;
    return String(av).localeCompare(String(bv)) * direction;
  });

  const prices = rows.map((row) => row.Price).filter((value) => typeof value === "number");
  const scops = rows.map((row) => row.SCOP).filter((value) => typeof value === "number");
  const median = (values) => {
    if (!values.length) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length / 2)];
  };

  document.getElementById("dashboard-metrics").innerHTML = [
    metricCard("Offers", rows.length.toLocaleString("en-US")),
    metricCard("Manufacturers", new Set(rows.map((row) => row["Manufacturer display"])).size),
    metricCard("Median price", formatEur(median(prices))),
    metricCard("Median SCOP", median(scops).toFixed(2)),
  ].join("");

  document.querySelector("#offers-table tbody").innerHTML = rows.slice(0, 300).map((row) => `
    <tr>
      <td>${row["Manufacturer display"] || ""}</td>
      <td>${row.Titel || ""}</td>
      <td>${formatEur(row.Price || 0)}</td>
      <td>${row["Rated Power low T [kW]"] ?? ""}</td>
      <td>${typeof row.SCOP === "number" ? row.SCOP.toFixed(2) : ""}</td>
      <td>${row.Configuration || ""}</td>
      <td>${row.Refrigerant || ""}</td>
      <td>${row.Website ? `<a href="${row.Website}" target="_blank" rel="noreferrer">${websiteLabel(row.Website)}</a>` : ""}</td>
    </tr>
  `).join("");
}

function setupDashboard() {
  renderMultiSelect("manufacturer-filter", "manufacturer-options", uniqueRows("Manufacturer display"), "All manufacturers");
  renderMultiSelect("configuration-filter", "configuration-options", uniqueRows("Configuration"), "All configurations");

  const refrigerant = document.getElementById("refrigerant-filter");
  refrigerant.append(new Option("All refrigerants", "All refrigerants"));
  uniqueRows("Refrigerant").forEach((value) => refrigerant.append(new Option(value, value)));

  const prices = state.rows.map((row) => row.Price).filter((value) => typeof value === "number");
  const powers = state.rows.map((row) => row["Rated Power low T [kW]"]).filter((value) => typeof value === "number");
  const priceStep = 250;
  const powerStep = 0.5;
  const priceLimits = [
    Math.floor(Math.min(...prices) / priceStep) * priceStep,
    Math.ceil(Math.max(...prices) / priceStep) * priceStep,
  ];
  const powerLimits = [
    Math.floor(Math.min(...powers) / powerStep) * powerStep,
    Math.ceil(Math.max(...powers) / powerStep) * powerStep,
  ];
  setRangeLimits("price-min-filter", "price-max-filter", priceLimits, priceStep);
  setRangeLimits("power-min-filter", "power-max-filter", powerLimits, powerStep);
  syncDashboardRanges();

  [
    "table-search",
    "refrigerant-filter",
    "price-min-filter",
    "price-max-filter",
    "power-min-filter",
    "power-max-filter",
  ].forEach((id) => {
    document.getElementById(id).addEventListener("input", () => {
      syncDashboardRanges();
      renderDashboard();
    });
  });

  document.querySelectorAll("th[data-sort]").forEach((heading) => {
    heading.addEventListener("click", () => {
      const key = heading.dataset.sort;
      if (state.sortKey === key) {
        state.sortDirection = state.sortDirection === "asc" ? "desc" : "asc";
      } else {
        state.sortKey = key;
        state.sortDirection = "asc";
      }
      renderDashboard();
    });
  });

  renderDashboard();
}

async function init() {
  const response = await fetch("data/app-data.json");
  state.data = await response.json();
  state.rows = state.data.dashboardRows;

  setOptions("config", state.data.options.config);
  setOptions("tank", state.data.options.tank);
  setOptions("refrigerant", state.data.options.refrigerant);
  setInputLimits("power", state.data.limits.power);
  setInputLimits("storage", state.data.limits.storage);
  setInputLimits("noise", state.data.limits.noise);
  setInputLimits("standby", state.data.limits.standby);

  document.getElementById("model-metrics").innerHTML = [
    metricCard("Holdout R2", state.data.modelMetrics.r2.toFixed(2)),
    metricCard("Mean absolute error", formatEur(state.data.modelMetrics.mae)),
    metricCard("Mean percentage error", `${state.data.modelMetrics.mape.toFixed(1)}%`),
  ].join("");

  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => activateTab(button.dataset.tab));
  });

  document.getElementById("price-form").addEventListener("submit", safeCheckPrice);
  setupDashboard();
  safeCheckPrice();
}

init();
