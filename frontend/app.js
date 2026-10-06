"use strict";

const $ = (id) => document.getElementById(id);
let files = [];

// 支持 "1/250"、"0.004"、"2" 等写法，返回秒（有限正数），非法返回 NaN
function parseShutter(text) {
  const s = String(text).trim();
  if (!s) return NaN;
  if (s.includes("/")) {
    const parts = s.split("/");
    if (parts.length !== 2) return NaN;
    const a = Number(parts[0]), b = Number(parts[1]);
    if (!isFinite(a) || !isFinite(b) || b === 0) return NaN;
    return a / b;
  }
  return Number(s);
}

function renderList() {
  const list = $("file-list");
  list.innerHTML = "";
  files.forEach((f, i) => {
    const row = document.createElement("div");
    row.className = "file-row";
    const url = URL.createObjectURL(f);
    row.innerHTML = `
      <img src="${url}" alt="预览">
      <span class="fname" title="${f.name}">${i + 1}. ${f.name}</span>
      <label>快门(秒)
        <input class="shutter" data-i="${i}" placeholder="如 1/60 或 0.0167" inputmode="decimal">
      </label>
      <label class="ref-label">
        <input type="radio" name="ref" value="${i}"
               ${i === Math.floor(files.length / 2) ? "checked" : ""}> 参照帧
      </label>`;
    list.appendChild(row);
  });
  validate();
}

function validate() {
  const countOk = files.length >= 3 && files.length <= 7;
  let ok = countOk;
  document.querySelectorAll(".shutter").forEach((inp) => {
    const v = parseShutter(inp.value);
    const good = isFinite(v) && v > 0;
    inp.classList.toggle("bad", !good);
    if (!good) ok = false;
  });
  $("submit").disabled = !ok;
  $("form-hint").textContent = countOk
    ? `已选 ${files.length} 张，填写快门时间（秒）后可合成`
    : `需要 3～7 张 JPEG（当前 ${files.length} 张）`;
}

$("files").addEventListener("change", (e) => {
  files = [...e.target.files];
  renderList();
});

document.addEventListener("input", (e) => {
  if (e.target.classList && e.target.classList.contains("shutter")) validate();
});

$("submit").addEventListener("click", async () => {
  $("error").hidden = true;
  $("results").hidden = true;
  $("loading").hidden = false;
  $("submit").disabled = true;

  const fd = new FormData();
  files.forEach((f) => fd.append("files", f));
  const times = [...document.querySelectorAll(".shutter")]
    .map((inp) => parseShutter(inp.value));
  fd.append("shutter_times", JSON.stringify(times));
  const ref = document.querySelector("input[name=ref]:checked");
  fd.append("reference_index", ref ? ref.value : Math.floor(files.length / 2));

  try {
    const resp = await fetch("/api/merge", { method: "POST", body: fd });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || `合成失败（HTTP ${resp.status}）`);
    showResults(data);
  } catch (err) {
    $("error").textContent = err.message;
    $("error").hidden = false;
  } finally {
    $("loading").hidden = true;
    validate();
  }
});

function showResults(data) {
  const id = data.job_id;
  const bust = `?t=${Date.now()}`;
  $("ref-img").src = `/api/jobs/${id}/reference${bust}`;
  $("result-img").src = `/api/jobs/${id}/result${bust}`;
  $("mask-img").src = `/api/jobs/${id}/mask${bust}`;
  $("dl-result").href = `/api/jobs/${id}/result`;
  $("dl-mask").href = `/api/jobs/${id}/mask`;
  const m = data.meta;
  $("meta").textContent =
    `尺寸 ${m.width}×${m.height} ｜ 有效重叠 ${(m.overlap * 100).toFixed(1)}% ｜ ` +
    `动态区域 ${(m.dynamic * 100).toFixed(1)}% ｜ 不确定像素 ${(m.uncertain * 100).toFixed(1)}%`;
  $("results").hidden = false;
  $("results").scrollIntoView({ behavior: "smooth" });
}
