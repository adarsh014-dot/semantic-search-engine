"use strict";
/* Semantic Search UI. Plain JS, no build step. All text from the server is inserted with
   textContent (never innerHTML), so document content can't inject markup. */

const $ = (id) => document.getElementById(id);
const el = (tag, props = {}, ...kids) => {
  const n = Object.assign(document.createElement(tag), props);
  kids.flat().forEach((k) => n.append(k));
  return n;
};

const state = { pending: [], docs: [], topK: 5, maxBytes: 25 * 1024 * 1024, allowed: [".pdf", ".txt", ".docx"], pollTimer: null, busy: false };

/* ---------- API helper ---------- */
async function api(path, options = {}) {
  let res;
  try { res = await fetch(path, options); }
  catch { throw new Error("Cannot reach the server. Is the backend running?"); }
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    const err = new Error(data?.error?.message || `Request failed (${res.status}).`);
    err.code = data?.error?.code; err.status = res.status;
    throw err;
  }
  return data;
}

const fmtSize = (b) => (b >= 1048576 ? (b / 1048576).toFixed(1) + " MB" : Math.max(1, Math.round(b / 1024)) + " KB");
const ext = (name) => ("." + name.split(".").pop()).toLowerCase();

/* ---------- header / health ---------- */
async function loadHealth() {
  try {
    const h = await api("/api/health");
    state.maxBytes = h.max_upload_mb * 1048576; state.allowed = h.allowed_extensions;
    $("maxMb").textContent = h.max_upload_mb;
    if (!state.kSet) { setTopK(h.default_top_k); state.kSet = true; }
    const chip = $("modelChip"); chip.textContent = "";
    chip.append(el("b", { textContent: h.embedding_model }),
      ` — ${h.indexed_chunks} chunk${h.indexed_chunks === 1 ? "" : "s"} indexed` + (h.dimension ? `, ${h.dimension}-dim` : ""));
  } catch (e) { $("modelChip").textContent = e.message; }
}

/* ---------- upload ---------- */
function addFiles(fileList) {
  $("uploadNote").textContent = "";
  for (const file of fileList) {
    let error = null;
    if (!state.allowed.includes(ext(file.name))) error = "Unsupported file type. Please upload PDF, TXT, or DOCX.";
    else if (file.size === 0) error = "The file is empty.";
    else if (file.size > state.maxBytes) error = `File is larger than the ${state.maxBytes / 1048576} MB limit.`;
    else if (state.pending.some((p) => p.file.name === file.name && p.file.size === file.size && !p.error)) continue;
    state.pending.push({ file, error, status: error ? "error" : "ready", progress: 0, message: error || fmtSize(file.size) });
  }
  renderPending();
}

function renderPending() {
  const list = $("pendingList"); list.replaceChildren();
  state.pending.forEach((p, i) => {
    const cls = p.status === "error" ? "err" : p.status === "done" ? "ok" : "";
    const li = el("li", {},
      el("span", { className: "name", textContent: p.file.name, title: p.file.name }),
      el("button", { className: "x", type: "button", textContent: "✕", title: "Remove", disabled: state.busy,
        onclick: () => { state.pending.splice(i, 1); renderPending(); } }),
      el("span", { className: "state " + cls, textContent: p.message }));
    if (p.status === "uploading") li.append(el("div", { className: "bar" }, el("i", { style: `width:${p.progress}%` })));
    list.append(li);
  });
  const ready = state.pending.filter((p) => p.status === "ready").length;
  $("uploadBtn").hidden = ready === 0;
  $("uploadBtn").disabled = state.busy;
  $("uploadBtn").textContent = ready > 1 ? `Upload and index ${ready} files` : "Upload and index";
}

// XMLHttpRequest because fetch() cannot report upload progress.
function uploadOne(p) {
  return new Promise((resolve) => {
    const xhr = new XMLHttpRequest(); const fd = new FormData();
    fd.append("files", p.file);
    xhr.open("POST", "/api/documents/upload");
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) { p.progress = (e.loaded / e.total) * 100; renderPending(); } };
    xhr.onerror = () => resolve({ status: "rejected", message: "Cannot reach the server." });
    xhr.onload = () => {
      try {
        const body = JSON.parse(xhr.responseText);
        resolve(xhr.status === 200 ? body.results[0] : { status: "rejected", message: body?.error?.message || "Upload failed." });
      } catch { resolve({ status: "rejected", message: "Upload failed." }); }
    };
    xhr.send(fd);
  });
}

async function uploadAll() {
  state.busy = true; const newIds = [];
  for (const p of state.pending.filter((x) => x.status === "ready")) {
    p.status = "uploading"; p.message = "Uploading…"; renderPending();
    const r = await uploadOne(p);
    if (r.status === "uploaded") { p.status = "done"; p.message = "Uploaded. Processing…"; newIds.push(r.document.id); }
    else { p.status = "error"; p.message = r.message; }
    renderPending();
  }
  state.busy = false;
  if (newIds.length) {
    try { await api("/api/documents/index", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ document_ids: newIds }) }); }
    catch (e) { $("uploadNote").textContent = e.message; }
  }
  await loadDocs();
  setTimeout(() => { state.pending = state.pending.filter((p) => p.status !== "done"); renderPending(); }, 2500);
}

/* ---------- documents ---------- */
async function loadDocs() {
  try { state.docs = await api("/api/documents"); renderDocs(); loadHealth(); }
  catch (e) { $("docs").replaceChildren(el("p", { className: "empty", textContent: e.message })); }
  clearTimeout(state.pollTimer);
  if (state.docs.some((d) => d.status === "processing")) state.pollTimer = setTimeout(loadDocs, 1200);
}

function renderDocs() {
  const box = $("docs"); box.replaceChildren();
  $("reindexAll").hidden = state.docs.length === 0;
  if (!state.docs.length) {
    box.append(el("p", { className: "empty" }, el("strong", { textContent: "No documents indexed yet." }), "Upload a document to begin semantic search."));
    return;
  }
  for (const d of state.docs) {
    const busy = d.status === "processing";
    box.append(el("div", { className: "doc" },
      el("div", { className: "doc-top" },
        el("span", { className: "doc-name", textContent: d.filename }),
        el("span", { className: "badge " + d.status, textContent: { uploaded: "Not indexed", processing: "Processing", indexed: "Indexed", failed: "Failed" }[d.status] })),
      el("div", { className: "doc-meta" },
        el("span", {}, "Type ", el("b", { textContent: d.file_type.toUpperCase() })),
        el("span", {}, "Pages ", el("b", { textContent: d.pages ?? "n/a" })),
        el("span", {}, "Chunks ", el("b", { textContent: d.chunks }))),
      d.error ? el("p", { className: "doc-err", textContent: d.error }) : "",
      el("div", { className: "doc-actions" },
        el("button", { className: "btn small", type: "button", textContent: d.status === "indexed" ? "Re-index" : "Index", disabled: busy, onclick: () => processDoc(d, d.status === "indexed") }),
        el("button", { className: "btn small danger", type: "button", textContent: "Delete", disabled: busy, onclick: () => deleteDoc(d) }))));
  }
}

async function processDoc(d, reindex) {
  try { await api(`/api/documents/${reindex ? "reindex" : "index"}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ document_ids: [d.id] }) }); }
  catch (e) { $("uploadNote").textContent = e.message; }
  loadDocs();
}

async function deleteDoc(d) {
  if (!confirm(`Delete "${d.filename}" and remove it from the search index?`)) return;
  try { await api(`/api/documents/${d.id}`, { method: "DELETE" }); $("results").replaceChildren(); setStatus(""); }
  catch (e) { $("uploadNote").textContent = e.message; }
  loadDocs();
}

/* ---------- search ---------- */
function setTopK(k) {
  state.topK = k; state.kSet = true;
  document.querySelectorAll(".topk button").forEach((b) => b.setAttribute("aria-checked", String(+b.dataset.k === k)));
}
function setStatus(msg, isErr = false) { const s = $("searchStatus"); s.textContent = msg; s.classList.toggle("err", isErr); }

async function runSearch(e) {
  e.preventDefault();
  const query = $("query").value.trim(); if (!query) return;
  const out = $("results"); out.replaceChildren(); setStatus("Searching…"); $("searchBtn").disabled = true;
  try {
    const data = await api("/api/search", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query, top_k: state.topK }) });
    if (!data.results.length) {
      setStatus("");
      out.append(el("div", { className: "placeholder" }, el("strong", { textContent: "No sufficiently relevant results found." }), "Try changing your query."));
    } else {
      setStatus(`${data.count} result${data.count === 1 ? "" : "s"} found in ${Math.round(data.took_ms)} ms`);
      data.results.forEach((r, i) => out.append(resultCard(r, i)));
    }
  } catch (err) {
    setStatus(err.message, true);
  } finally { $("searchBtn").disabled = false; }
}

function resultCard(r) {
  const pct = Math.max(0, r.score) * 100;
  const body = el("div", { className: "body" },
    el("div", { className: "src" },
      el("span", { className: "file", textContent: r.source }),
      el("span", { className: "page", textContent: r.page ? `Page ${r.page}` : "No page numbers in this format" })),
    el("p", { className: "snippet", textContent: r.text }));
  const btn = el("button", { className: "btn small view", type: "button", textContent: "View source" });
  let panel = null;
  btn.onclick = async () => {
    if (panel) { panel.remove(); panel = null; btn.textContent = "View source"; return; }
    btn.disabled = true;
    try {
      const c = await api(`/api/documents/${r.document_id}/chunks/${r.chunk_index}`);
      const block = (title, chunk, hit) => chunk ? el("div", { className: "ctx" + (hit ? " hit" : "") }, el("h4", { textContent: title }), chunk.text) : "";
      panel = el("div", { className: "src-panel" },
        el("div", { className: "doc-meta" },
          el("span", {}, "File ", el("b", { textContent: c.document.filename })),
          el("span", {}, "Chunk ", el("b", { textContent: `${c.chunk.chunk_index + 1} of ${c.document.chunks}` })),
          el("span", {}, "Cosine similarity ", el("b", { textContent: r.score.toFixed(4) }))),
        block("Previous chunk", c.previous), block("Matched chunk", c.chunk, true), block("Next chunk", c.next));
      body.append(panel); btn.textContent = "Hide source";
    } catch (err) { setStatus(err.message, true); }
    btn.disabled = false;
  };
  body.append(btn);
  return el("article", { className: "result" },
    el("div", { className: "meter", style: `--fill:${Math.min(100, pct).toFixed(1)}%`, title: `Cosine similarity ${r.score.toFixed(4)}` },
      el("b", { textContent: pct.toFixed(1) + "%" }), el("small", { textContent: "similarity" })),
    body);
}

function showInitialResults() {
  $("results").append(el("div", { className: "placeholder" }, el("strong", { textContent: "Search by meaning, not keywords." }),
    "Ask a question in plain language. Results are the document passages whose meaning is closest to it."));
}

/* ---------- wiring ---------- */
const drop = $("drop"), input = $("fileInput");
$("browse").onclick = (e) => { e.stopPropagation(); input.click(); };
drop.onclick = () => input.click();
drop.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } };
input.onchange = () => { addFiles(input.files); input.value = ""; };
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => addFiles(e.dataTransfer.files));
$("uploadBtn").onclick = uploadAll;
$("searchForm").onsubmit = runSearch;
document.querySelectorAll(".topk button").forEach((b) => (b.onclick = () => setTopK(+b.dataset.k)));
$("reindexAll").onclick = async () => {
  if (!confirm("Rebuild the whole search index from the original files? This re-embeds every document.")) return;
  try { await api("/api/documents/reindex", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" }); }
  catch (e) { $("uploadNote").textContent = e.message; }
  loadDocs();
};

showInitialResults();
loadHealth().then(loadDocs);
