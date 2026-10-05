"use strict";
// All text from the model or the notes is inserted with textContent, never innerHTML.
const $ = (id) => document.getElementById(id);

function el(tag, props = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") n.className = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v);
  }
  for (const kid of kids) n.append(kid);
  return n;
}

async function api(path, opts = {}) {
  const init = { method: opts.method || "GET", headers: {} };
  if (opts.json !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(opts.json);
  }
  if (opts.form) init.body = opts.form;
  let res;
  try {
    res = await fetch(path, init);
  } catch {
    throw new Error("Saathi's server is not running. Start it again with: python -m saathi");
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || (data.errors && data.errors[0]) || `Something went wrong (${res.status}).`);
  return data;
}

function waitText(sec) {
  if (sec == null) return "";
  if (sec < 90) return "in a minute or two";
  if (sec < 5400) return `in about ${Math.round(sec / 60)} minutes`;
  if (sec < 172800) return `in about ${Math.round(sec / 3600)} hours`;
  return `in about ${Math.round(sec / 86400)} days`;
}

// ------------------------------------------------------------------ tabs
let currentTab = "study";
function showTab(name) {
  currentTab = name;
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  document.querySelectorAll(".view").forEach((v) => (v.hidden = v.id !== `tab-${name}`));
  if (name === "study") loadStudy();
  if (name === "notes") loadNotes();
  if (name === "progress") loadProgress();
}
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

// ---------------------------------------------------------------- margin
async function refreshMargin() {
  try {
    const [s, p] = await Promise.all([api("/api/stats"), api("/api/profile")]);
    $("m-due").firstElementChild.textContent = s.due_now;
    $("m-streak").firstElementChild.textContent = s.streak;
    $("m-for").textContent = p.friend_name ? `Made for ${p.friend_name}${p.goal ? `, working toward ${p.goal}` : ""}.` : "";
  } catch { /* the banner on the study tab explains server problems */ }
}

const features = { voice: false };

async function refreshEngine() {
  try {
    const h = await api("/api/health");
    const e = $("engine");
    $("web-wrap").hidden = !h.web;
    features.voice = !!h.voice;
    if (h.llm === "mock") {
      e.textContent = "Demo mode: no model running";
      e.className = "engine demo";
    } else if (h.llm === "cloud") {
      e.textContent = `${h.model}, hosted model`;
      e.className = "engine";
    } else {
      e.textContent = `${h.model}, running on this computer`;
      e.className = "engine";
    }
  } catch { /* ignore */ }
}

// ----------------------------------------------------------------- study
let card = null, confident = false, answered = false;

function setConfidence(v) {
  confident = v;
  $("conf-yes").classList.toggle("on", v);
  $("conf-no").classList.toggle("on", !v);
  $("conf-yes").setAttribute("aria-pressed", String(v));
  $("conf-no").setAttribute("aria-pressed", String(!v));
}
$("conf-yes").onclick = () => setConfidence(true);
$("conf-no").onclick = () => setConfidence(false);

async function makeQuestions(btn, docId) {
  btn.disabled = true;
  const old = btn.textContent;
  btn.textContent = "Writing questions… a laptop can take a minute";
  try {
    const r = await api("/api/generate", { method: "POST", json: { doc_id: docId, max_chunks: 4, per_chunk: 2 } });
    btn.textContent = old;
    btn.disabled = false;
    await refreshMargin();
    if (currentTab === "study") loadStudy();
    return r;
  } catch (e) {
    btn.textContent = old;
    btn.disabled = false;
    alert(e.message);
  }
}

async function loadStudy() {
  const empty = $("study-empty"), box = $("study-card");
  let n;
  try {
    n = await api("/api/next");
  } catch (e) {
    box.hidden = true; empty.hidden = false;
    empty.replaceChildren(el("p", {}, e.message));
    return;
  }
  refreshMargin();
  if (!n.card) {
    box.hidden = true; empty.hidden = false;
    const notes = await api("/api/notes");
    empty.replaceChildren();
    if (!notes.length) {
      empty.append(el("h2", {}, "Start with your notes"),
        el("p", {}, "Saathi writes questions from your own notes, so nothing is learned from the internet. Add a chapter to begin."),
        el("button", { class: "primary", onclick: () => showTab("notes") }, "Add notes"));
    } else if (!n.total) {
      const b = el("button", { class: "primary" }, "Make questions from my notes");
      b.onclick = () => makeQuestions(b);
      empty.append(el("h2", {}, "Your notes are ready"),
        el("p", {}, "Next step: turn them into questions."), b);
    } else {
      const b = el("button", { class: "primary" }, "Make more questions");
      b.onclick = () => makeQuestions(b);
      empty.append(el("h2", {}, "All done for now"),
        el("p", {}, `Your next card is due ${waitText(n.next_due_in)}. Reviewing at the right moment is what makes it stick.`),
        b);
    }
    return;
  }
  empty.hidden = true; box.hidden = false;
  card = n.card; answered = false;
  $("result").hidden = true;
  $("q").textContent = card.question;
  const list = $("opts");
  list.replaceChildren();
  card.options.forEach((text, i) => {
    const b = el("button", { class: "opt", "data-i": i, onclick: () => choose(i) },
      el("span", { class: "key" }, "abcd"[i] + "."), el("span", { class: "txt" }, text));
    list.append(el("li", {}, b));
  });
}

async function choose(i) {
  if (answered || !card) return;
  answered = true;
  const btns = [...document.querySelectorAll(".opt")];
  btns.forEach((b) => (b.disabled = true));
  let r;
  try {
    r = await api("/api/answer", { method: "POST", json: { card_id: card.id, chosen: i, confident } });
  } catch (e) {
    answered = false; btns.forEach((b) => (b.disabled = false)); alert(e.message); return;
  }
  btns[r.correct_index].classList.add("right");
  if (!r.correct) btns[i].classList.add("wrong");
  const v = $("r-line");
  v.className = "verdict " + (r.correct ? "good" : "bad");
  v.textContent = r.correct ? (confident ? "Right, and you knew it." : "Right. Good guess, now make it a sure thing.")
    : r.confident_but_wrong ? "Not this one. You felt sure, so this is worth fixing." : "Not this one.";
  $("r-expl").textContent = r.explanation;
  $("r-next").textContent = r.next_review_days >= 1
    ? `Saathi will bring this back in ${Math.round(r.next_review_days)} day${Math.round(r.next_review_days) === 1 ? "" : "s"}.`
    : "Saathi will bring this back in a few minutes.";
  const why = $("r-why");
  why.hidden = true;
  $("result").hidden = false;
  $("next").focus();
  refreshMargin();
  if (!r.correct) {
    why.textContent = "Working out where it slipped…";
    why.hidden = false;
    try {
      const d = await api(`/api/attempts/${r.attempt_id}/diagnose`, { method: "POST" });
      if (d.why) why.textContent = d.why; else why.hidden = true;
    } catch { why.hidden = true; }
  }
}
$("next").onclick = loadStudy;

document.addEventListener("keydown", (e) => {
  if (currentTab !== "study" || e.target.closest("input, textarea, dialog") || e.ctrlKey || e.metaKey) return;
  if (!answered && "1234".includes(e.key) && e.key) { const i = Number(e.key) - 1; if (card && i < card.options.length) choose(i); }
  else if (e.key.toLowerCase() === "s" && !answered) setConfidence(!confident);
  else if (e.key === "Enter" && answered && document.activeElement !== $("next")) loadStudy();
});

// ----------------------------------------------------------------- notes
async function loadNotes() {
  const ul = $("note-list");
  let docs;
  try { docs = await api("/api/notes"); } catch (e) { ul.replaceChildren(el("li", {}, e.message)); return; }
  ul.replaceChildren(...docs.map((d) => {
    const gen = el("button", { class: "link" }, "Make questions");
    gen.onclick = async () => { const r = await makeQuestions(gen, d.id); if (r) { $("note-msg").textContent = `Added ${r.added} question${r.added === 1 ? "" : "s"}.`; loadNotes(); } };
    const del = el("button", { class: "link", onclick: async () => {
      if (!confirm(`Delete "${d.title}" and its questions?`)) return;
      await api(`/api/notes/${d.id}`, { method: "DELETE" }); loadNotes(); refreshMargin();
    } }, "Delete");
    return el("li", {}, el("span", { class: "t" }, d.title),
      el("span", { class: "small" }, `${d.cards} question${d.cards === 1 ? "" : "s"}`), gen, del);
  }));
}

$("note-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const msg = $("note-msg"), file = $("n-file").files[0];
  msg.textContent = "Saving…";
  try {
    let r;
    if (file) {
      const f = new FormData();
      f.append("file", file); f.append("title", $("n-title").value);
      r = await api("/api/notes", { method: "POST", form: f });
    } else {
      r = await api("/api/notes", { method: "POST", json: { title: $("n-title").value, text: $("n-text").value } });
    }
    msg.textContent = `Saved "${r.title}" in ${r.chunks} part${r.chunks === 1 ? "" : "s"}${r.embedded ? " (semantic search on)" : ""}. Now choose Make questions.`;
    $("note-form").reset();
    loadNotes();
  } catch (err) { msg.textContent = err.message; }
});

// ------------------------------------------------------------------- ask
$("ask-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const out = $("ask-out"), btn = e.submitter;
  btn.disabled = true;
  out.replaceChildren(el("p", { class: "small" }, "Reading your notes…"));
  try {
    const r = await api("/api/ask", { method: "POST", json: { question: $("ask-q").value, web: $("ask-web").checked } });
    out.replaceChildren(el("p", { class: "answer" }, r.answer),
      ...r.sources.map((s) => {
        const safe = s.url && /^https?:\/\//.test(s.url);
        const head = safe ? el("a", { href: s.url, target: "_blank", rel: "noopener noreferrer" }, `${s.title} (web)`) : `${s.title}`;
        return el("div", { class: "src" }, `[${s.n}] `, head, `: ${s.text}${safe ? "" : "…"}`);
      }));
  } catch (err) { out.replaceChildren(el("p", {}, err.message)); }
  btn.disabled = false;
});

// -------------------------------------------------------------- progress
async function loadProgress() {
  const box = $("progress");
  let s;
  try { s = await api("/api/stats"); } catch (e) { box.replaceChildren(el("p", {}, e.message)); return; }
  const kids = [];
  if (!s.answered_7d) {
    kids.push(el("p", {}, "Nothing to show yet. Answer a few questions and this page fills in."));
  } else {
    kids.push(el("p", { class: "stat" }, "This week you answered ", el("b", {}, String(s.answered_7d)),
      " questions and got ", el("b", {}, `${s.accuracy_7d}%`), ` right, over ${s.days_active_7d} day${s.days_active_7d === 1 ? "" : "s"}.`));
    if (s.confident_wrong_7d) kids.push(el("p", {}, `${s.confident_wrong_7d} of your mistakes were answers you felt sure about. Those are the most useful ones to fix.`));
    if (s.best_topic) kids.push(el("p", {}, `Strongest topic: ${s.best_topic}.`));
  }
  if (s.top_misconceptions.length) {
    kids.push(el("h2", {}, "What keeps tripping you up"));
    kids.push(el("ul", { class: "mis" }, ...s.top_misconceptions.map((m) =>
      el("li", {}, `${m.tag}, ${m.count} time${m.count === 1 ? "" : "s"}`, m.example_question ? el("span", { class: "ex" }, m.example_question) : ""))));
  }
  kids.push(el("p", { class: "small" }, `${s.total_cards} questions in total. ${s.mature_cards} are well learned (not due for 6 days or more).`));
  box.replaceChildren(...kids);
}

// ---------------------------------------------------------------- letter
$("write-letter").onclick = async (e) => {
  const btn = e.currentTarget, out = $("letter");
  btn.disabled = true; out.hidden = false; out.textContent = "Writing…";
  try {
    const r = await api("/api/letter", { method: "POST" });
    out.textContent = r.letter; $("print-letter").hidden = false;
    $("listen-letter").hidden = !features.voice;
  } catch (err) { out.textContent = err.message; }
  btn.disabled = false;
};
$("print-letter").onclick = () => window.print();
$("listen-letter").onclick = async (e) => {
  const btn = e.currentTarget;
  btn.disabled = true;
  try {
    const res = await fetch("/api/speak", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: $("letter").textContent }) });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error || "Could not read the letter aloud.");
    const audio = $("letter-audio");
    audio.src = URL.createObjectURL(await res.blob());
    await audio.play();
  } catch (err) { alert(err.message); }
  btn.disabled = false;
};

// --------------------------------------------------------------- profile
const dlg = $("profile");
async function openProfile() {
  const p = await api("/api/profile");
  $("p-name").value = p.friend_name; $("p-goal").value = p.goal; $("p-lang").value = p.language_style;
  $("p-tone").value = p.tone; $("p-mot").value = p.motivation;
  dlg.showModal();
}
$("open-profile").onclick = openProfile;
$("p-cancel").onclick = () => dlg.close();
$("profile-form").addEventListener("submit", async () => {
  await api("/api/profile", { method: "POST", json: {
    friend_name: $("p-name").value, goal: $("p-goal").value, language_style: $("p-lang").value,
    tone: $("p-tone").value, motivation: $("p-mot").value } });
  refreshMargin();
});

// ------------------------------------------------------------------ boot
(async function boot() {
  refreshEngine();
  loadStudy();
  try {
    const p = await api("/api/profile");
    if (!p.friend_name) openProfile();
  } catch { /* ignore */ }
})();
