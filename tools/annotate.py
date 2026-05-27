#!/usr/bin/env python3
"""A simple app for annotating articles."""

import json
import os
import argparse
import sys
import webbrowser
import datetime
from threading import Timer
from flask import Flask, jsonify, render_template_string, request

# Parse args.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_path", 
    type=str, 
    default="./data/raw/mmc.jsonl",
    help="Path to jsonl file containing articles in original format."
)
parser.add_argument(
    "--out_path", 
    type=str, 
    default="./data/raw/annotations.jsonl",
    help="Path to jsonl file where annotations will be stored."
)
args = parser.parse_args()

jsonl_path = args.data_path
out_path = args.out_path

# Load articles.
articles = []
try:
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                articles.append(json.loads(line))
except FileNotFoundError:
    print(f"Error: Could not find {jsonl_path}. Please check your path.")
    sys.exit(1)

if not articles:
    print("No articles found.")
    sys.exit(1)

# Load existing annotations.
annotations: dict[str, dict] = {}  # id -> {locations_explicit, locations_implicit, true_sentiment}
if os.path.exists(out_path):
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                annotations[str(row["id"])] = {
                    # Fallback to true_locations if the old format was used
                    "locations_explicit": row.get("locations_explicit", row.get("true_locations", [])),
                    "locations_implicit": row.get("locations_implicit", []),
                    "true_sentiment": row.get("true_sentiment", ""),
                }

def save_annotation(article_id: str, sentiment: str, locations_explicit: list, locations_implicit: list):
    annotations[article_id] = {
        "true_sentiment": sentiment, 
        "locations_explicit": locations_explicit,
        "locations_implicit": locations_implicit
    }
    # Ensure the directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for aid, data in annotations.items():
            f.write(json.dumps({
                "id": aid, 
                "true_sentiment": data["true_sentiment"],
                "locations_explicit": data["locations_explicit"],
                "locations_implicit": data["locations_implicit"]
            }, ensure_ascii=False) + "\n")

# Flask app.
app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html lang="sl">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Annotator</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #f5f0e8;
      --paper: #fffef9;
      --ink: #1a1612;
      --muted: #6b6560;
      --rule: #e0d8cc;
      --accent: #c0392b;
      --good: #27ae60;
      --bad: #c0392b;
      --neutral: #2980b9;
      --done-bg: #eaf6ed;
      --done-border: #27ae60;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'IBM Plex Sans', sans-serif;
      background: var(--bg);
      color: var(--ink);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    /* ── Top bar ── */
    header {
      background: var(--ink);
      color: #f5f0e8;
      padding: 10px 28px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      position: sticky;
      top: 0;
      z-index: 100;
    }
    header h1 {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.85rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      opacity: 0.9;
    }
    #progress-wrap {
      display: flex;
      align-items: center;
      gap: 10px;
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.78rem;
      opacity: 0.75;
    }
    #progress-bar-outer {
      width: 120px;
      height: 4px;
      background: rgba(255,255,255,0.18);
      border-radius: 2px;
      overflow: hidden;
    }
    #progress-bar-inner {
      height: 100%;
      background: #e8c84a;
      border-radius: 2px;
      transition: width 0.3s;
    }

    /* ── Nav strip ── */
    #nav {
      background: var(--paper);
      border-bottom: 1px solid var(--rule);
      padding: 8px 28px;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    #nav button {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.78rem;
      padding: 5px 14px;
      border: 1px solid var(--rule);
      background: var(--paper);
      cursor: pointer;
      border-radius: 3px;
      color: var(--ink);
      transition: background 0.15s;
    }
    #nav button:hover { background: var(--bg); }
    #nav button:disabled { opacity: 0.35; cursor: default; }
    #nav-info {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.78rem;
      color: var(--muted);
      margin-left: auto;
    }
    #jump-wrap {
      display: flex;
      align-items: center;
      gap: 6px;
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.78rem;
      color: var(--muted);
    }
    #jump-input {
      width: 52px;
      padding: 4px 6px;
      border: 1px solid var(--rule);
      border-radius: 3px;
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.78rem;
      background: var(--bg);
    }

    /* ── Main layout ── */
    main {
      flex: 1;
      display: grid;
      grid-template-columns: 1fr 320px;
      gap: 0;
      max-width: 1200px;
      width: 100%;
      margin: 0 auto;
      padding: 28px 20px;
      align-items: start;
    }

    /* ── Article panel ── */
    #article-panel {
      padding-right: 28px;
    }

    #article-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin-bottom: 14px;
    }
    .meta-tag {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.7rem;
      padding: 3px 8px;
      background: var(--ink);
      color: var(--bg);
      border-radius: 2px;
      letter-spacing: 0.06em;
    }
    .meta-date {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.72rem;
      color: var(--muted);
    }
    .meta-authors {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.72rem;
      color: var(--muted);
    }

    #article-title {
      font-family: 'Lora', serif;
      font-size: 1.7rem;
      font-weight: 600;
      line-height: 1.25;
      margin-bottom: 12px;
      color: var(--ink);
    }
    #article-lead {
      font-family: 'Lora', serif;
      font-style: italic;
      font-size: 1.05rem;
      line-height: 1.6;
      color: var(--muted);
      border-left: 3px solid var(--accent);
      padding-left: 14px;
      margin-bottom: 22px;
    }
    .divider {
      border: none;
      border-top: 1px solid var(--rule);
      margin: 18px 0;
    }
    #article-body p {
      font-family: 'Lora', serif;
      font-size: 1rem;
      line-height: 1.75;
      margin-bottom: 14px;
      color: var(--ink);
    }
    #article-keywords {
      margin-top: 20px;
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .kw {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.68rem;
      padding: 3px 8px;
      border: 1px solid var(--rule);
      border-radius: 20px;
      color: var(--muted);
    }

    /* ── Annotation sidebar ── */
    #annotation-panel {
      position: sticky;
      top: 58px;
      background: var(--paper);
      border: 1px solid var(--rule);
      border-radius: 6px;
      padding: 20px;
      box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    }
    #annotation-panel h2 {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.75rem;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 16px;
    }

    .field-label {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 7px;
      display: block;
    }

    /* sentiment buttons */
    #sentiment-group {
      display: flex;
      flex-direction: column;
      gap: 7px;
      margin-bottom: 20px;
    }
    .sent-btn {
      padding: 10px 12px;
      border: 2px solid var(--rule);
      border-radius: 4px;
      background: var(--paper);
      cursor: pointer;
      font-family: 'IBM Plex Sans', sans-serif;
      font-size: 0.88rem;
      font-weight: 500;
      text-align: left;
      transition: all 0.15s;
      display: flex;
      align-items: center;
      gap: 9px;
    }
    .sent-btn:hover { background: var(--bg); }
    .sent-btn.active-good  { border-color: var(--good);    background: #eaf6ed; }
    .sent-btn.active-bad   { border-color: var(--bad);     background: #fdecea; }
    .sent-btn.active-neutral { border-color: var(--neutral); background: #eaf2fb; }
    .sent-dot {
      width: 10px; height: 10px;
      border-radius: 50%;
      flex-shrink: 0;
    }
    .dot-good    { background: var(--good); }
    .dot-bad     { background: var(--bad); }
    .dot-neutral { background: var(--neutral); }

    /* locations */
    .locations-input {
      width: 100%;
      padding: 9px 10px;
      border: 1px solid var(--rule);
      border-radius: 4px;
      font-family: 'IBM Plex Sans', sans-serif;
      font-size: 0.88rem;
      background: var(--bg);
      color: var(--ink);
      resize: vertical;
      min-height: 80px;
      margin-bottom: 18px;
    }
    .locations-input:focus {
      outline: none;
      border-color: #aaa;
    }

    #save-btn {
      width: 100%;
      padding: 11px;
      background: var(--ink);
      color: #f5f0e8;
      border: none;
      border-radius: 4px;
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.82rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      cursor: pointer;
      transition: opacity 0.15s;
    }
    #save-btn:hover { opacity: 0.82; }

    #save-status {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.72rem;
      text-align: center;
      margin-top: 10px;
      min-height: 18px;
      color: var(--good);
    }

    /* annotated badge */
    #annotated-badge {
      display: none;
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.7rem;
      padding: 4px 10px;
      background: var(--done-bg);
      border: 1px solid var(--done-border);
      color: var(--done-border);
      border-radius: 3px;
      margin-bottom: 14px;
      text-align: center;
    }

    .url-link {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.7rem;
      color: var(--muted);
      word-break: break-all;
      display: block;
      margin-top: 20px;
    }
    .url-link a { color: var(--muted); }

    @media (max-width: 760px) {
      main { grid-template-columns: 1fr; }
      #article-panel { padding-right: 0; margin-bottom: 24px; }
      #annotation-panel { position: static; }
    }
  </style>
</head>
<body>
  <header>
    <h1>📰 Annotator</h1>
    <div id="progress-wrap">
      <div id="progress-bar-outer"><div id="progress-bar-inner"></div></div>
      <span id="progress-text"></span>
    </div>
  </header>

  <div id="nav">
    <button id="btn-prev" onclick="navigate(-1)">← Nazaj</button>
    <button id="btn-next" onclick="navigate(1)">Naprej →</button>
    <button onclick="skipToNext()">Preskoči neanotatiran ↓</button>
    <div id="jump-wrap">
      Pojdi na #
      <input id="jump-input" type="number" min="1" onkeydown="if(event.key==='Enter') jumpTo()">
      <button onclick="jumpTo()">↵</button>
    </div>
    <span id="nav-info"></span>
  </div>

  <main>
    <section id="article-panel">
      <div id="article-meta"></div>
      <h1 id="article-title"></h1>
      <p id="article-lead"></p>
      <hr class="divider">
      <div id="article-body"></div>
      <div id="article-keywords"></div>
      <span id="article-url" class="url-link"></span>
    </section>

    <aside id="annotation-panel">
      <h2>Anotacija</h2>
      <div id="annotated-badge">✔ Že anotirano</div>

      <label class="field-label">Sentiment</label>
      <div id="sentiment-group">
        <button class="sent-btn" id="btn-good" onclick="setSentiment('dobra novica')">
          <span class="sent-dot dot-good"></span> Dobra novica
        </button>
        <button class="sent-btn" id="btn-bad" onclick="setSentiment('slaba novica')">
          <span class="sent-dot dot-bad"></span> Slaba novica
        </button>
        <button class="sent-btn" id="btn-neutral" onclick="setSentiment('nevtralna novica')">
          <span class="sent-dot dot-neutral"></span> Nevtralna novica
        </button>
      </div>

      <label class="field-label" for="locations-explicit-input">Eksplicitne Lokacije (ena na vrstico)</label>
      <textarea id="locations-explicit-input" class="locations-input" placeholder="npr.&#10;Ljubljana&#10;Maribor"></textarea>

      <label class="field-label" for="locations-implicit-input">Implicitne Lokacije (ena na vrstico)</label>
      <textarea id="locations-implicit-input" class="locations-input" placeholder="npr.&#10;Slovenija&#10;Evropska unija"></textarea>

      <button id="save-btn" onclick="saveAnnotation()">💾 Shrani &amp; naprej</button>
      <div id="save-status"></div>
    </aside>
  </main>

  <script>
    const TOTAL = {{ total }};
    const ANNOTATED = {{ annotated | tojson }};  // set of already-annotated IDs

    let currentIndex = 0;
    let currentSentiment = "";
    let annotatedSet = new Set(ANNOTATED);

    // keyboard shortcuts
    document.addEventListener("keydown", e => {
      if (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT") return;
      if (e.key === "ArrowRight") navigate(1);
      if (e.key === "ArrowLeft")  navigate(-1);
      if (e.key === "1") setSentiment("dobra novica");
      if (e.key === "2") setSentiment("slaba novica");
      if (e.key === "3") setSentiment("nevtralna novica");
      if (e.key === "Enter") saveAnnotation();
    });

    async function loadArticle(idx) {
      currentIndex = idx;
      currentSentiment = "";
      document.getElementById("save-status").textContent = "";

      const res = await fetch(`/article/${idx}`);
      const data = await res.json();

      const meta = document.getElementById("article-meta");
      meta.innerHTML = `
        <span class="meta-tag">${data.topics || "—"}</span>
        <span class="meta-date">${data.date || ""}</span>
        <span class="meta-authors">${(data.authors || []).join(", ")}</span>
        <span class="meta-tag" style="background:#555">ID: ${data.id}</span>
      `;

      document.getElementById("article-title").textContent = data.title || "";
      document.getElementById("article-lead").textContent = data.lead || "";

      const body = document.getElementById("article-body");
      body.innerHTML = (data.paragraphs || []).map(p => `<p>${p}</p>`).join("");

      const kw = document.getElementById("article-keywords");
      const words = data.keywords || data.gpt_keywords || [];
      kw.innerHTML = words.map(w => `<span class="kw">${w}</span>`).join("");

      const urlEl = document.getElementById("article-url");
      if (data.url) {
        urlEl.innerHTML = `<a href="${data.url}" target="_blank">${data.url}</a>`;
      }

      // existing annotation?
      const badge = document.getElementById("annotated-badge");
      if (data.existing) {
        badge.style.display = "block";
        setSentiment(data.existing.true_sentiment, false);
        document.getElementById("locations-explicit-input").value =
          (data.existing.locations_explicit || []).join("\\n");
        document.getElementById("locations-implicit-input").value =
          (data.existing.locations_implicit || []).join("\\n");
      } else {
        badge.style.display = "none";
        clearSentimentButtons();
        document.getElementById("locations-explicit-input").value = "";
        document.getElementById("locations-implicit-input").value = "";
      }

      // nav
      document.getElementById("btn-prev").disabled = idx === 0;
      document.getElementById("btn-next").disabled = idx === TOTAL - 1;
      document.getElementById("nav-info").textContent =
        `Člen ${idx + 1} / ${TOTAL}`;

      // progress
      const pct = Math.round((annotatedSet.size / TOTAL) * 100);
      document.getElementById("progress-bar-inner").style.width = pct + "%";
      document.getElementById("progress-text").textContent =
        `${annotatedSet.size} / ${TOTAL} (${pct}%)`;

      window.scrollTo(0, 0);
    }

    function navigate(dir) {
      const next = currentIndex + dir;
      if (next >= 0 && next < TOTAL) loadArticle(next);
    }

    function jumpTo() {
      const v = parseInt(document.getElementById("jump-input").value);
      if (!isNaN(v) && v >= 1 && v <= TOTAL) loadArticle(v - 1);
    }

    function skipToNext() {
      for (let i = currentIndex + 1; i < TOTAL; i++) {
        fetch(`/is_annotated/${i}`).then(r => r.json()).then(d => {
          if (!d.annotated) { loadArticle(i); }
        });
        break;
      }
    }

    function clearSentimentButtons() {
      document.getElementById("btn-good").className = "sent-btn";
      document.getElementById("btn-bad").className = "sent-btn";
      document.getElementById("btn-neutral").className = "sent-btn";
      currentSentiment = "";
    }

    function setSentiment(val, updateUI = true) {
      currentSentiment = val;
      if (!updateUI) return;
      document.getElementById("btn-good").className =
        "sent-btn" + (val === "dobra novica" ? " active-good" : "");
      document.getElementById("btn-bad").className =
        "sent-btn" + (val === "slaba novica" ? " active-bad" : "");
      document.getElementById("btn-neutral").className =
        "sent-btn" + (val === "nevtralna novica" ? " active-neutral" : "");
    }

    async function saveAnnotation() {
      if (!currentSentiment) {
        document.getElementById("save-status").style.color = "#c0392b";
        document.getElementById("save-status").textContent = "⚠ Izberi sentiment!";
        return;
      }
      
      const rawExplicit = document.getElementById("locations-explicit-input").value;
      const locationsExplicit = rawExplicit.split("\\n").map(s => s.trim()).filter(Boolean);

      const rawImplicit = document.getElementById("locations-implicit-input").value;
      const locationsImplicit = rawImplicit.split("\\n").map(s => s.trim()).filter(Boolean);

      const res = await fetch("/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          index: currentIndex,
          true_sentiment: currentSentiment,
          locations_explicit: locationsExplicit,
          locations_implicit: locationsImplicit,
        }),
      });
      const data = await res.json();
      if (data.ok) {
        annotatedSet.add(data.id);
        document.getElementById("save-status").style.color = "#27ae60";
        document.getElementById("save-status").textContent = "✔ Shranjeno!";
        const pct = Math.round((annotatedSet.size / TOTAL) * 100);
        document.getElementById("progress-bar-inner").style.width = pct + "%";
        document.getElementById("progress-text").textContent =
          `${annotatedSet.size} / ${TOTAL} (${pct}%)`;
        // auto-advance
        setTimeout(() => {
          if (currentIndex < TOTAL - 1) navigate(1);
        }, 600);
      }
    }

    // init
    loadArticle(0);
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    annotated_ids = list(annotations.keys())
    return render_template_string(HTML, total=len(articles), annotated=annotated_ids)

@app.route("/article/<int:idx>")
def get_article(idx):
    if idx < 0 or idx >= len(articles):
        return jsonify({"error": "out of range"}), 404
    a = articles[idx]
    
    raw_id = a.get("id")
    if raw_id is None:
        raw_id = a.get("_id", "")
    if isinstance(raw_id, float) and raw_id.is_integer():
        aid = str(int(raw_id))
    else:
        aid = str(raw_id)
        
    date_val = a.get("date", "")
    if isinstance(date_val, (int, float)):
        try:
            date_val = datetime.datetime.fromtimestamp(date_val / 1000).strftime('%Y-%m-%d %H:%M')
        except Exception:
            date_val = str(date_val)
    elif isinstance(date_val, str) and "T" in date_val:
        date_val = date_val.replace("T", " ")[:16]
        
    raw_text = a.get("paragraphs") or a.get("text") or a.get("content") or []
    if isinstance(raw_text, str):
        paragraphs = [p.strip() for p in raw_text.split('\n') if p.strip()]
    elif isinstance(raw_text, list):
        paragraphs = raw_text
    else:
        paragraphs = []

    result = {
        "id": aid,
        "title": a.get("title", ""),
        "lead": a.get("lead", ""),
        "paragraphs": paragraphs,
        "date": date_val,
        "authors": a.get("authors", []),
        "topics": a.get("topics", ""),
        "keywords": a.get("keywords", []) or a.get("gpt_keywords", []),
        "url": a.get("url", ""),
        "existing": annotations.get(aid),
    }
    return jsonify(result)

@app.route("/is_annotated/<int:idx>")
def is_annotated(idx):
    if idx < 0 or idx >= len(articles):
        return jsonify({"annotated": False})
    a = articles[idx]
    raw_id = a.get("id", a.get("_id", ""))
    if isinstance(raw_id, float) and raw_id.is_integer():
        aid = str(int(raw_id))
    else:
        aid = str(raw_id)
    return jsonify({"annotated": aid in annotations})

@app.route("/save", methods=["POST"])
def save():
    data = request.get_json()
    idx = data["index"]
    if idx < 0 or idx >= len(articles):
        return jsonify({"ok": False, "error": "out of range"}), 400
    a = articles[idx]
    raw_id = a.get("id", a.get("_id", ""))
    if isinstance(raw_id, float) and raw_id.is_integer():
        aid = str(int(raw_id))
    else:
        aid = str(raw_id)
    
    save_annotation(aid, data["true_sentiment"], data.get("locations_explicit", []), data.get("locations_implicit", []))
    
    return jsonify({"ok": True, "id": aid})

def open_browser():
    webbrowser.open("http://127.0.0.1:5050")

if __name__ == "__main__":
    print(f"Loaded {len(articles)} articles from {jsonl_path}")
    print(f"Annotations will be saved to {out_path}")
    print(f"Already annotated: {len(annotations)}")
    print("Opening http://127.0.0.1:5050 ...")
    Timer(0.8, open_browser).start()
    app.run(port=5050, debug=False)