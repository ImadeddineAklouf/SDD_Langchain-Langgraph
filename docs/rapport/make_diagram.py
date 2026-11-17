"""Génère graph.svg : le graphe LangGraph du pipeline."""
from pathlib import Path
K = {"det": ("#e6edf5", "#4f6f91"), "llm": ("#fdebd8", "#c9731d"), "hum": ("#e0f1e3", "#3b8a4a"), "end": ("#12304d", "#12304d")}
rows = [("START", "end", ""), ("constitution", "det", "découpe la constitution en C-001…"),
        ("specify", "llm", "découpe le brief (B-001…), extrait REQ-xxx"),
        ("clarify", "llm", "produit uniquement des questions Q-xxx"),
        ("review_spec", "hum", "répondre / approuver / quitter"),
        ("plan", "llm", "éléments PLN-xxx, technos sourcées"),
        ("review_plan", "hum", "approuver / réviser / quitter"),
        ("tasks", "llm", "tâches T-xxx, ordre topologique"),
        ("analyze", "hum", "contrôle croisé sans LLM + validation"),
        ("implement", "llm", "une tâche à la fois, fichiers déclarés"),
        ("review_impl", "hum", "écrit les fichiers si approuvé"),
        ("verify", "det", "lance pytest dans workspace/"),
        ("review_verify", "hum", "résultat final"),
        ("END", "end", "")]
W, X, BW, BH, GAP, TOP = 700, 250, 150, 30, 46, 16
y = {n: TOP + i * GAP for i, (n, _, _) in enumerate(rows)}
o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {TOP + len(rows) * GAP + 40}" font-family="DejaVu Sans, sans-serif" font-size="11">',
     '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#555"/></marker></defs>']
for i, (n, k, note) in enumerate(rows):
    fill, stroke = K[k]
    small = k == "end"
    w = 80 if small else BW
    x = X + (BW - w) / 2
    o.append(f'<rect x="{x}" y="{y[n]}" width="{w}" height="{BH}" rx="{15 if small else 5}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    o.append(f'<text x="{X + BW/2}" y="{y[n] + 19}" text-anchor="middle" fill="{"#fff" if small else "#1d232b"}" font-family="DejaVu Sans Mono, monospace" font-weight="bold">{n}</text>')
    if note:
        o.append(f'<text x="{X + BW + 14}" y="{y[n] + 19}" fill="#444" font-size="10.5">{note}</text>')
    if i < len(rows) - 1:
        o.append(f'<line x1="{X + BW/2}" y1="{y[n] + BH}" x2="{X + BW/2}" y2="{y[rows[i+1][0]] - 1}" stroke="#555" stroke-width="1.3" marker-end="url(#a)"/>')
lab = {"review_spec": "réviser (réponses → A-xxx)", "review_plan": "réviser", "analyze": "réviser", "review_impl": "réviser / tâche suivante",
       "review_verify": "réviser → fix"}
loops = [("review_spec", "specify", 200), ("review_plan", "plan", 200), ("analyze", "tasks", 200), ("review_impl", "implement", 200)]
for a, b, lx in loops:
    ya, yb = y[a] + BH / 2, y[b] + BH / 2
    o.append(f'<path d="M{X},{ya} H{lx} V{yb} H{X - 1}" fill="none" stroke="#3b8a4a" stroke-width="1.3" stroke-dasharray="4 3" marker-end="url(#a)"/>')
    o.append(f'<text x="{lx - 6}" y="{(ya + yb) / 2 + 4}" text-anchor="end" fill="#3b8a4a" font-size="10">{lab[a]}</text>')
# fix loop, to the left of verify
fx, fy = 40, y["verify"] + GAP / 2
o.append(f'<rect x="{fx}" y="{fy}" width="110" height="{BH}" rx="5" fill="{K["llm"][0]}" stroke="{K["llm"][1]}" stroke-width="1.5"/>')
o.append(f'<text x="{fx + 55}" y="{fy + 19}" text-anchor="middle" font-family="DejaVu Sans Mono, monospace" font-weight="bold">fix</text>')
vy = y["verify"] + BH / 2
o.append(f'<path d="M{X},{vy} H{fx + 55} V{fy - 1}" fill="none" stroke="#c9731d" stroke-width="1.3" marker-end="url(#a)"/>')
o.append(f'<text x="{fx + 62}" y="{vy - 5}" fill="#c9731d" font-size="10">échec, essais restants</text>')
o.append(f'<path d="M{fx + 110},{fy + BH/2} H{X - 20} V{vy + 6} H{X - 1}" fill="none" stroke="#c9731d" stroke-width="1.3" marker-end="url(#a)"/>')
rv = y["review_verify"] + BH / 2
o.append(f'<path d="M{X},{rv} H{fx + 30} V{fy + BH + 1}" fill="none" stroke="#3b8a4a" stroke-width="1.3" stroke-dasharray="4 3" marker-end="url(#a)"/>')
o.append(f'<text x="{X - 8}" y="{rv - 5}" text-anchor="end" fill="#3b8a4a" font-size="10">réviser</text>')
# légende
ly = TOP + len(rows) * GAP + 8
for i, (k, t) in enumerate([("det", "code déterministe (sans LLM)"), ("llm", "appel LLM + validateur"), ("hum", "point de contrôle humain (interrupt)")]):
    lx = 40 + i * 220
    o.append(f'<rect x="{lx}" y="{ly}" width="16" height="12" fill="{K[k][0]}" stroke="{K[k][1]}"/><text x="{lx + 22}" y="{ly + 10}" font-size="10">{t}</text>')
o.append("</svg>")
Path(__file__).with_name("graph.svg").write_text("\n".join(o), encoding="utf-8")
