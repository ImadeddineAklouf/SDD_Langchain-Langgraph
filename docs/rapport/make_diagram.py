"""Génère graph.svg (graphe SDD + Gardien) et guardian.svg (graphe interne du Gardien)."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
K = {"det": ("#e6edf5", "#4f6f91"), "llm": ("#fdebd8", "#c9731d"), "hum": ("#e0f1e3", "#3b8a4a"),
     "grd": ("#f3e6f7", "#7a3b8f"), "end": ("#12304d", "#12304d")}
HEAD = ('<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
        '<path d="M0,0 L10,5 L0,10 z" fill="#555"/></marker>'
        '<marker id="p" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
        '<path d="M0,0 L10,5 L0,10 z" fill="#7a3b8f"/></marker></defs>')


def box(o, x, y, w, h, kind, label, rx=5):
    fill, stroke = K[kind]
    o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    o.append(f'<text x="{x + w/2}" y="{y + h/2 + 4}" text-anchor="middle" fill="{"#fff" if kind == "end" else "#1d232b"}" '
             f'font-family="DejaVu Sans Mono, monospace" font-weight="bold">{label}</text>')


def arrow(o, x1, y1, x2, y2, color="#555", marker="a", dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    o.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.3"{d} marker-end="url(#{marker})"/>')


def text(o, x, y, s, color="#444", size=10.5, anchor="start", weight="normal"):
    o.append(f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">{s}</text>')


# ------------------------------------------------------------------ graph.svg
rows = [("START", "end", ""), ("constitution", "det", "découpe la constitution (C-001…)"),
        ("specify", "llm", "découpe le brief (B-001…), extrait REQ-xxx"),
        ("clarify", "llm", "questions Q-xxx uniquement"),
        ("plan", "llm", "PLN-xxx, technos sourcées"),
        ("tasks", "llm", "T-xxx, ordre topologique"),
        ("analyze", "det", "contrôle croisé (rapport)"),
        ("implement", "llm", "une tâche, fichiers déclarés"),
        ("apply_impl", "det", "écrit les fichiers"),
        ("verify", "det", "pytest ⇄ fix (LLM, borné)"),
        ("END", "end", "")]
guarded = {"clarify", "plan", "analyze", "implement", "verify"}
W, X, BW, BH, GAP, TOP = 760, 30, 130, 28, 44, 14
y = {n: TOP + i * GAP for i, (n, _, _) in enumerate(rows)}
H = TOP + len(rows) * GAP + 30
o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="DejaVu Sans, sans-serif" font-size="11">', HEAD]
GX, GW = 470, 120
gy1, gy2 = y["clarify"] - 6, y["verify"] + BH + 6
o.append(f'<rect x="{GX}" y="{gy1}" width="{GW}" height="{gy2 - gy1}" rx="8" fill="#f3e6f7" stroke="#7a3b8f" stroke-width="1.8"/>')
text(o, GX + GW / 2, gy1 + 22, "guardian", "#3d1a49", 12, "middle", "bold")
text(o, GX + GW / 2, gy1 + 38, "(graphe séparé)", "#7a3b8f", 10, "middle")
for i, line in enumerate(["contrôles", "déterministes", "+ sémantiques", "(LLM, facultatif)", "", "rapport :", "⛔ bloquante",
                          "⚠️ avertissement", "ℹ️ info"]):
    text(o, GX + GW / 2, gy1 + 70 + i * 15, line, "#3d1a49", 10, "middle")
AX = 640
ay = (gy1 + gy2) / 2 - 22
box(o, AX - 30, ay, 140, 44, "hum", "gate (pause)")
text(o, AX + 40, ay + 60, "interrupt() : l'humain", "#3b8a4a", 10, "middle")
text(o, AX + 40, ay + 74, "répond, corrige, accepte", "#3b8a4a", 10, "middle")
text(o, AX + 40, ay + 88, "le risque ou arrête", "#3b8a4a", 10, "middle")
arrow(o, GX + GW, ay + 22, AX - 31, ay + 22, "#7a3b8f", "p")
text(o, GX + GW + 4, ay + 14, "alerte", "#7a3b8f", 10)
ry = y["specify"] - 8
o.append(f'<path d="M{AX + 40},{ay} V{ry} H{X + BW - 12} V{y["specify"] - 1}" fill="none" stroke="#3b8a4a" '
         f'stroke-width="1.3" stroke-dasharray="4 3" marker-end="url(#a)"/>')
text(o, AX + 36, ry - 4, "reprise au nœud fautif (specify, plan, tasks, implement ou fix)", "#3b8a4a", 10, "end")
for i, (n, k, note) in enumerate(rows):
    small = k == "end"
    w = 80 if small else BW
    box(o, X + (BW - w) / 2, y[n], w, BH, k, n, 14 if small else 5)
    if note:
        text(o, X + BW + 10, y[n] + 18, note)
    if n in guarded:
        arrow(o, X + BW + 230, y[n] + BH / 2, GX - 1, y[n] + BH / 2, "#7a3b8f", "p")
    if i < len(rows) - 1:
        nxt = rows[i + 1][0]
        guard = n in guarded and nxt != "END"
        arrow(o, X + BW / 2, y[n] + BH, X + BW / 2, y[nxt] - 1, "#7a3b8f" if guard else "#555", "p" if guard else "a",
              "5 3" if guard else "")
text(o, X, H - 10, "Flèches violettes : chaque étape productrice est contrôlée par le Gardien ; on ne passe à l'étape "
     "suivante (pointillés violets) que s'il laisse passer.", "#555", 10)
o.append("</svg>")
(HERE / "graph.svg").write_text("\n".join(o), encoding="utf-8")

# ------------------------------------------------------------------ guardian.svg
o = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 152" font-family="DejaVu Sans, sans-serif" font-size="11">', HEAD]
box(o, 10, 55, 70, 28, "end", "START", 14)
box(o, 120, 50, 140, 38, "det", "deterministic")
box(o, 330, 10, 120, 38, "llm", "semantic")
box(o, 520, 50, 140, 38, "grd", "consolidate")
box(o, 700, 55, 52, 28, "end", "END", 14)
arrow(o, 80, 69, 119, 69)
arrow(o, 260, 62, 329, 32)
arrow(o, 260, 72, 519, 72)
arrow(o, 450, 30, 519, 60)
arrow(o, 660, 69, 699, 69)
text(o, 322, 22, "rien de bloquant", "#c9731d", 9.5, "end")
text(o, 300, 88, "bloquant, ou pas de LLM : on saute", "#555", 9.5)
text(o, 190, 112, "code pur : citations, couverture,", "#4f6f91", 9.5, "middle")
text(o, 190, 125, "questions, fichiers, dépendances…", "#4f6f91", 9.5, "middle")
text(o, 390, 64, "ne peut qu'ajouter", "#c9731d", 9.5, "middle")
text(o, 590, 112, "dédoublonne, rétrograde", "#7a3b8f", 9.5, "middle")
text(o, 590, 125, "les risques acceptés, trie", "#7a3b8f", 9.5, "middle")
text(o, 10, 145, "État du Gardien : stage, artifacts (instantané), accepted ; alerts = Annotated[list, operator.add] ; report.",
     "#555", 9.5)
o.append("</svg>")
(HERE / "guardian.svg").write_text("\n".join(o), encoding="utf-8")
