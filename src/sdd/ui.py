"""Interface web locale (Streamlit) du pipeline SDD.

Lancement depuis la racine du dépôt :  sdd-ui   (ou : streamlit run src/sdd/ui.py)

L'interface ne fait que remplacer le terminal de `cli.py` : elle envoie le brief et la constitution
au graphe, affiche chaque étape et chaque artefact, et transmet les décisions humaines
(approuver / réviser / arrêter) via `Command(resume=...)`. Les garde-fous de déterminisme
(validateurs, sources citables, points de contrôle) restent ceux du graphe, inchangés.
"""
from __future__ import annotations

import io
import json
import os
import sqlite3
import sys
import uuid
import zipfile
from pathlib import Path

ROOT_OUT = Path(os.getenv("SDD_UI_OUT", "out/ui"))
EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
ARTIFACTS = ["spec.md", "plan.md", "tasks.md", "analyze.md", "verify.md", "sources.json"]


def launch() -> None:
    """Point d'entrée de la commande `sdd-ui`."""
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", str(Path(__file__).resolve()), "--browser.gatherUsageStats=false",
                "--client.toolbarMode=minimal",
                "--runner.magicEnabled=false", *sys.argv[1:]]
    sys.exit(stcli.main())


# --------------------------------------------------------------------------- #
def _example(name: str) -> str:
    p = EXAMPLES / name
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _graph(thread: str, max_retries: int, require_tests: bool):
    """Un graphe par session, avec son checkpointer SQLite (reprise possible, comme la CLI)."""
    import streamlit as st

    from sdd.llm import fallback_name, model_name, provider

    @st.cache_resource(show_spinner=False)
    def build(thread: str, max_retries: int, require_tests: bool, _llm_key: tuple):
        from langgraph.checkpoint.sqlite import SqliteSaver

        from sdd.graph import build_graph
        from sdd.llm import get_fallback_llm, get_llm

        out = ROOT_OUT / thread
        out.mkdir(parents=True, exist_ok=True)
        saver = SqliteSaver(sqlite3.connect(out / "checkpoints.db", check_same_thread=False))
        return build_graph(get_llm(), out, max_retries, checkpointer=saver, require_tests=require_tests,
                           fallback_llm=get_fallback_llm())

    # le graphe est reconstruit si l'on change de fournisseur ou de modèle
    return build(thread, max_retries, require_tests, (provider(), model_name(), fallback_name()))


def _cfg(thread: str) -> dict:
    return {"configurable": {"thread_id": thread}}


def _zip(folder: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(folder.rglob("*")):
            if p.is_file() and p.name != "checkpoints.db" and "__pycache__" not in p.parts:
                z.write(p, p.relative_to(folder))
    return buf.getvalue()


# --------------------------------------------------------------------------- #
def render() -> None:
    import streamlit as st
    from dotenv import load_dotenv

    from langgraph.types import Command

    from sdd.runner import NODE_TO_STEP, STAGE_TITLES, STEPS, advance, summarize

    st.cache_resource(load_dotenv)()  # une seule fois par processus : les choix de la barre latérale priment ensuite
    st.set_page_config(page_title="SDD Pipeline", page_icon="📐", layout="wide")
    ss = st.session_state
    ss.setdefault("thread", None)
    ss.setdefault("log", [])          # [(node, résumé)]
    ss.setdefault("pending", None)    # charge utile de l'interrupt en attente
    ss.setdefault("finished", False)
    ss.setdefault("error", None)

    # ------------------------------ barre latérale ------------------------------
    with st.sidebar:
        st.header("Configuration")
        from sdd.llm import API_KEYS, DEFAULTS, PROVIDERS, fallback_name, gemini_models, ollama_models, provider

        prov = st.selectbox("Fournisseur", PROVIDERS, index=PROVIDERS.index(provider()) if provider() in PROVIDERS else 0,
                            help="gemini / groq : API en ligne gratuites (avec quotas). ollama : LLM local, à installer soi-même.")
        if prov != provider():  # changement de fournisseur : on repart de son modèle par défaut
            os.environ.pop("SDD_MODEL", None)
            os.environ.pop("SDD_FALLBACK_MODEL", None)
        os.environ["SDD_PROVIDER"] = prov
        wanted = os.getenv("SDD_MODEL") or DEFAULTS[prov]
        available = st.cache_data(ttl=600, show_spinner=False)(lambda _key: gemini_models())(
            os.getenv("GOOGLE_API_KEY")) if prov == "gemini" else None
        if available:  # liste réelle des modèles ouverts pour cette clé
            if wanted not in available:
                st.warning(f"`{wanted}` n'est pas disponible pour votre clé : choisissez un modèle ci-dessous.")
                wanted = DEFAULTS[prov] if DEFAULTS[prov] in available else available[0]
            model = st.selectbox("Modèle", available, index=available.index(wanted), key=f"model_{prov}_list")
            fb_options = ["(aucun)"] + [m for m in available if m != model]
            fb = fallback_name() if fallback_name() in fb_options else "(aucun)"
            fb = st.selectbox("Modèle de secours (si quota atteint)", fb_options, index=fb_options.index(fb))
            os.environ["SDD_FALLBACK_MODEL"] = "" if fb == "(aucun)" else fb
        else:
            model = st.text_input("Modèle", value=wanted, key=f"model_{prov}").strip()
            if fallback_name():
                st.caption(f"Modèle de secours : **{fallback_name()}**")
        os.environ["SDD_MODEL"] = model
        if prov == "ollama":
            pulled = ollama_models()
            if pulled is None:
                st.error("Ollama ne répond pas (ou `pip install -e \".[ollama]\"` manquant).")
            elif model not in pulled and f"{model}:latest" not in pulled:
                st.warning(f"Modèle absent. Dans un terminal : `ollama pull {model}`")
                if pulled:
                    st.caption("Modèles disponibles : " + ", ".join(pulled))
            else:
                st.caption("✅ Ollama prêt, modèle disponible localement.")
        key_var = API_KEYS.get(prov)
        if key_var and not os.getenv(key_var):
            st.warning(f"{key_var} absente du fichier .env")
            key = st.text_input(f"{key_var} (gardée en mémoire, jamais écrite sur disque)", type="password")
            if key:
                os.environ[key_var] = key.strip()
                st.rerun()
        max_retries = st.number_input("Corrections automatiques par étape", 0, 6, 3)
        require_tests = st.checkbox("Exiger une tâche de test par exigence fonctionnelle", value=True)
        st.divider()
        st.subheader("Reprendre une session")
        old = st.text_input("Identifiant de session", placeholder="ex. 3f9a1c2b")
        if st.button("Reprendre", disabled=not old):
            ss.update(thread=old.strip(), log=[], finished=False, error=None)
            snap = _graph(ss.thread, max_retries, require_tests).get_state(_cfg(ss.thread))
            if not snap.values:
                ss.error = f"Session {ss.thread} introuvable dans {ROOT_OUT}."
                ss.thread = None
            else:
                ss.pending = snap.interrupts[0].value if snap.interrupts else None
                ss.finished = not snap.next
            st.rerun()
        if ss.thread:
            st.divider()
            st.caption(f"Session en cours : `{ss.thread}`  \nDossier : `{ROOT_OUT / ss.thread}`")
            if st.button("Nouvelle session"):
                ss.update(thread=None, log=[], pending=None, finished=False, error=None)
                st.rerun()

    st.title("📐 SDD Pipeline déterministe")
    st.caption("constitution → specify → clarify → plan → tasks → analyze → implement. "
               "Le modèle transforme vos textes ; il n'a pas le droit d'inventer. Chaque phrase devient une source "
               "citable (C-001, B-001…) et chaque étape est validée par du code avant de vous être montrée.")

    if ss.error:
        st.error(ss.error)

    # Avancement des 7 étapes
    reached = {NODE_TO_STEP.get(n) for n, _ in ss.log}
    cols = st.columns(len(STEPS))
    current = NODE_TO_STEP.get(ss.log[-1][0]) if ss.log else None
    for col, (key, label) in zip(cols, STEPS):
        icon = "✅" if key in reached and (key != current or ss.finished) else ("⏳" if key == current else "○")
        col.markdown(f"**{icon} {label}**")

    def on_update(container):
        def cb(node: str, update: dict) -> None:
            ss.log.append((node, summarize(node, update)))
            container.write(f"✔︎ **{node}** — {ss.log[-1][1]}")
        return cb

    def run(inp, label: str) -> None:
        graph = _graph(ss.thread, max_retries, require_tests)
        with st.status(label, expanded=True) as status:
            try:
                ss.pending = advance(graph, inp, _cfg(ss.thread), on_update(status))
                ss.finished = ss.pending is None
                ss.error = None
                status.update(label="Point de contrôle : à vous de décider" if ss.pending else "Workflow terminé", state="complete")
            except Exception as exc:  # erreur API (clé, modèle, quota, réseau) : la session reste reprenable
                from sdd.llm import explain_error

                ss.error = (f"{explain_error(exc)}\n\nLa session `{ss.thread}` est conservée : corrigez puis cliquez "
                            "sur « Reprendre là où le workflow s'est arrêté ».")
                status.update(label="Erreur", state="error")
        st.rerun()

    def run_resume(review: dict, label: str) -> None:
        run(Command(resume=review), label)

    # ------------------------------ 1. saisie ------------------------------
    if not ss.thread:
        if st.button("Charger l'exemple (gestionnaire de tâches)"):
            ss.brief_txt, ss.const_txt = _example("brief.md"), _example("constitution.md")
            st.rerun()
        with st.form("inputs"):
            brief = st.text_area("Brief : votre besoin, une idée par phrase ou par puce", key="brief_txt", height=260)
            constitution = st.text_area("Constitution (facultative) : règles du projet (langage, tests, dépendances…)",
                                        key="const_txt", height=160)
            start = st.form_submit_button("▶ Lancer le workflow SDD", type="primary")
        if start and not brief.strip():
            st.warning("Le brief est vide.")
        elif start:
            ss.update(thread=uuid.uuid4().hex[:8], log=[], pending=None, finished=False, error=None)
            run({"brief": brief, "constitution": constitution}, "Exécution : constitution → specify → clarify…")
        return

    # ------------------------------ 2. point de contrôle ------------------------------
    p = ss.pending
    if p:
        stage = p["stage"]
        st.subheader("⏸ " + STAGE_TITLES.get(stage, stage))
        if p.get("file") and Path(p["file"]).exists():
            with st.expander(f"Artefact : {Path(p['file']).name}", expanded=True):
                st.markdown(Path(p["file"]).read_text(encoding="utf-8"))
        if stage == "impl":
            st.caption(f"Tâche **{p['task_id']}** — fichiers écrits seulement si vous approuvez.")
            for path, content in p.get("preview", {}).items():
                with st.expander(path, expanded=True):
                    st.code(content, language="python" if path.endswith(".py") else None)
        if "passed" in p:
            if p["passed"]:
                st.success("Tests : OK")
            else:
                st.error("Tests : échec")
        for e in p.get("errors", []):
            st.error(e)
        for w in p.get("warnings", []):
            st.warning(w)

        with st.form(f"review-{stage}-{len(ss.log)}"):
            answers = {}
            qs = p.get("open_questions", [])
            if qs:
                st.markdown("**Questions ouvertes** : vos réponses deviennent des sources citables (A-001…). "
                            "Laissez vide pour ne pas répondre.")
            for q in qs:
                answers[q["id"]] = st.text_input(f"{q['id']} — {q['question']}")
            feedback = st.text_area("Retour libre (devient une source F-xxx si vous choisissez « Réviser »)")
            b1, b2, b3 = st.columns(3)
            approve = b1.form_submit_button("✅ Approuver et continuer", type="primary")
            revise = b2.form_submit_button("✏️ Réviser avec mes réponses")
            abort = b3.form_submit_button("⏹ Arrêter")
        if approve:
            run_resume({"action": "approve"}, "Reprise après approbation…")
        elif revise:
            filled = {k: v.strip() for k, v in answers.items() if v.strip()}
            if not filled and not feedback.strip():
                st.warning("Pour réviser, répondez à au moins une question ou écrivez un retour.")
            else:
                run_resume({"action": "revise", "answers": filled, "feedback": feedback}, "Révision en cours…")
        elif abort:
            run_resume({"action": "abort"}, "Arrêt…")
    elif not ss.finished:  # interrompu par une erreur (quota, réseau…) : reprise depuis le dernier checkpoint
        if st.button("↻ Reprendre là où le workflow s'est arrêté", type="primary"):
            run(None, "Reprise depuis le dernier checkpoint…")
    else:
        st.success(f"Workflow terminé (ou arrêté à votre demande). Fichiers produits dans `{ROOT_OUT / ss.thread}`.")

    # ------------------------------ 3. journal + artefacts ------------------------------
    out = ROOT_OUT / ss.thread
    with st.expander("Journal des étapes", expanded=False):
        for node, msg in ss.log:
            st.write(f"**{node}** — {msg}")
    st.subheader("Fichiers générés")
    present = [a for a in ARTIFACTS if (out / a).exists()]
    ws = out / "workspace"
    code = [str(f.relative_to(ws)) for f in sorted(ws.rglob("*")) if f.is_file() and "__pycache__" not in f.parts] \
        if ws.exists() else []
    if not present and not code:
        st.caption("Aucun fichier pour l'instant.")
        return
    tabs = st.tabs(present + (["Code (workspace)"] if code else []))
    for tab, name in zip(tabs, present):
        with tab:
            txt = (out / name).read_text(encoding="utf-8")
            if name.endswith(".json"):
                st.json(json.loads(txt), expanded=False)
            else:
                st.markdown(txt)
    if code:
        with tabs[-1]:
            f = st.selectbox("Fichier", code)
            st.code((ws / f).read_text(encoding="utf-8", errors="replace"),
                    language="python" if f.endswith(".py") else None)
    st.download_button("⬇ Télécharger tous les fichiers (.zip)", _zip(out), file_name=f"sdd-{ss.thread}.zip")


if __name__ == "__main__":  # exécuté par `streamlit run`
    render()
