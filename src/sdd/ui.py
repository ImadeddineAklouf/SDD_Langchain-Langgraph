"""Interface web locale (Streamlit) du pipeline SDD.

Lancement depuis la racine du dépôt :  sdd-ui   (ou : streamlit run src/sdd/ui.py)

L'interface ne fait que remplacer le terminal de `cli.py` : elle envoie le brief et la constitution
au graphe, affiche chaque étape et chaque artefact, présente les alertes du Gardien (boîte d'alertes)
et transmet la décision humaine (répondre / corriger / accepter le risque / continuer / arrêter) via
`Command(resume=...)`. Les garde-fous (validateurs, Gardien, journal d'audit) restent ceux du graphe.
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
ARTIFACTS = ["alerts.md", "audit.md", "spec.md", "plan.md", "tasks.md", "analyze.md", "verify.md", "sources.json"]


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


def _graph(thread: str, max_retries: int, require_tests: bool, factory: bool):
    """Un graphe par session, avec son checkpointer SQLite (reprise possible, comme la CLI)."""
    import streamlit as st

    from sdd.llm import fallback_name, guardian_name, guardian_semantic_enabled, model_name, provider

    @st.cache_resource(show_spinner=False)
    def build(thread: str, max_retries: int, require_tests: bool, factory: bool, _llm_key: tuple):
        from langgraph.checkpoint.sqlite import SqliteSaver

        from sdd.graph import SUPERVISED, build_graph
        from sdd.llm import get_fallback_llm, get_guardian_llm, get_llm

        out = ROOT_OUT / thread
        out.mkdir(parents=True, exist_ok=True)
        saver = SqliteSaver(sqlite3.connect(out / "checkpoints.db", check_same_thread=False))
        return build_graph(get_llm(), out, max_retries, checkpointer=saver, require_tests=require_tests,
                           fallback_llm=get_fallback_llm(), guardian_llm=get_guardian_llm(),
                           checkpoints=() if factory else SUPERVISED)

    # le graphe est reconstruit si l'on change de fournisseur, de modèle ou de réglage du Gardien
    key = (provider(), model_name(), fallback_name(), guardian_name(), guardian_semantic_enabled())
    return build(thread, max_retries, require_tests, factory, key)


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
        st.subheader("🛡️ Gardien")
        factory = st.checkbox("Mode usine : pause uniquement sur alerte du Gardien", value=True,
                              help="Décoché (mode supervisé) : pause aussi à la fin de chaque étape.")
        semantic = st.checkbox("Contrôle sémantique par LLM (en plus des contrôles déterministes)", value=True,
                               help="Le LLM du Gardien ne peut que lever des alertes, jamais valider.")
        os.environ["SDD_GUARDIAN_SEMANTIC"] = "1" if semantic else "0"
        if semantic:
            g_opts = ["(même modèle)"] + [m for m in (available or []) if m != model]
            g_cur = os.getenv("SDD_GUARDIAN_MODEL") or "(même modèle)"
            g_model = st.selectbox("Modèle du Gardien (idéalement différent du producteur)", g_opts,
                                   index=g_opts.index(g_cur) if g_cur in g_opts else 0)
            os.environ["SDD_GUARDIAN_MODEL"] = "" if g_model == "(même modèle)" else g_model
        st.divider()
        st.subheader("Reprendre une session")
        old = st.text_input("Identifiant de session", placeholder="ex. 3f9a1c2b")
        if st.button("Reprendre", disabled=not old):
            ss.update(thread=old.strip(), log=[], finished=False, error=None)
            snap = _graph(ss.thread, max_retries, require_tests, factory).get_state(_cfg(ss.thread))
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

    st.title("📐 SDD Pipeline déterministe · 🛡️ Gardien")
    st.caption("constitution → specify → clarify → plan → tasks → analyze → implement. "
               "Le modèle transforme vos textes ; il n'a pas le droit d'inventer. Chaque phrase devient une source "
               "citable (C-001, B-001…). Après chaque étape, le Gardien, un vérificateur indépendant, contrôle la "
               "cohérence brief ↔ spec ↔ plan ↔ tâches ↔ code et met le workflow en pause au moindre problème.")

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
        graph = _graph(ss.thread, max_retries, require_tests, factory)
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

    # ------------------------------ 2. boîte d'alertes du Gardien ------------------------------
    p = ss.pending
    if p:
        stage = p["stage"]
        if p["kind"] == "alert":
            st.subheader("🚨 Boîte d'alertes du Gardien : " + STAGE_TITLES.get(stage, stage))
            st.caption("Le Gardien a détecté un problème : le workflow SDD est **en pause** et attend votre décision. "
                       "Rien n'est écrit ni généré tant que vous n'avez pas tranché.")
        else:
            st.subheader("⏸ Point de contrôle : " + STAGE_TITLES.get(stage, stage))
            st.caption("Mode supervisé : le Gardien n'a rien trouvé, mais vous validez chaque étape.")
        if p.get("gate_error"):
            st.error("Décision refusée : " + p["gate_error"])
        for a in p.get("alerts", []):
            refs = f" · {', '.join(a['refs'])}" if a["refs"] else ""
            text = (f"**{a['code']}** ({a['origin']}{refs})  \n{a['message']}  \n"
                    f"_Reprise à l'étape : {a['redo']}_")
            (st.error if a["severity"] == "blocking" else st.warning)(text, icon="⛔" if a["severity"] == "blocking" else "⚠️")
        if p.get("info"):
            with st.expander(f"Informations du Gardien ({len(p['info'])})"):
                for a in p["info"]:
                    st.info(("Risque déjà accepté : " if a.get("accepted") else "") + f"**{a['code']}** {a['message']}")
        if p.get("file") and Path(p["file"]).exists():
            with st.expander(f"Artefact : {Path(p['file']).name}", expanded=p["kind"] != "alert"):
                st.markdown(Path(p["file"]).read_text(encoding="utf-8"))
        if stage == "impl":
            st.caption(f"Tâche **{p['task_id']}** : les fichiers ne sont écrits qu'après votre décision.")
            for path, content in p.get("preview", {}).items():
                with st.expander(path, expanded=p["kind"] != "alert"):
                    st.code(content, language="python" if path.endswith(".py") else None)
        if "passed" in p:
            (st.success if p["passed"] else st.error)("Tests : " + ("OK" if p["passed"] else "échec"))

        with st.form(f"gate-{stage}-{len(ss.log)}"):
            answers = {}
            qs = p.get("open_questions", [])
            if qs:
                st.markdown("**Questions ouvertes** : vos réponses deviennent des sources citables (A-001…) "
                            "et le workflow reprend à l'étape concernée.")
            for q in qs:
                answers[q["id"]] = st.text_input(f"{q['id']} — {q['question']}")
            feedback = st.text_area("Correction (devient une source F-xxx, puis l'étape fautive est relancée)")
            justification = st.text_area("Justification (obligatoire pour accepter le risque, tracée dans l'audit)")
            cols = st.columns(5)
            b_answer = cols[0].form_submit_button("💬 Répondre et reprendre", type="primary", disabled=not qs)
            b_correct = cols[1].form_submit_button("✏️ Corriger et reprendre")
            b_accept = cols[2].form_submit_button("⚠️ Accepter le risque", disabled=p["kind"] != "alert")
            b_continue = cols[3].form_submit_button("▶ Continuer", disabled=p["blocking"],
                                                    type="primary" if not qs else "secondary")
            b_abort = cols[4].form_submit_button("⏹ Arrêter")
        filled = {k: v.strip() for k, v in answers.items() if v.strip()}
        decision = None
        if b_answer:
            decision = {"action": "answer", "answers": filled, "feedback": feedback}
        elif b_correct:
            decision = {"action": "correct", "answers": filled, "feedback": feedback}
        elif b_accept:
            decision = {"action": "accept_risk", "answers": filled, "justification": justification}
        elif b_continue:
            decision = {"action": "continue", "answers": filled}
        elif b_abort:
            decision = {"action": "abort"}
        if decision:  # la validation (justification, réponses…) est faite par le graphe, pas par l'UI
            run_resume(decision, "Reprise du workflow…")
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
