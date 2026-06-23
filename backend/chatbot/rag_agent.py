"""
rag_agent.py — Retrieval-Augmented Generation chatbot using Local Ollama.
Supports dynamic French/English responses and smart rule-based fallbacks with DB context.
"""

from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from database.database import query
import ollama

# --- CONFIGURATION ---
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434") 
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:7b")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")

client = ollama.Client(host=OLLAMA_HOST)

# ── Language Detection Helper ─────────────────────────────────────────────────
def _detect_language(question: str) -> str:
    """Simple heuristic to detect if the question is French or English."""
    french_keywords = [
        "pourquoi", "comment", "quel", "quelle", "est", "sont", "le", "la", "les", 
        "un", "une", "rendement", "anomalie", "cause", "améliorer", "vapeur", 
        "maintenance", "gta", "production", "consommation", "bilan"
    ]
    q = question.lower()
    # If at least 2 French keywords are found, assume French
    score = sum(1 for kw in french_keywords if kw in q)
    return "fr" if score >= 2 else "en"

# ── Context builder ───────────────────────────────────────────────────────────
def _build_context(context_hours: int = 24) -> str:
    """Pull recent DB data into a compact text block for the LLM prompt."""
    
    # 🔥 PRIORITY 1: LIVE SCADA SIMULATION DATA (Real-time)
    sim_data = query("""
        SELECT gta_type, adm_debit, adm_temp, adm_pression, 
               sout_debit, sout_pression, puissance_mw, rendement,
               vitesse, vib1, vib2, oil_pression, oil_temp,
               timestamp
        FROM simulation_data 
        WHERE gta_type IN ('GTA1', 'GTA2', 'GTA3')
        ORDER BY timestamp DESC 
        LIMIT 3
    """)
    
    ctx_parts = []
    
    # 🔥 CRITICAL: Put LIVE data FIRST with clear priority label
    if sim_data:
        ctx_parts.append("=== 🚨 LIVE SCADA DATA (REAL-TIME - USE THIS FOR CURRENT STATUS) ===")
        for sim in sim_data:
            gta = sim.get('gta_type', '?')
            debit = float(sim.get('adm_debit') or 0)
            temp = float(sim.get('adm_temp') or 0)
            pression = float(sim.get('adm_pression') or 0)
            puissance = float(sim.get('puissance_mw') or 0)
            rendement = float(sim.get('rendement') or 0)
            vib1 = float(sim.get('vib1') or 0)
            vib2 = float(sim.get('vib2') or 0)
            oil_temp = float(sim.get('oil_temp') or 0)
            ts = sim.get('timestamp', '?')
            
            ctx_parts.append(
                f"  [{gta}] @ {ts[:16]} | "
                f"débit={debit:.1f} t/h | T°={temp:.1f}°C | P={pression:.1f} bar | "
                f"Puissance={puissance:.1f} MW | Rendement={rendement:.1f}% | "
                f"Vib1={vib1:.2f} mm/s | Vib2={vib2:.2f} mm/s | T° huile={oil_temp:.1f}°C"
            )
        ctx_parts.append("")
    
    # 🔥 SECONDARY: Historical data (reference only)
    rows = query("SELECT * FROM historical_data ORDER BY timestamp DESC LIMIT ?", [context_hours])
    anoms = query("SELECT * FROM anomalies ORDER BY timestamp DESC LIMIT 10")
    preds = query("SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 5")
    recs  = query("SELECT * FROM recommendations ORDER BY timestamp DESC LIMIT 3")

    if rows:
        ctx_parts.append("=== 📚 HISTORICAL DATA (REFERENCE ONLY - DO NOT USE FOR CURRENT STATUS) ===")
        r = rows[0]
        prod = r.get('production') or 0
        bilan = r.get('bilan_net') or 0
        eff = r.get('efficiency') or 0
        pres = r.get('pressure') or 0
        vib = r.get('vibration') or 0
        steam = r.get('steam_hp') or 0
        
        ctx_parts.append(
            f"  production={float(prod):.1f} MWh | bilan_net={float(bilan):.1f} MWh | "
            f"efficiency={float(eff):.3f} | pressure={float(pres):.1f} bar | "
            f"vibration={float(vib):.2f} mm/s | steam_hp={float(steam):.1f}"
        )
        ctx_parts.append("")

    if anoms:
        ctx_parts.append("=== ⚠️ Active Anomalies ===")
        for a in anoms[:3]:
            sev = (a.get('severity') or '?').upper()
            cause = a.get('cause') or '?'
            score = float(a.get('score') or 0)
            ts = a.get('timestamp') or '?'
            ctx_parts.append(f"  [{sev}] {cause} (score={score:.2f}) at {ts}")
        ctx_parts.append("")

    if preds:
        ctx_parts.append("=== 🔮 Latest Predictions ===")
        for p in preds:
            var = p.get('variable') or '?'
            hor = p.get('horizon') or '?'
            val = float(p.get('predicted_value') or 0)
            conf = float(p.get('confidence') or 0)
            ctx_parts.append(f"  {var} ({hor}) = {val:.0f} (conf={conf:.2f})")
        ctx_parts.append("")

    if recs:
        ctx_parts.append("=== 💡 Latest Recommendations ===")
        for rc in recs:
            action = rc.get('action') or '?'
            gain = float(rc.get('expected_gain_mwh') or 0)
            conf = float(rc.get('confidence') or 0)
            ctx_parts.append(f"  Action: {action} | gain={gain:.0f} MWh | conf={conf:.2f}")
        ctx_parts.append("")

    return "\n".join(ctx_parts)

# ── Main RAG agent ────────────────────────────────────────────────────────────
class RAGAgent:
    """
    Retrieval-Augmented Generation agent using Local Ollama.
    """

    # 🔥 CRITICAL: Force LLM to prioritize LIVE data
    SYSTEM_PROMPT = (
        "You are an expert industrial AI assistant for an OCP cogeneration plant in Morocco. "
        "You have access to real-time sensor data, anomaly alerts, forecasting predictions, "
        "and RL-based recommendations provided in the 'Context' section below.\n\n"
        "🚨 CRITICAL INSTRUCTIONS:\n"
        "1. DATA PRIORITY: When the user asks for CURRENT or REAL-TIME data, you MUST use the "
        "'🚨 LIVE SCADA DATA (REAL-TIME - USE THIS FOR CURRENT STATUS)' section. "
        "DO NOT use historical data for current status questions.\n"
        "2. LANGUAGE DETECTION: Detect the language of the user's question. "
        "If the question is in French, you MUST answer in clear, technical French. "
        "If the question is in English, you MUST answer in clear, technical English.\n"
        "3. CONTEXT RELIANCE: You MUST base your answer strictly on the provided 'Context' data. "
        "Always cite specific numbers (e.g., vibration levels, efficiency percentages) from the context when available. "
        "Do not hallucinate or use generic knowledge if the context provides specific plant data.\n"
        "4. SOURCE CITATION: Always mention which data source you used (LIVE SCADA or HISTORICAL)."
    )

    def __init__(self):
        self._client = client

    def answer(self, question: str, context_hours: int = 24) -> dict:
        """Answer an engineer question with context from the DB."""
        context = _build_context(context_hours)

        # Try Ollama LLM
        try:
            return self._ollama_answer(question, context)
        except Exception as exc:
            print(f"[RAG] Ollama error: {exc} — falling back to rule-based")

        # Rule-based fallback (Now bilingual and includes live data!)
        rule = _rule_answer(question, context)
        if rule:
            return rule

        # Default fallback (Bilingual)
        lang = _detect_language(question)
        default_msg = (
            "Je n'ai pas assez d'informations pour répondre précisément. Veuillez consulter les tableaux de bord de surveillance et de détection d'anomalies pour les données en direct." 
            if lang == "fr" else 
            "I don't have enough information to answer that question precisely. Please check the Monitoring and Anomaly Detection dashboards for live data."
        )

        return {
            "answer": default_msg,
            "sources": [],
            "confidence": 0.0,
        }

    def _ollama_answer(self, question: str, context: str) -> dict:
        messages = [
            {"role": "system",  "content": self.SYSTEM_PROMPT},
            {"role": "user",    "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
        
        # Call local Ollama model
        response = self._client.chat(
            model=LLM_MODEL,
            messages=messages,
            options={"temperature": 0.2, "num_predict": 600}
        )
        
        answer = response['message']['content'].strip()
        
        return {
            "answer":     answer,
            "sources":    ["live_scada", "historical_data", "anomalies", "predictions", "recommendations"],
            "confidence": 0.85,
            "model":      LLM_MODEL,
        }

    def get_embedding(self, text: str) -> list[float]:
        """Generate embeddings using the local nomic-embed-text model."""
        response = self._client.embeddings(model=EMBED_MODEL, prompt=text)
        return response['embedding']