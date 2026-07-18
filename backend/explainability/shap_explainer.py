"""
shap_explainer.py — SHAP TreeExplainer wrapper for XGBoost forecasts.
Returns data formatted specifically for the Dash frontend charts.
"""

from __future__ import annotations
import re
import numpy as np
import pandas as pd

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

from forecasting.xgboost_predictor import XGBoostPredictor

# Human-readable narrative templates per feature
_NARRATIVES: dict[str, dict[str, str]] = {
    "bilan_net":    {"high": "Net balance is favourable.",
                     "low":  "Net balance is below threshold."},
    "pressure":     {"high": "Pressure exceeds normal range.",
                     "low":  "Pressure remains stable."},
    "steam_hp":     {"high": "Steam availability is high.",
                     "low":  "Steam supply is limited."},
    "steam_mp":     {"high": "Medium-pressure steam is available.",
                     "low":  "Medium-pressure steam is low."},
    "vibration":    {"high": "Vibration is elevated — maintenance may be needed.",
                     "low":  "Vibration within normal bounds."},
    "efficiency":   {"high": "Operational efficiency is good.",
                     "low":  "Efficiency is degraded."},
    "temperature":  {"high": "Temperature is elevated.",
                     "low":  "Temperature is nominal."},
    "production":   {"high": "Production is above target.",
                     "low":  "Production is below target."},
}


# ── French labels / units for the natural-language layer ─────────────────────

_FR_BASE_LABELS: dict[str, str] = {
    "production":   "la production totale",
    "consumption":  "l'autoconsommation des auxiliaires",
    "bilan_net":    "le bilan énergétique net",
    "efficiency":   "le rendement global",
    "steam_hp":     "le débit de vapeur HP",
    "steam_mp":     "le débit de vapeur MP",
    "steam_bp":     "le débit de vapeur BP",
    "steam_ratio":  "le ratio vapeur/production (IPE)",
    "pressure":     "la pression HP",
    "temperature":  "la température d'admission",
    "vibration":    "le niveau de vibration",
    "gta_balance":  "l'équilibre de charge entre GTA",
    "import_export_ratio": "le ratio import/export",
    "day_of_week":  "l'effet jour de semaine",
    "is_weekend":   "l'effet week-end",
    "hour":         "l'effet horaire",
    "month":        "l'effet saisonnier (mois)",
    "quarter":      "l'effet trimestriel",
    "year":         "l'effet annuel",
}

_TARGET_UNITS: dict[str, tuple[str, float]] = {
    # target → (unit label, DH conversion per unit — 0 = not monetary)
    "production": ("MWh/j", 700.0),
    "bilan_net":  ("MWh/j", 700.0),
    "steam_hp":   ("t/h", 0.0),
    "efficiency": ("point de rendement", 0.0),
}

# Non-actionable context features (never turned into recommendations)
_CONTEXT_PATTERNS = ("month", "hour", "day_of_week", "is_weekend", "quarter",
                     "year", "_sin", "_cos")


def _fr_label(feat: str) -> str:
    """Human-readable French label for an engineered feature name."""
    suffix = ""
    base = feat
    for pat, txt in [("_roll_mean_3d", " (moyenne 3 j)"), ("_roll_7d", " (moyenne 7 j)"),
                     ("_roll_30d", " (moyenne 30 j)"), ("_std_7d", " (variabilité 7 j)")]:
        if base.endswith(pat):
            base, suffix = base[: -len(pat)], txt
            break
    m = re.match(r"(.+)_lag_(\d+)$", base)
    if m:
        base, suffix = m.group(1), f" (il y a {m.group(2)} j)"
    if base.endswith("_sin") or base.endswith("_cos"):
        base = base[:-4]
    prefix = ""
    if base.startswith("delta_"):
        base, prefix = base[6:], "la variation de "

    m = re.match(r"(.+)_gta(\d)$", base)
    if m:
        stem, num = m.group(1), m.group(2)
        stems = {"rendement": "le rendement", "debit_adm": "le débit d'admission",
                 "debit_sout": "le débit de soutirage", "debit_ext": "le débit d'extraction",
                 "pression_adm": "la pression d'admission", "temp_adm": "la température d'admission",
                 "h_adm": "l'enthalpie d'admission", "h_sout": "l'enthalpie de soutirage",
                 "h_ext": "l'enthalpie d'extraction"}
        core = f"{stems.get(stem, stem)} du GTA{num}"
    elif re.fullmatch(r"gta[0-9ab]", base):
        core = f"la production du {base.upper()}"
    else:
        core = _FR_BASE_LABELS.get(base, base.replace("_", " "))
    return prefix + core + suffix


# Actionable recommendation templates: (regex on feature, action, why)
_RECO_RULES: list[tuple[str, str, str]] = [
    (r"rendement_gta(\d)",
     "Inspecter les aubages et les buses d'admission du GTA{g} (encrassement probable) "
     "et recaler ses paramètres d'admission.",
     "Le rendement isentropique du GTA{g} tire la prévision vers le bas : chaque point "
     "de rendement perdu se paie directement en vapeur consommée pour le même MWh."),
    (r"(debit_adm_gta(\d)|steam_hp)",
     "Sécuriser la disponibilité vapeur HP : vérifier les chaudières de récupération (HRS), "
     "les vannes d'admission et, si besoin, engager la chaudière auxiliaire.",
     "Le manque de vapeur HP admise limite mécaniquement la détente disponible, donc la "
     "puissance produite par les turbo-alternateurs."),
    (r"(pression_adm_gta(\d)|pressure)",
     "Contrôler la pression du collecteur HP (consignes de détente, fuites réseau, purges).",
     "Une pression d'admission dégradée réduit l'enthalpie disponible et donc l'énergie "
     "récupérable par MW de vapeur."),
    (r"(temp_adm_gta(\d)|temperature)",
     "Régler la désurchauffe pour ramener la température d'admission dans la plage nominale "
     "(455–470 °C).",
     "Un écart de température d'admission dégrade le rendement du cycle et augmente les "
     "contraintes thermiques sur le corps HP."),
    (r"vibration",
     "Planifier une intervention sur la ligne d'arbre (équilibrage, alignement, paliers).",
     "La vibration contribue négativement à la prévision : elle traduit une dégradation "
     "mécanique qui précède les pertes de production."),
    (r"gta([123])$",
     "Recharger le GTA{g} si la vapeur disponible le permet (montée de consigne progressive).",
     "La production du GTA{g} est le facteur qui manque le plus à la prévision : une part "
     "de capacité installée n'est pas exploitée."),
    (r"(steam_mp|debit_sout_gta(\d))",
     "Optimiser le soutirage MP en fonction de la demande réelle des procédés aval.",
     "Un soutirage MP mal calé détourne de la vapeur qui pourrait être détendue jusqu'au "
     "condenseur pour produire de l'électricité."),
    (r"consumption",
     "Réduire l'autoconsommation des auxiliaires (pompes, ventilation) pendant les heures "
     "creuses de production.",
     "L'autoconsommation ampute directement le bilan net exporté vers le complexe."),
    (r"steam_ratio",
     "Suivre l'IPE (Tvap/MWh) et viser la cible ISO 50001 de 2,4 : rééquilibrer la charge "
     "entre GTA pour minimiser la vapeur consommée par MWh.",
     "Le ratio vapeur/production s'écarte de sa plage optimale, signe d'une conversion "
     "énergétique inefficace."),
    (r"gta_balance",
     "Rééquilibrer la répartition de charge entre GTA1, GTA2 et GTA3 selon leurs rendements "
     "respectifs.",
     "Un déséquilibre de charge fait travailler les groupes hors de leur point de "
     "fonctionnement optimal."),
]


class SHAPExplainer:

    def __init__(self, predictor: XGBoostPredictor):
        self.predictor   = predictor
        self._explainers: dict[str, object] = {}

    # ── Build explainer ───────────────────────────────────────────────────────

    def _get_explainer(self, target: str, horizon: str):
        key = f"{target}_{horizon}"
        if key in self._explainers:
            return self._explainers[key]
        if not _HAS_SHAP:
            raise ImportError("pip install shap")
        if key not in self.predictor.models:
            self.predictor._load(key)
        if key not in self.predictor.models:
            raise ValueError(f"No trained model for {key}")
        ex = shap.TreeExplainer(self.predictor.models[key])
        self._explainers[key] = ex
        return ex

    # ── Explain predictions (returns data formatted for Dash frontend) ────────

    def explain(self, df: pd.DataFrame, target: str = "bilan_net",
                horizon: str = "24h") -> dict:
        """Compute SHAP values for a sample of data to support frontend charts."""
        key     = f"{target}_{horizon}"
        feats   = self.predictor.feature_names.get(key)
        if feats is None:
            self.predictor._load(key)
            feats = self.predictor.feature_names.get(key, [])
            
        if not feats:
            raise ValueError(f"No features found for model {key}")

        # Prepare input data: take the last 100 rows for charts, or all if less
        n_samples = min(100, len(df))
        X_sample = df.tail(n_samples)[feats].fillna(0).values

        explainer   = self._get_explainer(target, horizon)
        
        # Compute SHAP values for the sample (returns a 2D matrix)
        shap_values_matrix = explainer.shap_values(X_sample)
        if isinstance(shap_values_matrix, list):
            shap_values_matrix = shap_values_matrix[0]
            
        # Base value
        base_val   = float(explainer.expected_value
                           if not hasattr(explainer.expected_value, "__len__")
                           else explainer.expected_value[0])
                           
        # Last row prediction
        sv_last = shap_values_matrix[-1]
        pred_val = float(base_val + sv_last.sum())
        
        # Compute mean absolute SHAP for the bar chart
        mean_abs_shap = np.abs(shap_values_matrix).mean(axis=0).tolist()
        
        # Top features for narrative
        contributions = {f: round(float(v), 4) for f, v in zip(feats, sv_last)}
        top = dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:10])

        # Return EXACTLY the keys the Dash frontend expects
        return {
            "feature_names": feats,
            "mean_abs_shap": mean_abs_shap,
            "shap_values": shap_values_matrix.tolist(), # 2D list for heatmap
            "base_value": round(base_val, 2),
            "predicted_value": round(pred_val, 2),
            "explanation": self._narrative(target, horizon, base_val, pred_val,
                                           contributions),
            "recommendations": self._recommendations(target, contributions),
            "target": target,
            "horizon": horizon,
        }

    # ── Explain an RL recommendation ─────────────────────────────────────────

    def explain_recommendation(self, action_label: str,
                                shap_result: dict) -> str:
        """Convert a SHAP result dict into a human-readable recommendation text."""
        top = list(shap_result.get("top_features", {}).keys())[:3]
        if not top and "feature_names" in shap_result:
            mean_abs = shap_result.get("mean_abs_shap", [])
            if mean_abs:
                top_idx = sorted(range(len(mean_abs)), key=lambda i: abs(mean_abs[i]), reverse=True)[:3]
                top = [shap_result["feature_names"][i] for i in top_idx]
                
        parts = [f"Recommendation: {action_label}", "", "Reason:"]
        
        contributions = shap_result.get("feature_contributions", {})
        if not contributions and "feature_names" in shap_result and "shap_values" in shap_result:
            feats = shap_result["feature_names"]
            sv_last = shap_result["shap_values"][-1]
            contributions = {f: v for f, v in zip(feats, sv_last)}
            
        for feat in top:
            val = contributions.get(feat, 0)
            direction = "high" if val > 0 else "low"
            tmpl = _NARRATIVES.get(feat, {}).get(direction, f"{feat} is a key driver.")
            parts.append(f"  • {tmpl}")
        return "\n".join(parts)

    # ── Narrative builder (French, multi-paragraph) ───────────────────────────

    @staticmethod
    def _fmt_contrib(feat: str, val: float, unit: str) -> str:
        label = _fr_label(feat)
        label = label[0].lower() + label[1:]
        return f"{label} ({val:+.2f} {unit})"

    def _narrative(self, target: str, horizon: str, base: float, pred: float,
                   contributions: dict[str, float]) -> str:
        unit, _ = _TARGET_UNITS.get(target, ("", 0.0))
        target_label = _fr_label(target)
        diff = pred - base

        ranked = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        pos = [(f, v) for f, v in ranked if v > 0][:3]
        neg = [(f, v) for f, v in ranked if v < 0][:3]

        p1 = (f"Pour {target_label[0].lower()}{target_label[1:]} à l'horizon {horizon}, "
              f"le modèle XGBoost prédit {pred:.2f} {unit} alors que sa valeur de base "
              f"(prédiction moyenne sans information sur l'état actuel) est de "
              f"{base:.2f} {unit}. L'état actuel de la centrale déplace donc la "
              f"prévision de {diff:+.2f} {unit}.")

        parts = [p1]
        if pos:
            parts.append(
                "Ce qui joue en faveur de la prévision : "
                + " ; ".join(self._fmt_contrib(f, v, unit) for f, v in pos)
                + ". Ces conditions sont favorables — il s'agit de les maintenir.")
        if neg:
            parts.append(
                "Ce qui pénalise la prévision : "
                + " ; ".join(self._fmt_contrib(f, v, unit) for f, v in neg)
                + ". Ce sont les leviers d'action prioritaires : chaque contribution "
                  "négative représente un manque à gagner récupérable si la cause "
                  "physique est corrigée.")
        if not pos and not neg:
            parts.append("Aucune variable ne s'écarte significativement de la normale : "
                         "la centrale fonctionne à son point d'équilibre habituel.")

        balance = sum(v for _, v in contributions.items())
        if balance < 0:
            parts.append(f"Au global, l'état courant coûte {abs(balance):.2f} {unit} "
                         f"par rapport au fonctionnement moyen — voir les "
                         f"recommandations ci-dessous pour les récupérer.")
        else:
            parts.append(f"Au global, l'état courant apporte {balance:+.2f} {unit} "
                         f"par rapport au fonctionnement moyen.")
        return "\n\n".join(parts)

    # ── SHAP-driven recommendations ───────────────────────────────────────────

    def _recommendations(self, target: str,
                         contributions: dict[str, float]) -> list[dict]:
        """Actionable steps derived from the strongest *negative* SHAP
        contributors (the factors dragging the forecast down). Top-3 per F16."""
        unit, dh_per_unit = _TARGET_UNITS.get(target, ("", 0.0))

        negatives = sorted(((f, v) for f, v in contributions.items() if v < 0),
                           key=lambda x: x[1])
        recos: list[dict] = []
        used_actions: set[str] = set()
        for feat, val in negatives:
            if any(p in feat for p in _CONTEXT_PATTERNS):
                continue                       # seasonal/time context — not actionable
            for pattern, action_tpl, why_tpl in _RECO_RULES:
                m = re.search(pattern, feat)
                if not m:
                    continue
                g = next((x for x in m.groups() if x and x.isdigit()), "")
                action = action_tpl.format(g=g)
                if action in used_actions:
                    break                      # one step per action type
                used_actions.add(action)

                gain_txt = f"{abs(val):.2f} {unit} récupérables sur la prévision"
                if dh_per_unit and unit.startswith("MWh"):
                    gain_txt += (f", soit ≈ {abs(val) * dh_per_unit:,.0f} DH/jour "
                                 f"(tarif 700 DH/MWh)")
                elif target == "efficiency":
                    gain_txt = (f"{abs(val) * 100:.2f} point(s) de rendement "
                                f"récupérables sur la prévision")

                recos.append({
                    "action":        action,
                    "cause":         (f"{_fr_label(feat)} contribue pour {val:+.2f} {unit} "
                                      f"à la prévision (valeur SHAP)"),
                    "why":           why_tpl.format(g=g),
                    "expected_gain": gain_txt,
                    "feature":       feat,
                    "shap_value":    round(float(val), 4),
                })
                break
            if len(recos) >= 3:
                break
        return recos