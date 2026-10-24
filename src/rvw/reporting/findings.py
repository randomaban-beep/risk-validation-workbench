"""Algorithmic, rule-based validation findings generator."""

from __future__ import annotations

from typing import Any, Dict, List


def generate_rule_based_findings(summary_data: Dict[str, Any]) -> List[str]:
    """Generate objective, rule-based validation findings from computed backtest metrics.

    No fabricated statements: all findings are strictly derived from computed statistics.

    Args:
        summary_data: Dictionary loaded from results/summary.json.

    Returns:
        List[str]: List of bulleted qualitative findings.
    """
    findings: List[str] = []
    test_metrics = summary_data.get("test_metrics", {})
    val_metrics = summary_data.get("validation_metrics", {})
    hp = summary_data.get("hyperparameter_selection", {})

    # Data origin finding
    if summary_data.get("is_synthetic", False):
        findings.append(
            "DATA STATUS: Validation was executed on calibrated SYNTHETIC market data. "
            "Statistical outcomes demonstrate code execution and pipeline integrity, not real-world market dynamics."
        )
    else:
        findings.append(
            f"DATA STATUS: Validation executed on real market data ({summary_data.get('source', 'yfinance')}) "
            f"covering {len(summary_data.get('tickers', []))} assets."
        )

    # Hyperparameter selection finding
    if hp:
        findings.append(
            f"Hyperparameter Grid Search selected a rolling window of {hp.get('selected_window_length')} "
            f"trading days and an EWMA decay factor of {hp.get('selected_ewma_lambda')}."
        )

    # Test period model evaluations
    green_models = []
    yellow_models = []
    red_models = []
    kupiec_rejects = []
    indep_rejects = []
    mf_rejects = []

    for model_name, m in test_metrics.items():
        zone = m.get("basel_zone", "Unknown")
        if zone == "Green":
            green_models.append(model_name)
        elif zone == "Yellow":
            yellow_models.append(model_name)
        elif zone == "Red":
            red_models.append(model_name)

        if m.get("kupiec_p", 1.0) < 0.05:
            kupiec_rejects.append((model_name, m.get("exceptions"), m.get("expected_exceptions"), m.get("kupiec_p")))

        if m.get("independence_p", 1.0) < 0.05:
            indep_rejects.append((model_name, m.get("independence_p")))

        if m.get("mcneil_frey_p", 1.0) < 0.05:
            mf_rejects.append((model_name, m.get("mcneil_frey_p")))

    # Basel summary
    if green_models:
        findings.append(
            f"Basel Traffic Light (99% VaR): Models classified in the Green zone (<= 4 exceptions): "
            f"{', '.join(f'`{m}`' for m in green_models)}."
        )
    if yellow_models:
        findings.append(
            f"Basel Traffic Light (99% VaR): Models in the Yellow zone (supervisory review): "
            f"{', '.join(f'`{m}`' for m in yellow_models)}."
        )
    if red_models:
        findings.append(
            f"Basel Traffic Light (99% VaR): Models in the Red zone (model rejection): "
            f"{', '.join(f'`{m}`' for m in red_models)}."
        )

    # Unconditional coverage findings
    if kupiec_rejects:
        for m_name, exc, exp, p_val in kupiec_rejects:
            findings.append(
                f"Kupiec Unconditional Coverage rejection: Model `{m_name}` failed at alpha=0.05 "
                f"(p={p_val:.4f}) with {exc} realized exceptions versus {exp:.1f} expected."
            )
    else:
        findings.append(
            "Kupiec Unconditional Coverage: No models rejected the null hypothesis of accurate exception frequency at alpha=0.05."
        )

    # Independence findings
    if indep_rejects:
        for m_name, p_val in indep_rejects:
            findings.append(
                f"Christoffersen Independence rejection: Model `{m_name}` exhibited significant exception clustering "
                f"(p={p_val:.4f} < 0.05), violating serial independence."
            )
    else:
        findings.append(
            "Christoffersen Independence: Realized exceptions did not exhibit statistically significant clustering for any model."
        )

    # Expected Shortfall findings
    if mf_rejects:
        for m_name, p_val in mf_rejects:
            findings.append(
                f"McNeil-Frey ES Exceedance test rejection: Model `{m_name}` rejected H0 (p={p_val:.4f} < 0.05), "
                "indicating realized tail losses significantly exceeded the Expected Shortfall forecasts."
            )
    else:
        findings.append(
            "McNeil-Frey ES Exceedance: No model showed statistically significant tail underestimation at alpha=0.05."
        )

    return findings
