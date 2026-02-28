"""Tests for hyperscaler energy strategies, ratepayer protection, BNEF H₂, Korea CSP investment."""
import pytest


def test_csp_energy_strategies_exist():
    from modules.m12_industry import CSP_ENERGY_STRATEGIES
    assert len(CSP_ENERGY_STRATEGIES) == 6
    for name in ["Google", "Amazon", "Meta", "Microsoft", "Samsung_SDS", "Naver"]:
        assert name in CSP_ENERGY_STRATEGIES
        assert "energy_mix" in CSP_ENERGY_STRATEGIES[name]


def test_get_csp_strategy():
    from modules.m12_industry import get_csp_strategy
    s = get_csp_strategy("Google")
    assert s["strategy"] == "BYPASS_QUEUE"
    assert abs(sum(s["energy_mix"].values()) - 1.0) < 0.01
    with pytest.raises(ValueError):
        get_csp_strategy("NonExistent")


def test_compare_csp_strategies():
    from modules.m12_industry import compare_csp_strategies
    results = compare_csp_strategies()
    assert len(results) == 6
    for r in results:
        assert "lcoe_krw_per_kwh" in r
        assert "carbon_tco2_per_mwh" in r
        assert "grid_dependency_pct" in r
        assert r["lcoe_krw_per_kwh"] > 0


def test_ratepayer_protection_scenario():
    from modules.m11_policy import POLICY_SCENARIOS, PolicySimulator
    assert "ratepayer_protection" in POLICY_SCENARIOS
    rp = POLICY_SCENARIOS["ratepayer_protection"]
    assert rp["self_generation_requirement"] == 0.80

    sim = PolicySimulator()
    result = sim.simulate_policy_impact("ratepayer_protection")
    assert result["self_generation_mw"] == 80.0
    assert result["consumer_protection"] == "소비자 요금 전가 0%"


def test_bnef_lcoh():
    from modules.m05_h2 import BNEF_LCOH_2025, get_lcoh_by_country, compare_lcoh_all
    assert len(BNEF_LCOH_2025) == 8
    assert get_lcoh_by_country("Korea") == 8.5
    assert get_lcoh_by_country("China") == 3.2
    with pytest.raises(ValueError):
        get_lcoh_by_country("Mars")

    data = compare_lcoh_all()
    assert len(data["countries"]) == 8
    assert len(data["lcoh_usd_per_kg"]) == 8


def test_korea_csp_investment():
    from modules.m13_investment import estimate_korea_csp_investment
    result = estimate_korea_csp_investment("Amazon", 500)
    assert result["dc_capacity_mw"] == 500
    assert result["benchmark_csp"] == "Amazon"
    assert result["estimated_kr_cost_billion_usd"] > 0
    assert result["korea_multiplier"] == 1.15

    # Unknown CSP → uses average
    result2 = estimate_korea_csp_investment("Unknown", 100)
    assert result2["benchmark_csp"] == "평균"
