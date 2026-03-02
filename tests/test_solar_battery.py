"""Solar Battery H₂ 생산 모듈 테스트"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.m05_h2 import solar_battery_h2_production


class TestSolarBatteryH2:

    def test_basic_production(self):
        """기본 H₂ 생산량 계산 확인"""
        result = solar_battery_h2_production(
            solar_irradiance_kwh_per_m2=5.0,
            area_m2=1000,
            eta_capture=0.80,
            eta_h2=0.72,
            storage_days=0,
        )
        assert result["method"] == "solar_battery"
        assert result["h2_production_kg"] > 0
        # 5000 kWh * 0.80 * 0.72 / 39.39 ≈ 73.1 kg
        assert abs(result["h2_production_kg"] - 73.1) < 1.0

    def test_sth_efficiency_nominal(self):
        """STH 효율이 η_capture × η_H2 ≈ 57.6%인지 확인"""
        result = solar_battery_h2_production(
            solar_irradiance_kwh_per_m2=5.0,
            area_m2=100,
            storage_days=0,
        )
        assert abs(result["sth_efficiency"] - 0.576) < 0.001

    def test_energy_conservation(self):
        """에너지 보존: h2_energy <= captured_energy <= total_solar_energy"""
        result = solar_battery_h2_production(
            solar_irradiance_kwh_per_m2=6.0,
            area_m2=500,
            storage_days=5,
            operating_years=3,
        )
        assert result["h2_energy_kwh"] <= result["stored_energy_kwh"]
        assert result["stored_energy_kwh"] <= result["captured_energy_kwh"]
        assert result["captured_energy_kwh"] <= result["total_solar_energy_kwh"]

    def test_eta_out_of_range_raises(self):
        """η 범위 [0,1] 벗어나면 ValueError"""
        with pytest.raises(ValueError):
            solar_battery_h2_production(5.0, 100, eta_capture=1.5)
        with pytest.raises(ValueError):
            solar_battery_h2_production(5.0, 100, eta_h2=-0.1)

    def test_zero_irradiance(self):
        """일사량 0이면 H₂ 생산도 0"""
        result = solar_battery_h2_production(0.0, 1000)
        assert result["h2_production_kg"] == 0.0
        assert result["sth_efficiency"] == 0.0

    def test_degradation_reduces_output(self):
        """운전 년수 증가 → 생산량 감소"""
        r0 = solar_battery_h2_production(5.0, 1000, operating_years=0)
        r10 = solar_battery_h2_production(5.0, 1000, operating_years=10)
        assert r10["h2_production_kg"] < r0["h2_production_kg"]
        assert r10["degradation_factor"] < 1.0

    def test_storage_loss_increases_with_days(self):
        """저장 일수 증가 → 저장 손실 증가"""
        r0 = solar_battery_h2_production(5.0, 1000, storage_days=0)
        r10 = solar_battery_h2_production(5.0, 1000, storage_days=10)
        assert r10["h2_production_kg"] < r0["h2_production_kg"]
        assert r10["storage_loss_factor"] < r0["storage_loss_factor"]
