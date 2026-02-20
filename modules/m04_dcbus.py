"""
M4. DC Bus 전력 분배 모듈
PV, HESS, H₂, 그리드, AIDC 간 전력 흐름 관리
순시 전력 균형 제약 구현
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings

from config import CONVERTER_EFFICIENCY


class DCBusModule:
    """DC Bus 전력 분배 모듈"""
    
    def __init__(self,
                 bus_voltage: str = "380V",
                 converter_tech: str = "default",
                 grid_capacity_mw: float = 20.0):
        """
        DC Bus 모듈 초기화
        
        Args:
            bus_voltage: 버스 전압 ("48V" or "380V")
            converter_tech: 변환기 기술 ("default": SiC, "advanced": GaN+)
            grid_capacity_mw: 그리드 연계 용량 (MW, 양방향)
        """
        self.bus_voltage = bus_voltage
        self.converter_tech = converter_tech
        self.grid_capacity_mw = grid_capacity_mw
        
        # 변환 효율 설정
        if converter_tech in CONVERTER_EFFICIENCY:
            self.efficiencies = CONVERTER_EFFICIENCY[converter_tech].copy()
        else:
            self.efficiencies = CONVERTER_EFFICIENCY["default"].copy()
        
        # 전력 흐름 기록
        self.power_flows: List[Dict] = []
    
    def calculate_power_balance(self,
                               pv_power_mw: float,
                               aidc_demand_mw: float,
                               bess_available_mw: float = 0,
                               bess_soc: float = 0.5,
                               h2_electrolyzer_max_mw: float = 0,
                               h2_fuelcell_max_mw: float = 0,
                               grid_import_limit_mw: Optional[float] = None,
                               grid_export_limit_mw: Optional[float] = None) -> Dict[str, float]:
        """
        순시 전력 균형 계산 및 최적 배분
        
        Args:
            pv_power_mw: PV 발전량 (MW)
            aidc_demand_mw: AIDC 부하 (MW)
            bess_available_mw: BESS 가용 용량 (MW, 양방향)
            bess_soc: BESS SoC (0-1)
            h2_electrolyzer_max_mw: 수전해 최대 용량 (MW)
            h2_fuelcell_max_mw: 연료전지 최대 용량 (MW)  
            grid_import_limit_mw: 계통 구매 제한 (None=제한없음)
            grid_export_limit_mw: 계통 판매 제한 (None=제한없음)
            
        Returns:
            전력 흐름 결과 딕셔너리
        """
        if grid_import_limit_mw is None:
            grid_import_limit_mw = self.grid_capacity_mw
        if grid_export_limit_mw is None:
            grid_export_limit_mw = self.grid_capacity_mw
        
        # 0. 입력 유효성 검증 (NaN 방지)
        pv_power_mw = max(0.0, float(pv_power_mw)) if np.isfinite(pv_power_mw) else 0.0
        aidc_demand_mw = max(0.0, float(aidc_demand_mw)) if np.isfinite(aidc_demand_mw) else 0.0
        
        # 1. 변환 효율 적용된 공급/수요 계산
        pv_to_bus_mw = pv_power_mw * self.efficiencies['pv_to_dcbus']
        aidc_from_bus_mw = aidc_demand_mw / self.efficiencies['dcbus_to_aidc']
        
        # 2. 기본 전력 불균형 계산
        power_imbalance = pv_to_bus_mw - aidc_from_bus_mw  # 양수: 잉여, 음수: 부족
        
        # 3. 전력 흐름 변수 초기화
        result = {
            # 입력 (공급원)
            'pv_power_mw': pv_power_mw,
            'bess_discharge_mw': 0,
            'h2_fuelcell_mw': 0,
            'grid_import_mw': 0,
            
            # 출력 (수요원)
            'aidc_load_mw': aidc_demand_mw,
            'bess_charge_mw': 0,
            'h2_electrolyzer_mw': 0,
            'grid_export_mw': 0,
            
            # 손실
            'conversion_loss_mw': 0,
            
            # 상태
            'power_balance_mw': 0,  # 0에 가까워야 함
            'energy_autonomous': True,  # 계통 의존도
            'curtailment_mw': 0,  # PV 출력 제한
        }
        
        remaining_imbalance = power_imbalance
        
        # 4. 잉여 전력 처리 (PV > AIDC)
        if remaining_imbalance > 0:
            result.update(self._handle_surplus_power(
                remaining_imbalance, bess_available_mw, bess_soc,
                h2_electrolyzer_max_mw, grid_export_limit_mw
            ))
        
        # 5. 부족 전력 처리 (PV < AIDC)
        elif remaining_imbalance < 0:
            result.update(self._handle_deficit_power(
                -remaining_imbalance, bess_available_mw, bess_soc,
                h2_fuelcell_max_mw, grid_import_limit_mw
            ))
        
        # 6. 변환 손실 계산
        result['conversion_loss_mw'] = self._calculate_total_losses(result)
        
        # 7. 최종 전력 균형 검증
        total_supply = (result['pv_power_mw'] * self.efficiencies['pv_to_dcbus'] +
                       result['bess_discharge_mw'] * self.efficiencies['dcbus_to_bess'] +
                       result['h2_fuelcell_mw'] * self.efficiencies['fc_to_dcbus'] +
                       result['grid_import_mw'] * self.efficiencies['grid_bidirectional'])
        
        total_demand = (result['aidc_load_mw'] / self.efficiencies['dcbus_to_aidc'] +
                       result['bess_charge_mw'] / self.efficiencies['dcbus_to_bess'] +
                       result['h2_electrolyzer_mw'] / self.efficiencies['dcbus_to_electrolyzer'] +
                       result['grid_export_mw'] / self.efficiencies['grid_bidirectional'])
        
        result['power_balance_mw'] = total_supply - total_demand
        result['energy_autonomous'] = (result['grid_import_mw'] == 0)
        
        # 기록 저장
        self.power_flows.append(result.copy())
        
        return result
    
    def _handle_surplus_power(self,
                            surplus_mw: float,
                            bess_available_mw: float,
                            bess_soc: float,
                            h2_max_mw: float,
                            grid_export_limit: float) -> Dict[str, float]:
        """잉여 전력 처리 로직"""
        result = {'bess_charge_mw': 0, 'h2_electrolyzer_mw': 0, 
                 'grid_export_mw': 0, 'curtailment_mw': 0}
        
        remaining_surplus = surplus_mw
        
        # 우선순위 1: BESS 충전 (SoC 90% 이하에서)
        if remaining_surplus > 0 and bess_soc < 0.9:
            bess_charge_capacity = min(
                bess_available_mw,
                remaining_surplus * self.efficiencies['dcbus_to_bess']
            )
            result['bess_charge_mw'] = bess_charge_capacity
            remaining_surplus -= bess_charge_capacity / self.efficiencies['dcbus_to_bess']
        
        # 우선순위 2: 수전해 (장주기 저장)
        if remaining_surplus > 0 and h2_max_mw > 0:
            h2_electrolyzer_power = min(
                h2_max_mw,
                remaining_surplus * self.efficiencies['dcbus_to_electrolyzer']
            )
            result['h2_electrolyzer_mw'] = h2_electrolyzer_power
            remaining_surplus -= h2_electrolyzer_power / self.efficiencies['dcbus_to_electrolyzer']
        
        # 우선순위 3: 계통 판매
        if remaining_surplus > 0:
            grid_export_power = min(
                grid_export_limit,
                remaining_surplus * self.efficiencies['grid_bidirectional']
            )
            result['grid_export_mw'] = grid_export_power
            remaining_surplus -= grid_export_power / self.efficiencies['grid_bidirectional']
        
        # 마지막: PV 출력 제한 (curtailment)
        if remaining_surplus > 0:
            result['curtailment_mw'] = remaining_surplus
        
        return result
    
    def _handle_deficit_power(self,
                            deficit_mw: float,
                            bess_available_mw: float,
                            bess_soc: float,
                            h2_max_mw: float,
                            grid_import_limit: float) -> Dict[str, float]:
        """부족 전력 처리 로직"""
        result = {'bess_discharge_mw': 0, 'h2_fuelcell_mw': 0, 
                 'grid_import_mw': 0, 'load_shedding_mw': 0}
        
        remaining_deficit = deficit_mw
        
        # 우선순위 1: BESS 방전 (SoC 20% 이상에서)
        if remaining_deficit > 0 and bess_soc > 0.2:
            bess_discharge_capacity = min(
                bess_available_mw,
                remaining_deficit / self.efficiencies['dcbus_to_bess']
            )
            result['bess_discharge_mw'] = bess_discharge_capacity
            remaining_deficit -= bess_discharge_capacity * self.efficiencies['dcbus_to_bess']
        
        # 우선순위 2: 연료전지 (H₂ 저장 활용)
        if remaining_deficit > 0 and h2_max_mw > 0:
            h2_fuelcell_power = min(
                h2_max_mw,
                remaining_deficit / self.efficiencies['fc_to_dcbus']
            )
            result['h2_fuelcell_mw'] = h2_fuelcell_power
            remaining_deficit -= h2_fuelcell_power * self.efficiencies['fc_to_dcbus']
        
        # 우선순위 3: 계통 구매 (비상)
        if remaining_deficit > 0:
            grid_import_power = min(
                grid_import_limit,
                remaining_deficit / self.efficiencies['grid_bidirectional']
            )
            result['grid_import_mw'] = grid_import_power
            remaining_deficit -= grid_import_power * self.efficiencies['grid_bidirectional']
        
        # 최후: 부하 차단 (비상사태)
        if remaining_deficit > 0:
            result['load_shedding_mw'] = remaining_deficit
            warnings.warn(f"부하 차단 발생: {remaining_deficit:.2f} MW")
        
        return result
    
    def _calculate_total_losses(self, power_flows: Dict[str, float]) -> float:
        """총 변환 손실 계산"""
        losses = 0
        
        # PV → DC Bus 손실
        pv_loss = power_flows['pv_power_mw'] * (1 - self.efficiencies['pv_to_dcbus'])
        losses += pv_loss
        
        # BESS 충방전 손실
        bess_charge_loss = power_flows['bess_charge_mw'] * (1 - self.efficiencies['dcbus_to_bess'])
        bess_discharge_loss = power_flows['bess_discharge_mw'] * (1 - self.efficiencies['dcbus_to_bess'])
        losses += bess_charge_loss + bess_discharge_loss
        
        # H₂ 시스템 손실
        h2_elec_loss = power_flows['h2_electrolyzer_mw'] * (1 - self.efficiencies['dcbus_to_electrolyzer'])
        h2_fc_loss = power_flows['h2_fuelcell_mw'] * (1 - self.efficiencies['fc_to_dcbus'])
        losses += h2_elec_loss + h2_fc_loss
        
        # 그리드 연계 손실
        grid_import_loss = power_flows['grid_import_mw'] * (1 - self.efficiencies['grid_bidirectional'])
        grid_export_loss = power_flows['grid_export_mw'] * (1 - self.efficiencies['grid_bidirectional'])
        losses += grid_import_loss + grid_export_loss
        
        # AIDC 공급 손실
        aidc_loss = power_flows['aidc_load_mw'] * (1 - self.efficiencies['dcbus_to_aidc'])
        losses += aidc_loss
        
        return losses
    
    def simulate_time_series(self,
                           pv_data: pd.DataFrame,
                           aidc_data: pd.DataFrame,
                           bess_capacity_mw: float = 200,
                           initial_bess_soc: float = 0.5,
                           h2_electrolyzer_mw: float = 50,
                           h2_fuelcell_mw: float = 30) -> pd.DataFrame:
        """
        시계열 전력 균형 시뮬레이션
        
        Args:
            pv_data: PV 발전 데이터 ('power_mw' 컬럼)
            aidc_data: AIDC 부하 데이터 ('total_power_mw' 컬럼)
            bess_capacity_mw: BESS 용량 (MW)
            initial_bess_soc: BESS 초기 SoC
            h2_electrolyzer_mw: 수전해 용량 (MW)  
            h2_fuelcell_mw: 연료전지 용량 (MW)
            
        Returns:
            시계열 전력 흐름 데이터
        """
        # 데이터 길이 맞추기
        min_len = min(len(pv_data), len(aidc_data))
        pv_power = pv_data['power_mw'].iloc[:min_len].values
        aidc_demand = aidc_data['total_power_mw'].iloc[:min_len].values
        
        results = []
        current_bess_soc = initial_bess_soc
        
        for i in range(min_len):
            # 현재 시점 전력 균형 계산
            balance_result = self.calculate_power_balance(
                pv_power_mw=pv_power[i],
                aidc_demand_mw=aidc_demand[i],
                bess_available_mw=bess_capacity_mw,
                bess_soc=current_bess_soc,
                h2_electrolyzer_max_mw=h2_electrolyzer_mw,
                h2_fuelcell_max_mw=h2_fuelcell_mw
            )
            
            # BESS SoC 업데이트 (1시간 기준)
            bess_energy_change = (balance_result['bess_charge_mw'] - 
                                balance_result['bess_discharge_mw'])  # MWh
            
            # BESS 총 에너지 용량 추정 (4시간 방전 가능 가정)
            bess_energy_capacity = bess_capacity_mw * 4  # MWh
            
            current_bess_soc += bess_energy_change / bess_energy_capacity
            current_bess_soc = np.clip(current_bess_soc, 0, 1)
            
            balance_result['bess_soc'] = current_bess_soc
            balance_result['hour'] = i
            results.append(balance_result)
        
        df = pd.DataFrame(results)
        df.set_index('hour', inplace=True)
        
        return df
    
    def get_energy_flows_summary(self, flow_data: pd.DataFrame) -> Dict[str, float]:
        """에너지 흐름 요약 통계"""
        if flow_data.empty:
            return {}
        
        summary = {
            # 총 에너지 (MWh)
            'total_pv_generation_mwh': flow_data['pv_power_mw'].sum(),
            'total_aidc_consumption_mwh': flow_data['aidc_load_mw'].sum(),
            'total_bess_charge_mwh': flow_data['bess_charge_mw'].sum(),
            'total_bess_discharge_mwh': flow_data['bess_discharge_mw'].sum(),
            'total_h2_production_mwh': flow_data['h2_electrolyzer_mw'].sum(),
            'total_h2_consumption_mwh': flow_data['h2_fuelcell_mw'].sum(),
            'total_grid_import_mwh': flow_data['grid_import_mw'].sum(),
            'total_grid_export_mwh': flow_data['grid_export_mw'].sum(),
            'total_curtailment_mwh': flow_data['curtailment_mw'].sum(),
            'total_losses_mwh': flow_data['conversion_loss_mw'].sum(),
            
            # 효율 지표
            'system_efficiency': (flow_data['aidc_load_mw'].sum() / 
                                 flow_data['pv_power_mw'].sum() 
                                 if flow_data['pv_power_mw'].sum() > 0 else 0),
            
            'grid_independence_ratio': (1 - flow_data['grid_import_mw'].sum() / 
                                      flow_data['aidc_load_mw'].sum() 
                                      if flow_data['aidc_load_mw'].sum() > 0 else 0),
            
            'curtailment_ratio': (flow_data['curtailment_mw'].sum() / 
                                flow_data['pv_power_mw'].sum() 
                                if flow_data['pv_power_mw'].sum() > 0 else 0),
            
            # BESS 활용도
            'bess_roundtrip_efficiency': (flow_data['bess_discharge_mw'].sum() / 
                                        flow_data['bess_charge_mw'].sum() 
                                        if flow_data['bess_charge_mw'].sum() > 0 else 0),
            
            'avg_bess_soc': flow_data['bess_soc'].mean() if 'bess_soc' in flow_data.columns else 0,
        }
        
        return summary
    
    def update_converter_efficiency(self, tech: str) -> None:
        """변환기 기술 업데이트"""
        if tech in CONVERTER_EFFICIENCY:
            self.converter_tech = tech
            self.efficiencies = CONVERTER_EFFICIENCY[tech].copy()
        else:
            raise ValueError(f"지원되지 않는 변환기 기술: {tech}")
    
    def clear_history(self) -> None:
        """전력 흐름 기록 초기화"""
        self.power_flows.clear()


# 테스트 코드
if __name__ == "__main__":
    # DC Bus 시스템 생성
    dcbus = DCBusModule(
        bus_voltage="380V",
        converter_tech="default",
        grid_capacity_mw=20
    )
    
    print("DC Bus 시스템 설정:")
    print(f"  버스 전압: {dcbus.bus_voltage}")
    print(f"  변환기 기술: {dcbus.converter_tech}")
    print(f"  그리드 용량: {dcbus.grid_capacity_mw} MW")
    print(f"  PV→DC Bus 효율: {dcbus.efficiencies['pv_to_dcbus']:.1%}")
    
    # 샘플 전력 균형 계산
    sample_result = dcbus.calculate_power_balance(
        pv_power_mw=80,    # PV 발전
        aidc_demand_mw=60,  # AIDC 부하
        bess_available_mw=50,
        bess_soc=0.7,
        h2_electrolyzer_max_mw=30
    )
    
    print(f"\n전력 균형 결과:")
    print(f"  PV 발전: {sample_result['pv_power_mw']:.1f} MW")
    print(f"  AIDC 부하: {sample_result['aidc_load_mw']:.1f} MW") 
    print(f"  BESS 충전: {sample_result['bess_charge_mw']:.1f} MW")
    print(f"  수전해: {sample_result['h2_electrolyzer_mw']:.1f} MW")
    print(f"  변환 손실: {sample_result['conversion_loss_mw']:.1f} MW")
    print(f"  전력 균형: {sample_result['power_balance_mw']:.3f} MW")
    print(f"  에너지 자립: {sample_result['energy_autonomous']}")