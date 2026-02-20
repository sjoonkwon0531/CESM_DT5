"""
M10. 기상 모듈 (Weather Module)
한국 중부 지역 기준 합성 TMY 데이터 생성
일사량(GHI), 온도, 습도 시계열 제공
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
import math

from config import WEATHER_PARAMS, MONTHLY_GHI_PATTERN, MONTHLY_TEMP_PATTERN


class WeatherModule:
    """기상 데이터 생성 및 관리 모듈"""
    
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.params = WEATHER_PARAMS.copy()
        
    def generate_tmy_data(self, 
                         year: int = 2024,
                         noise_level: float = 0.1) -> pd.DataFrame:
        """
        한국 중부 기준 합성 TMY (Typical Meteorological Year) 데이터 생성
        
        Args:
            year: 기준 연도
            noise_level: 노이즈 레벨 (0~1, 기본값 0.1)
            
        Returns:
            시간별 기상 데이터 DataFrame (8760 시간)
        """
        # 시간 인덱스 생성 (1년 8760시간)
        start_date = datetime(year, 1, 1)
        dates = pd.date_range(start_date, periods=8760, freq='h')
        
        # 기본 데이터프레임 생성
        weather_data = pd.DataFrame(index=dates)
        weather_data['hour'] = weather_data.index.hour
        weather_data['day_of_year'] = weather_data.index.dayofyear  
        weather_data['month'] = weather_data.index.month
        
        # 일사량(GHI) 생성
        weather_data['ghi_w_per_m2'] = self._generate_ghi_profile(
            weather_data, noise_level
        )
        
        # 온도 생성  
        weather_data['temp_celsius'] = self._generate_temperature_profile(
            weather_data, noise_level
        )
        
        # 습도 생성
        weather_data['humidity_percent'] = self._generate_humidity_profile(
            weather_data, noise_level
        )
        
        # 풍속 생성 (패널 냉각 효과용)
        weather_data['wind_speed_ms'] = self._generate_wind_profile(
            weather_data, noise_level
        )
        
        self.data = weather_data
        return weather_data
    
    def _generate_ghi_profile(self, df: pd.DataFrame, noise: float) -> np.ndarray:
        """전천일사량(GHI) 프로파일 생성"""
        n_hours = len(df)
        ghi = np.zeros(n_hours)
        
        for i, (idx, row) in enumerate(df.iterrows()):
            month = row['month']
            hour = row['hour'] 
            day_of_year = row['day_of_year']
            
            # 월별 기본 일사량 패턴
            monthly_factor = MONTHLY_GHI_PATTERN[month]
            
            # 일간 태양각도 패턴 (코사인 함수)
            if 6 <= hour <= 18:  # 일출~일몰
                solar_angle = math.cos(math.pi * (hour - 12) / 12)
                daily_pattern = max(0, solar_angle) ** 1.2  # 비선형 보정
            else:
                daily_pattern = 0
            
            # 연간 변동 (지구 공전 궤도)
            annual_variation = 1 + 0.033 * math.cos(2 * math.pi * day_of_year / 365.25)
            
            # 기본 일사량 계산
            base_ghi = 1000 * monthly_factor * daily_pattern * annual_variation
            
            # 구름 효과 (확률적 감소)  
            cloud_factor = self._generate_cloud_factor(month, noise)
            
            ghi[i] = max(0, base_ghi * cloud_factor)
        
        return ghi
    
    def _generate_cloud_factor(self, month: int, noise: float) -> float:
        """구름에 의한 일사량 감소 인자 (확률적)"""
        # 월별 구름 패턴 (7월 장마철 고려)
        cloud_probability = {
            1: 0.4, 2: 0.4, 3: 0.5, 4: 0.5, 5: 0.4, 6: 0.5,
            7: 0.7, 8: 0.6, 9: 0.4, 10: 0.3, 11: 0.4, 12: 0.4
        }
        
        base_cloud_prob = cloud_probability.get(month, 0.4)
        
        # 랜덤 구름 생성
        rand = np.random.random()
        if rand < base_cloud_prob:
            # 구름 있음: 일사량 10-90% 감소
            reduction = 0.1 + 0.8 * np.random.random()
            return 1 - reduction
        else:
            # 맑음: 노이즈만 적용
            return 1 - noise * 0.2 * (np.random.random() - 0.5)
    
    def _generate_temperature_profile(self, df: pd.DataFrame, noise: float) -> np.ndarray:
        """온도 프로파일 생성"""
        n_hours = len(df)
        temp = np.zeros(n_hours)
        
        for i, (idx, row) in enumerate(df.iterrows()):
            month = row['month']
            hour = row['hour']
            day_of_year = row['day_of_year']
            
            # 월별 평균 온도
            monthly_avg = MONTHLY_TEMP_PATTERN[month]
            
            # 일간 온도 변화 (코사인 함수, 14시 최고)
            daily_variation = 8 * math.cos(math.pi * (hour - 14) / 12)
            
            # 연간 온도 변화 미세 조정
            annual_adj = 2 * math.cos(2 * math.pi * day_of_year / 365.25)
            
            # 랜덤 노이즈
            temp_noise = noise * 5 * (np.random.random() - 0.5)
            
            temp[i] = monthly_avg + daily_variation + annual_adj + temp_noise
        
        return temp
    
    def _generate_humidity_profile(self, df: pd.DataFrame, noise: float) -> np.ndarray:
        """습도 프로파일 생성"""
        n_hours = len(df)
        humidity = np.zeros(n_hours)
        
        for i, (idx, row) in enumerate(df.iterrows()):
            month = row['month']
            hour = row['hour']
            temp = df.iloc[i]['temp_celsius'] if i < len(df) else 20
            
            # 월별 기본 습도 (여름 높음, 겨울 낮음)
            base_humidity = {
                1: 45, 2: 50, 3: 55, 4: 60, 5: 65, 6: 75,
                7: 85, 8: 80, 9: 70, 10: 60, 11: 55, 12: 50
            }
            
            monthly_base = base_humidity.get(month, 60)
            
            # 일간 변화 (새벽 높음, 오후 낮음)
            daily_variation = 15 * math.cos(math.pi * (hour - 6) / 12)
            
            # 온도 반비례 효과
            temp_effect = -0.5 * (temp - 20)
            
            # 노이즈
            humidity_noise = noise * 10 * (np.random.random() - 0.5)
            
            humidity[i] = np.clip(
                monthly_base + daily_variation + temp_effect + humidity_noise,
                10, 95
            )
        
        return humidity
    
    def _generate_wind_profile(self, df: pd.DataFrame, noise: float) -> np.ndarray:
        """풍속 프로파일 생성 (패널 냉각 효과용)"""
        n_hours = len(df)
        wind_speed = np.zeros(n_hours)
        
        for i in range(n_hours):
            # 기본 풍속 (2-5 m/s)
            base_wind = 2.5 + 1.5 * np.random.random()
            
            # 계절별 조정 (겨울 강풍)
            month = df.iloc[i]['month']
            seasonal_factor = 1.3 if month in [12, 1, 2] else 1.0
            
            # 노이즈
            wind_noise = noise * 2 * (np.random.random() - 0.5)
            
            wind_speed[i] = max(0.5, base_wind * seasonal_factor + wind_noise)
        
        return wind_speed
    
    def get_weather_at_time(self, hour_index: int) -> Dict[str, float]:
        """특정 시간의 기상 데이터 반환"""
        if self.data is None:
            raise ValueError("기상 데이터가 생성되지 않았습니다. generate_tmy_data()를 먼저 호출하세요.")
        
        if not (0 <= hour_index < len(self.data)):
            raise IndexError(f"시간 인덱스가 범위를 벗어났습니다: {hour_index}")
        
        row = self.data.iloc[hour_index]
        return {
            'ghi_w_per_m2': row['ghi_w_per_m2'],
            'temp_celsius': row['temp_celsius'], 
            'humidity_percent': row['humidity_percent'],
            'wind_speed_ms': row['wind_speed_ms']
        }
    
    def save_to_csv(self, filepath: str) -> None:
        """기상 데이터를 CSV 파일로 저장"""
        if self.data is None:
            raise ValueError("저장할 기상 데이터가 없습니다.")
        
        self.data.to_csv(filepath)
        print(f"기상 데이터가 저장되었습니다: {filepath}")
    
    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """CSV 파일에서 기상 데이터 로드"""
        self.data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        return self.data
    
    def get_statistics(self) -> Dict[str, float]:
        """기상 데이터 통계 반환"""
        if self.data is None:
            return {}
        
        stats = {}
        for col in ['ghi_w_per_m2', 'temp_celsius', 'humidity_percent', 'wind_speed_ms']:
            stats[f'{col}_mean'] = self.data[col].mean()
            stats[f'{col}_max'] = self.data[col].max()  
            stats[f'{col}_min'] = self.data[col].min()
            stats[f'{col}_std'] = self.data[col].std()
        
        # 연간 일사량 합계 (kWh/m²)
        stats['annual_ghi_kwh_per_m2'] = self.data['ghi_w_per_m2'].sum() / 1000
        
        return stats


# 테스트 코드
if __name__ == "__main__":
    weather = WeatherModule()
    data = weather.generate_tmy_data()
    
    print("기상 데이터 생성 완료:")
    print(f"데이터 포인트 수: {len(data)}")
    print(f"일사량 범위: {data['ghi_w_per_m2'].min():.1f} ~ {data['ghi_w_per_m2'].max():.1f} W/m²")
    print(f"온도 범위: {data['temp_celsius'].min():.1f} ~ {data['temp_celsius'].max():.1f} °C")
    print(f"연간 일사량: {data['ghi_w_per_m2'].sum()/1000:.0f} kWh/m²")