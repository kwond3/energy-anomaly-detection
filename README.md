[README 12.20.12 AM.md](https://github.com/user-attachments/files/30274326/README.12.20.12.AM.md)
# 제조업 생산 설비 에너지 비효율 분석 및 이상 탐지

> AI 기술을 활용한 지속가능한 에너지·환경 솔루션 아이디어 공모전 — 예선 제출물

## 프로젝트 개요

제조업 공장 13개 설비에서 수집된 **3,370만 건** 규모의 전력 센서 데이터를 분석하여 에너지 비효율 구간을 정의하고, Isolation Forest 기반 이상 탐지 모델을 설계한 프로젝트입니다.

| 항목 | 내용 |
|------|------|
| 분석 기간 | 2024-12-01 ~ 2025-04-29 (150일) |
| 설비 수 | 13개 |
| 데이터 행 수 | 약 3,370만 건 |
| 수집 주기 | 5초 |
| 주요 기술 | Python, Pandas, Scikit-learn, Isolation Forest |

---

## 분석 흐름

```
데이터 로드 및 전처리
    → 시간대별·설비별 EDA
        → 비효율 구간 정의 (IQR 기반)
            → Isolation Forest 모델 설계
                → 탐지 결과 검증
                    → 에너지·탄소 배출 분석
```

---

## 1. 데이터 전처리

- `timestamp` (epoch ms)와 `localtime` (YYYYMMDDHHMMSS) 두 시간 컬럼 간 8시간 차이 확인 → 미국 DST(일광절약시간) 차이로 판명
- `localtime` 기준으로 분석 진행, `timestamp` 컬럼 제거
- 결측치 없음 확인

```python
df['localtime_parsed'] = pd.to_datetime(df['localtime'].astype(str), format='%Y%m%d%H%M%S')
```

---

## 2. EDA — 설비 운영 패턴 분석

**핵심 발견: 데이터센터 특성 — 24시간 상시 가동**

- 전 설비 평균 `activePower`: **3,008~3,010W** (설비 간 최대 차이 2W)
- 시간대별, 주별, 월별 전력 소비 변동 **1% 미만**
- 피크 시간대 없음 → 시간대 기반 부하 조절보다 **역률 관리 및 이상 탐지**가 효과적

```python
hourly_by_equip = df.groupby(['module(equipment)', 'hour'])['activePower'].mean().unstack()
hourly_range = hourly_by_equip.max(axis=1) - hourly_by_equip.min(axis=1)
# → 전 설비 시간대별 최대-최소 차이 미미
```

---

## 3. 비효율 구간 정의

### 역률(Power Factor) 기반 임계값 도출

- 전체 평균 역률(`pf_avg`): **92.5%**, 표준편차 2.5
- 정규분포 기반 대신 **IQR 방식** 채택 (92~93% 구간에 강하게 집중된 왼쪽 꼬리 분포)

```python
df['pf_avg'] = (df['powerFactorR'] + df['powerFactorS'] + df['powerFactorT']) / 3

Q1 = df['pf_avg'].quantile(0.25)
Q3 = df['pf_avg'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR  # → 85.42%

inefficient = df[df['pf_avg'] < lower_bound]
# → 55,336건 (전체의 0.16%)
```

### 설비별 역률 급락 이벤트 집중도

| 설비 | 역률 급락 이벤트 | 비중 |
|------|----------------|------|
| 예비건조기 | 26,493건 | 47.9% |
| 6호기 | 25,944건 | 46.9% |
| 나머지 11개 설비 | 239~285건 (각) | 5.2% |

> 역률 급락 시 `activePower`, 전압, 전류가 정상 범위를 유지 → 역률 보상 회로(콘덴서 뱅크) 이상 또는 역률 기록 장치 오류 가능성

---

## 4. Isolation Forest 이상 탐지 모델

### Feature 구성

| Feature | 유형 | 선택 근거 |
|---------|------|----------|
| `reactivePowerLagging` | 원본 | 역률과 직접 연관 |
| `voltageRS` | 원본 | 전력품질 이상 감지 |
| `currentR` | 원본 | 부하 변동 반영 |
| `powerFactorR/S/T` | 원본 | 3상 역률 직접 측정값 |
| `pf_lag1`, `pf_lag2` | 파생 | 5·10초 전 역률 — 시계열 맥락 주입 |
| `pf_rolling_std3` | 파생 | 15초 역률 변동성 — 급락 패턴 포착 |

> `activePower`는 역률 급락 시 변화가 없어 구분력 낮음 → 제외

```python
df_sorted['pf_lag1'] = df_sorted.groupby('module(equipment)')['pf_avg'].shift(1)
df_sorted['pf_lag2'] = df_sorted.groupby('module(equipment)')['pf_avg'].shift(2)
df_sorted['pf_rolling_std3'] = df_sorted.groupby('module(equipment)')['pf_avg'].transform(
    lambda x: x.rolling(3, min_periods=1).std()
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_model[features])

iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.0016,  # EDA에서 도출된 역률 급락 비율 0.16%
    random_state=42,
    n_jobs=-1
)
```

---

## 5. 탐지 결과

| 항목 | 값 |
|------|-----|
| 탐지된 총 이상치 | 53,914건 |
| 전체 대비 비율 | 0.160% |
| 이상치 구간 역률 중앙값 | 87.3% |
| 예비건조기·6호기 집중도 | **88.6%** |

### 설비별 이상치 분포

| 설비 | 이상치 건수 | 비율 | 이상치 역률 중앙값 |
|------|-----------|------|-----------------|
| 예비건조기 | 36,149건 | 67.0% | 70.8% |
| 6호기 | 11,652건 | 21.6% | 86.4% |
| 나머지 11개 설비 | 6,113건 | 11.3% | 정상 범위 (다른 feature 패턴) |

### 모델 타당성 검증 (비지도학습 환경)

| 검증 방법 | 결과 |
|----------|------|
| EDA 결과와 교차 검증 | 예비건조기·6호기 집중 패턴 일치 |
| Anomaly Score 분포 | 이상 구간 score가 정상 구간 대비 유의미하게 낮음 |
| 이상치 구간 역률 분포 | 도메인 지식 부합 (87.3% vs 정상 92.5%) |

---

## 6. 에너지 소비 및 탄소 배출 분석

```python
# accumActiveEnergy 증분으로 실제 소비량 산출
df_sorted['energy_diff'] = df_sorted.groupby('module(equipment)')['accumActiveEnergy'].diff()
df_sorted['energy_diff'] = df_sorted['energy_diff'].clip(lower=0)

# 증분 평균 4.18Wh = 이론값 (3,009W × 5초 / 3600) → 데이터 신뢰성 검증

EMISSION_FACTOR = 0.4557  # kgCO2eq/kWh (한국 2023년 배출계수)
```

| 지표 | 값 |
|------|-----|
| 총 에너지 소비 | 140,866 kWh |
| 일평균 에너지 소비 | 932.9 kWh |
| 총 탄소 배출 | 64.19 tCO₂eq |
| 일평균 탄소 배출 | 425.1 kgCO₂eq |

---

## 7. AI 기반 에너지 관리 아이디어

**① 실시간 역률 이상 감지 시스템**
- Isolation Forest 모델을 스트리밍 파이프라인(Apache Kafka)에 통합
- 5초 단위 실시간 역률 모니터링 → 이상 탐지 시 즉시 알림

**② 역률 보상 자동 제어 (Adaptive Capacitor Control)**
- AI 모델이 역률 급락 패턴 사전 예측 (예측 지평: 1~5분)
- 콘덴서 뱅크 선제적 투입·조정 → 무효전력 손실 약 30% 감소 목표

**③ 탄소 배출 저감 계획 수립**
- 역률 95% 유지 시 연간 탄소 배출 **5~8% 감소** 가능
- LSTM, Prophet 등 예측 모델과 연계한 중장기 탄소 감축 시나리오 자동 생성

---

## 파일 구조

```
├── contest.ipynb     # 전체 분석 코드
└── README.md
```

---

## 기술 스택

`Python` `Pandas` `Scikit-learn` `Isolation Forest` `Matplotlib` `NumPy`
