# api_key = "D5G&4TTXUPK!WOQ4SKBSRNXNT&F&!SA60RBDP5BZM%R6567IB6#SGU2PG!@DEV&~"

import pandas as pd
import json

with open("fee.json", 'r', encoding='utf-8') as f:
    data = json.load(f) # 파일 객체를 json.load()에 전달하여 파싱

# JSON 데이터를 파싱하여 리스트로 변환
# data = json.loads(aws_billing_data_json)

# DataFrame 생성
df = pd.DataFrame(data)

# --- 데이터 전처리 ---
# 1. 중복 컬럼 정리 (snake_case로 통일)
# camelCase 컬럼들을 제거하고 snake_case 컬럼들을 유지합니다.
duplicate_camel_columns = [
    'productRegionName', 'billingEntity', 'lineItemTypeRefine', 'usageFee',
    'usageType', 'productRegion', 'billingPeriod', 'serviceName',
    'operationDesc', 'priceDescription', 'usageAmount', 'currencyCode'
]
df = df.drop(columns=duplicate_camel_columns)

# 2. 숫자형 데이터 타입 변환 (문자열 -> 숫자)
numeric_cols = ['usage_fee', 'usage_amount']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce') # 오류 발생 시 NaN으로 처리

# 3. 필요한 컬럼만 선택하여 간결하게 보기
# 모든 컬럼을 활용하되, 분석에 더 초점을 맞추어 순서를 조정하거나 일부를 제거할 수 있습니다.
# 여기서는 모든 컬럼을 사용하되, 가독성을 위해 상위 몇 개만 출력합니다.

print("--- 원본 데이터 (전처리 후 상위 5행) ---")
print(df.head())
print("\n데이터프레임 정보:")
df.info()

# --- 데이터 분석 및 요약 ---

print("\n\n--- 2025년 6월 AWS 월 요금 요약 ---")

# 1. 전체 총 요금
total_cost_usd = df['usage_fee'].sum()
print(f"\n1. 2025년 6월 전체 총 요금: ${total_cost_usd:.4f} USD")

# 2. 서비스별 요금
cost_by_service = df.groupby('service_name')['usage_fee'].sum().sort_values(ascending=False)
print("\n2. 서비스별 총 요금 (USD):")
print(cost_by_service.apply(lambda x: f"${x:.4f}"))

# 3. 리전(Region)별 요금
cost_by_region = df.groupby('product_region_name')['usage_fee'].sum().sort_values(ascending=False)
print("\n3. 리전(Region)별 총 요금 (USD):")
print(cost_by_region.apply(lambda x: f"${x:.4f}"))

# 4. 사용 유형(Usage Type)별 요금 (상위 5개)
cost_by_usage_type = df.groupby('usage_type')['usage_fee'].sum().sort_values(ascending=False).head(5)
print("\n4. 사용 유형(Usage Type)별 총 요금 (상위 5개, USD):")
print(cost_by_usage_type.apply(lambda x: f"${x:.4f}"))

# 5. 상세 요금 내역 (서비스, 리전, 사용 유형별)
# 총 요금과 총 사용량을 함께 보여줍니다.
detailed_cost_breakdown = df.groupby(['service_name', 'product_region_name', 'usage_type']).agg(
    Total_Fee=('usage_fee', 'sum'),
    Total_Usage=('usage_amount', 'sum')
).sort_values(by='Total_Fee', ascending=False)

print("\n5. 서비스, 리전, 사용 유형별 상세 요금 내역:")
# 가독성을 위해 float 형식 지정
detailed_cost_breakdown['Total_Fee'] = detailed_cost_breakdown['Total_Fee'].apply(lambda x: f"${x:.8f}")
detailed_cost_breakdown['Total_Usage'] = detailed_cost_breakdown['Total_Usage'].apply(lambda x: f"{x:.8f}")
print(detailed_cost_breakdown)

print("\n\n--- 분석 요약 ---")
print("이 데이터는 2025년 6월 한 달간의 AWS 서비스 사용 요금 내역입니다.")
print(f"총 청구된 금액은 {total_cost_usd:.4f} USD 입니다.")
print("\n주요 비용 발생 서비스 및 사용 유형:")
if not cost_by_service.empty:
    top_service = cost_by_service.index[0]
    top_service_cost = cost_by_service.iloc[0]
    print(f"- 가장 많은 비용이 발생한 서비스는 '{top_service}'로, ${top_service_cost:.4f}를 차지했습니다.")
if not cost_by_region.empty:
    top_region = cost_by_region.index[0]
    top_region_cost = cost_by_region.iloc[0]
    print(f"- 가장 많은 비용이 발생한 리전은 '{top_region}'로, ${top_region_cost:.4f}를 차지했습니다.")
if not cost_by_usage_type.empty:
    top_usage_type = cost_by_usage_type.index[0]
    top_usage_type_cost = cost_by_usage_type.iloc[0]
    print(f"- 특정 사용 유형 중에서는 '{top_usage_type}'가 가장 많은 요금을 발생시켰습니다.")

# print("\n데이터를 통해 볼 때, AWS는 서비스(Amplify, Bedrock 등), 리전(서울, 오레곤 등), 그리고 각 서비스 내의 세부 사용량(데이터 저장, 전송, 토큰 사용량 등)별로 매우 정밀하게 비용을 책정하고 있음을 알 수 있습니다. 특히, Bedrock을 통해 사용된 'Claude 3.5 Sonnet' 모델의 입력 토큰 비용이 전체 요금의 상당 부분을 차지하고 있습니다.")