import pandas as pd
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from langchain.tools import tool
import os
import requests


# --------------------
# 2. 전처리 함수
# --------------------
def preprocess_aws_cost_data(raw_data) -> pd.DataFrame:
# --- 데이터 전처리 ---
# 1. 중복 컬럼 정리 (snake_case로 통일)
# camelCase 컬럼들을 제거하고 snake_case 컬럼들을 유지합니다.
    # duplicate_camel_columns = [
    #     'productRegionName', 'billingEntity', 'lineItemTypeRefine', 'usageFee',
    #     'usageType', 'productRegion', 'billingPeriod', 'serviceName',
    #     'operationDesc', 'priceDescription', 'usageAmount', 'currencyCode'
    # ]
    print(raw_data['body'])
    print("======================")
    df = pd.DataFrame(raw_data['body'])

    # df = df.drop(columns=duplicate_camel_columns)
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
    return df

# --------------------
# 1. Tool: AWS 비용 API 호출 (daily & monthly)
# --------------------
@tool
def aws_costs_daily(from_date: str, to_date: str) -> dict:
    """
    지정된 날짜 범위 내의 AWS 온디맨드 비용 데이터를 가져옵니다.
    """
    token = os.getenv("FITCLOUD_TOKEN", "D5G&4TTXUPK!WOQ4SKBSRNXNT&F&!SA60RBDP5BZM%R6567IB6#SGU2PG!@DEV&~")
    url = "https://aws.fitcloud.co.kr/api/v1/costs/ondemand/corp/daily"
    headers = {"Authorization": f"Bearer {token}"}
    form_data = {"from": from_date, "to": to_date}
    try:
        response = requests.post(url, headers=headers, data=form_data)
        response.raise_for_status()
        result = response.json()
        
        return preprocess_aws_cost_data()
        # return response.json()
    except requests.RequestException as e:
        return {"error": str(e), "status_code": getattr(e.response, 'status_code', None)}

@tool
def aws_costs_monthly(from_date: str, to_date: str):
    """
    지정된 월 범위 내의 AWS 온디맨드 비용 데이터를 가져옵니다.
    """
    token = os.getenv("FITCLOUD_TOKEN", "D5G&4TTXUPK!WOQ4SKBSRNXNT&F&!SA60RBDP5BZM%R6567IB6#SGU2PG!@DEV&~")
    url = "https://aws.fitcloud.co.kr/api/v1/costs/ondemand/corp/monthly"
    headers = {"Authorization": f"Bearer {token}"}
    form_data = {"from": from_date, "to": to_date}
    print(form_data)
    try:
        response = requests.post(url, headers=headers, data=form_data)
        response.raise_for_status()
        return preprocess_aws_cost_data(response.json())
    except requests.RequestException as e:
        return {"error": str(e), "status_code": getattr(e.response, 'status_code', None)}



# --------------------
# 3. 보고서 포맷 함수 (Slack X)
# --------------------
# def generate_report_text(df: pd.DataFrame, year: str, month: str, week: str, cumulative_total: float) -> str:
#     total_cost = df['usage_fee'].sum()
#     cost_by_service = df.groupby('service_name')['usage_fee'].sum().sort_values(ascending=False)

#     lines = []
#     lines.append(f"AI서비스 본부 서비스팀 {year}년 {month}월 {week} AWS 주간 비용 보고서")
#     lines.append(f"총 비용: ${total_cost:.4f}\n")
#     lines.append("서비스별 비용:")
#     for service, fee in cost_by_service.items():
#         lines.append(f"- {service}: ${fee:.4f}")
#     lines.append(f"\n{year}년 {month}월 누적 월 비용: ${cumulative_total:.4f}")
#     return "\n".join(lines)

# # --------------------
# # 4. LangGraph State 정의
# # --------------------
# from langgraph.graph.message import add_messages
# class State(TypedDict):
#     # list 타입에 add_messages 적용(list 에 message 추가)
#     messages: Annotated[list, add_messages]
# # --------------------
# # 5. 노드 정의
# # --------------------
# def extract_dates_node(state: State) -> dict:
#     return {
#         "from_date": "20250501",
#         "to_date": "20250531"
#     }

# # tools = [aws_costs_daily, aws_costs_monthly]
# # tool_node = ToolNode(tools)

# fetch_costs_node = ToolNode([aws_costs_daily])
# fetch_monthly_node = ToolNode([aws_costs_monthly])

# def analyze_and_format_node(state: State) -> dict:
#     df = preprocess_aws_cost_data(state["raw_data"])
#     cumulative_df = preprocess_aws_cost_data(state["cumulative_data"])
#     cumulative_total = cumulative_df['usage_fee'].sum() if 'usage_fee' in cumulative_df else 0.0
#     report = generate_report_text(
#         df,
#         year="2025",
#         month="5",
#         week="2주차",
#         cumulative_total=cumulative_total
#     )
#     return {"summary_report": report}

# --------------------
# 6. LangGraph 구성
# --------------------
# graph = StateGraph(State)
# graph.add_node("extract_dates", extract_dates_node)
# graph.add_node("fetch_costs", fetch_costs_node)
# graph.add_node("fetch_monthly", fetch_monthly_node)
# graph.add_node("analyze_and_format", analyze_and_format_node)

# graph.set_entry_point("extract_dates")
# graph.add_edge("extract_dates", "fetch_costs")
# graph.add_edge("fetch_costs", "fetch_monthly")
# graph.add_edge("fetch_monthly", "analyze_and_format")
# graph.set_finish_point("analyze_and_format")

# compiled = graph.compile()

# --------------------
# 7. 실행 예시
# --------------------
if __name__ == "__main__":
    result = aws_costs_monthly.invoke({"from_date": "202505","to_date": "202505"})
    # print(result)
    # state = {
    #     "messages": [HumanMessage(content="2025년 5월 AWS 비용 알려줘")]
    # }
    # for step in compiled.stream(state):
    #     print("\n[OUTPUT STEP]", step)
    #     if "summary_report" in step:
    #         print("\n✅ 최종 보고서:\n")
    #         print(step["summary_report"])
