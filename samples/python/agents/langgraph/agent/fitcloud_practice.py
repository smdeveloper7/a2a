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
def preprocess_aws_cost_data(raw_data: dict) -> pd.DataFrame:
    df = pd.DataFrame(raw_data)
    duplicate_camel_columns = [
        'productRegionName', 'billingEntity', 'lineItemTypeRefine', 'usageFee',
        'usageType', 'productRegion', 'billingPeriod', 'serviceName',
        'operationDesc', 'priceDescription', 'usageAmount', 'currencyCode'
    ]
    df = df.drop(columns=[col for col in duplicate_camel_columns if col in df.columns], errors='ignore')
    for col in ['usage_fee', 'usage_amount']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
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
        return preprocess_aws_cost_data(response.json())
        # return response.json()
    except requests.RequestException as e:
        return {"error": str(e), "status_code": getattr(e.response, 'status_code', None)}

@tool
def aws_costs_monthly(from_date: str, to_date: str) -> dict:
    """
    지정된 월 범위 내의 AWS 온디맨드 비용 데이터를 가져옵니다.
    """
    token = os.getenv("FITCLOUD_TOKEN", "D5G&4TTXUPK!WOQ4SKBSRNXNT&F&!SA60RBDP5BZM%R6567IB6#SGU2PG!@DEV&~")
    url = "https://aws.fitcloud.co.kr/api/v1/costs/ondemand/corp/month"
    headers = {"Authorization": f"Bearer {token}"}
    form_data = {"from": from_date, "to": to_date}
    try:
        response = requests.post(url, headers=headers, data=form_data)
        response.raise_for_status()
        return response.json()
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
    result = aws_costs_daily.invoke({"from_date": "20250501", "to_date": "20250531"})
    print(result)
    # state = {
    #     "messages": [HumanMessage(content="2025년 5월 AWS 비용 알려줘")]
    # }
    # for step in compiled.stream(state):
    #     print("\n[OUTPUT STEP]", step)
    #     if "summary_report" in step:
    #         print("\n✅ 최종 보고서:\n")
    #         print(step["summary_report"])
