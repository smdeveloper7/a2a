from langchain.tools import tool
import requests
import os

@tool
def aws_costs_daily(from_date: str, to_date: str) -> dict:
    """
    지정된 날짜 범위 내의 AWS 온디맨드 비용 데이터를 가져옵니다.
    
    Parameters:
    - from_date: 조회 시작일 (형식: YYYYMMDD)
    - to_date: 조회 종료일 (형식: YYYYMMDD)
    
    Returns:
    - JSON 형태의 비용 분석 데이터
    """
    # 환경변수에서 토큰 읽기
    token = os.getenv("FITCLOUD_TOKEN","D5G&4TTXUPK!WOQ4SKBSRNXNT&F&!SA60RBDP5BZM%R6567IB6#SGU2PG!@DEV&~")
    if not token:
        raise ValueError("환경 변수 'FITCLOUD_TOKEN'을 찾을 수 없습니다. .env 파일 또는 환경 설정에서 값을 등록해주세요.")
    
    url = "https://aws.fitcloud.co.kr/api/v1/costs/ondemand/corp/daily"
    
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    form_data = {
        "from": from_date,
        "to": to_date
    }

    try:
        response = requests.post(url, headers=headers, data=form_data)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {
            "error": str(e),
            "status_code": getattr(e.response, 'status_code', None),
            "response_text": getattr(e.response, 'text', None)
        }


@tool
def aws_costs_monthly(from_date: str, to_date: str) -> dict:
    """
    지정된 날짜 범위 내의 AWS 온디맨드 비용 데이터를 가져옵니다.
    
    Parameters:
    - from_date: 조회 시작일 (형식: YYYYMMDD)
    - to_date: 조회 종료일 (형식: YYYYMMDD)
    
    Returns:
    - JSON 형태의 비용 분석 데이터
    """
    # 환경변수에서 토큰 읽기
    token = os.getenv("FITCLOUD_TOKEN","D5G&4TTXUPK!WOQ4SKBSRNXNT&F&!SA60RBDP5BZM%R6567IB6#SGU2PG!@DEV&~")
    if not token:
        raise ValueError("환경 변수 'FITCLOUD_TOKEN'을 찾을 수 없습니다. .env 파일 또는 환경 설정에서 값을 등록해주세요.")
    
    url = "https://aws.fitcloud.co.kr/api/v1/costs/ondemand/corp/daily"
    
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    form_data = {
        "from": from_date,
        "to": to_date
    }

    try:
        response = requests.post(url, headers=headers, data=form_data)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {
            "error": str(e),
            "status_code": getattr(e.response, 'status_code', None),
            "response_text": getattr(e.response, 'text', None)
        }
