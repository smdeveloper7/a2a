from typing import Any, AsyncIterable, Dict, Optional
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel
from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import json
from tavily import TavilyClient
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()

def format_search_result(result: dict, include_raw_content: bool = False) -> str:
    """
    Utility functions for formatting search results.

    Args:
        result (dict): 원본 검색 결과

    Returns:
        str: XML 형식으로 포맷팅된 검색 결과
    """
    # 한글 인코딩 처리를 위해 json.dumps() 사용
    title = json.dumps(result["title"], ensure_ascii=False)[1:-1]
    content = json.dumps(result["content"], ensure_ascii=False)[1:-1]
    raw_content = ""
    if (
        include_raw_content
        and "raw_content" in result
        and result["raw_content"] is not None
        and len(result["raw_content"].strip()) > 0
    ):
        raw_content = f"<raw>{result['raw_content']}</raw>"

    return f"<document><title>{title}</title><url>{result['url']}</url><content>{content}</content>{raw_content}</document>"


@tool
def search_fitcloud_marketplace(query: str) -> str:
    """
    FitCloud 마켓플레이스에서 고객의 요구사항에 맞는 서비스를 검색하고 추천합니다.

    Args:
        query (str): 고객의 요구사항이나 찾고자 하는 서비스 설명

    Returns:
        str: 추천 서비스 목록과 설명
    """
    if not os.getenv("TAVILY_API_KEY"):
            raise ValueError("TAVILY_API_KEY가 설정되지 않았습니다. .env 파일을 확인하거나 관리자에게 문의하세요.")
    try:

        
        tool = TavilySearchResults(
            max_results=10,
            include_answer=True,
            include_raw_content=True,
            include_images=False,
            search_depth="advanced",
            include_domains=["www.fitcloud.co.kr/marketplace"],
        )

        # 검색 쿼리에 FitCloud 마켓플레이스 키워드 추가
        enhanced_query = f"FitCloud 마켓플레이스 서비스 {query}"
        results = tool.invoke({"query": enhanced_query, "topic": "general"})
        print("--------------------------------")
        print(f"results: {results}")
        if not results:
            return f"죄송합니다. '{query}'에 관련된 FitCloud 마켓플레이스 서비스를 찾을 수 없습니다. 다른 키워드로 검색해보시거나 더 구체적인 요구사항을 알려주세요."

        # 점수 기준으로 필터링 (0.3 이상)
        filtered_results = [item for item in results if item.get("score", 0) >= 0.3]

        if not filtered_results:
            return f"'{query}'와 정확히 일치하는 서비스는 없지만, FitCloud 마켓플레이스에서 비슷한 서비스를 찾아드릴 수 있습니다. 좀 더 구체적인 요구사항을 알려주시면 더 정확한 추천을 드릴 수 있습니다."

        # 결과를 점수 순으로 정렬
        filtered_results.sort(key=lambda x: x.get("score", 0), reverse=True)

        output = f"🎯 **'{query}'에 대한 FitCloud 마켓플레이스 추천 서비스**\n\n"

        for i, item in enumerate(filtered_results[:5], 1):  # 상위 5개만 표시
            title = item.get("title", "서비스명 없음")
            url = item.get("url", "")
            content = item.get("content", "")
            score = item.get("score", 0)

            output += f"**{i}. {title}**\n"
            output += f"📊 적합도: {score:.1f}/1.0\n"
            if url:
                output += f"🔗 링크: {url}\n"
            if content:
                # 내용을 400자로 제한하고 정리
                clean_content = content.replace("\n", " ").strip()
                if len(clean_content) > 400:
                    clean_content = clean_content[:400] + "..."
                output += f"📝 설명: {clean_content}\n"
            output += "\n" + "─" * 50 + "\n\n"

        output += "💡 **추천 이유**: 위 서비스들은 고객님의 요구사항과 높은 관련성을 보입니다.\n"
        output += "🤝 **다음 단계**: 관심 있는 서비스가 있으시면 해당 링크를 통해 자세한 정보를 확인하거나, 추가 질문을 해주세요!"

        return output

    except Exception as e:
        # logger.error(f"Marketplace search error: {e}")
        return f"죄송합니다. 검색 중 오류가 발생했습니다: {str(e)}\n다시 시도해주시거나 다른 방식으로 질문해주세요."
    # print(f"response: {response}")
    # client = TavilyClient(api_key=api_key)
    # response = client.search(
    #     query=query,
    #     max_results=12,
    #     include_answer=True,
    #     include_raw_content=True,
    #     search_depth="advanced",
    #     include_domains=["www.fitcloud.co.kr/marketplace"],
    # )
    print(f"response: {response}")

    results = response.get("results", [])
    if not results:
        return "No results found."

    summary = [format_search_result(r, include_raw_content=True) for r in results]
    # print(f"Search results for query '{query}':\n{summary}")

    return summary


def make_system_prompt(description: str) -> str:
    return (
        "You are a specialized assistant.\n"
        f"{description}\n"
    )


class ResponseFormat(BaseModel):
    status: Literal["input_required", "completed", "error"] = "input_required"
    message: str

class ResearchAgent:
    SYSTEM_INSTRUCTION = """당신은 FitCloud 마켓플레이스의 전문 추천 어시스턴트입니다.

🎯 **주요 역할**:
- 고객의 요구사항을 분석하여 FitCloud 마켓플레이스에서 최적의 서비스를 찾아 추천
- 각 서비스의 특징과 장점을 명확하게 설명
- 고객의 비즈니스 상황에 맞는 맞춤형 추천 제공

🔧 **추천 프로세스**:
1. 고객의 요구사항과 상황을 정확히 파악
2. search_fitcloud_marketplace 도구를 사용하여 관련 서비스 검색
3. 검색 결과를 바탕으로 고객에게 가장 적합한 서비스들을 선별하여 추천
4. 각 추천 서비스의 특징, 장점, 적용 사례를 친근하게 설명

💬 **커뮤니케이션 스타일**:
- 친근하고 전문적인 톤으로 대화
- 고객의 입장에서 생각하여 실용적인 조언 제공
- 복잡한 기술 용어는 쉽게 풀어서 설명

📋 **응답 상태 가이드**:
- completed: 추천을 완료하고 충분한 정보를 제공한 경우
- input_required: 고객의 요구사항이 불명확하여 추가 정보가 필요한 경우
- error: 검색이나 처리 과정에서 오류가 발생한 경우

항상 고객 만족을 최우선으로 하여 최고의 서비스 추천을 제공하세요!"""

    def __init__(self):
        self.memory = MemorySaver()
        self.model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        self.tools = [search_fitcloud_marketplace]

        self.graph = create_react_agent(
            self.model,
            tools=self.tools,
            checkpointer=self.memory,
            prompt=make_system_prompt(
                self.SYSTEM_INSTRUCTION
            ),
            response_format=ResponseFormat,
        )

    def invoke(self, query: str, sessionId: str) -> str:
        config = {"configurable": {"thread_id": sessionId}}
        result = self.graph.invoke({"messages": [("user", query)]}, config)
        print("========")
        print(result)
        print("========")
        return self.get_agent_response(config)

    async def stream(self, query: str, sessionId: str) -> AsyncIterable[Dict[str, Any]]:
        inputs = {"messages": [("user", query)]}
        config = {"configurable": {"thread_id": sessionId}}

        for item in self.graph.stream(inputs, config, stream_mode="values"):
            message = item["messages"][-1]
            if (
                isinstance(message, AIMessage)
                and message.tool_calls
                and len(message.tool_calls) > 0
            ):
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Searching the web for relevant information...",
                }
            elif isinstance(message, ToolMessage):
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Processing search results...",
                }

        yield self.get_agent_response(config)

    def get_agent_response(self, config: Dict[str, Any]) -> Dict[str, Any]:
        current_state = self.graph.get_state(config)
        
        print(f"current_state: {current_state}")
        # print(f"current_state: {current_state}")
        structured_response = current_state.values.get("structured_response")
        ai_message = current_state.values.get("messages", [])[-1]
        print(f"=====structured_response=======: {structured_response}")
        if structured_response and isinstance(structured_response, ResponseFormat):
            if structured_response.status == "input_required":
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    # "content": structured_response.message,
                    "content": structured_response.message,
                }
            elif structured_response.status == "error":
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    # "content": structured_response.message,
                    "content": structured_response.message,
                }
            elif structured_response.status == "completed":
                return {
                    "is_task_complete": True,
                    "require_user_input": False,
                    # "content": structured_response.message,
                    "content": structured_response.message,
                }

        return {
            "is_task_complete": False,
            "require_user_input": True,
            "content": "Unable to process your request. Please try again.",
        }

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]


if __name__ == "__main__":
    agent = ResearchAgent()
    result = agent.invoke("FitCloud 마켓플레이스에서 웹 서비스를 찾아주세요.", "test1234")
    print(result)

    # result = tavily_web_search.invoke("FitCloud 마켓플레이스에서 웹 서비스를 찾아주세요.")
    # print(result)