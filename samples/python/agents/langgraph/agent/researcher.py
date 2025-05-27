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
def tavily_web_search(query: str, max_results: Optional[int] = 3) -> str:
    """
    A search engine optimized for comprehensive, accurate, and trusted results.
    Useful for when you need to answer questions about current events.
    Input should be a search query over 3 characters.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY is not set in environment variables.")

    client = TavilyClient(api_key=api_key)
    response = client.search(
        query=query,
        max_results=max_results,
        search_depth="basic",
        include_raw_content=True,
    )

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
        "Do not attempt to answer unrelated questions or use tools for other purposes.\n"
        "Set response status to input_required if the user needs to provide more information.\n"
        "Set response status to error if there is an error while processing the request.\n"
        "Set response status to completed if the request is complete."
    )


class ResponseFormat(BaseModel):
    status: Literal["input_required", "completed", "error"] = "input_required"
    message: str


class ResearchAgent:
    def __init__(self):
        self.memory = MemorySaver()
        self.model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        self.tools = [tavily_web_search]

        self.graph = create_react_agent(
            self.model,
            tools=self.tools,
            checkpointer=self.memory,
            prompt=make_system_prompt(
                "Your sole purpose is to act as a research assistant. "
            ),
            response_format=ResponseFormat,
        )

    def invoke(self, query: str, sessionId: str) -> str:
        config = {"configurable": {"thread_id": sessionId}}
        self.graph.invoke({"messages": [("user", query)]}, config)
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
        structured_response = current_state.values.get("structured_response")
        ai_message = current_state.values.get("messages", [])[-1]

        if structured_response and isinstance(structured_response, ResponseFormat):
            if structured_response.status == "input_required":
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    # "content": structured_response.message,
                    "content": ai_message.content,
                }
            elif structured_response.status == "error":
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    # "content": structured_response.message,
                    "content": ai_message.content,
                }
            elif structured_response.status == "completed":
                return {
                    "is_task_complete": True,
                    "require_user_input": False,
                    # "content": structured_response.message,
                    "content": ai_message.content,
                }

        return {
            "is_task_complete": False,
            "require_user_input": True,
            "content": "Unable to process your request. Please try again.",
        }

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]


if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv

    load_dotenv()

    async def run():
        agent = ResearchAgent()
        async for chunk in agent.stream("솔트웨어에 사업분야에 대해 알려줘", "a"):
            print(chunk["content"])

    asyncio.run(run())
