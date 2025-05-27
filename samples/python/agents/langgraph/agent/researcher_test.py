import os
from dotenv import load_dotenv
from typing import Annotated, Sequence, Optional, Union, Type, Any
from typing_extensions import TypedDict
import operator

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState

from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_experimental.utilities import PythonREPL

from langchain_teddynote.tools.tavily import TavilySearch

# === 환경 변수 로드 ===
load_dotenv()

# === 모델 이름 ===
MODEL_NAME = "gpt-4o-mini"


## 1.상태정의
class AgentState(TypedDict):
    messages: Annotated[
        Sequence[BaseMessage], operator.add
    ]  # Agent 간 공유하는 메시지 목록
    sender: Annotated[str, "The sender of the last message"]  # 마지막 메시지의 발신자


## 2.도구 정의
tavily_tool = TavilySearch(max_results=5)
python_repl = PythonREPL()


# Python 코드를 실행하는 도구 정의
@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        # 주어진 코드를 Python REPL에서 실행하고 결과 반환
        result = python_repl.run(code)
    except BaseException as e:
        return f"Failed to execute code. Error: {repr(e)}"
    # 실행 성공 시 결과와 함께 성공 메시지 반환
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )


# === 시스템 프롬프트 ===
def make_system_prompt(suffix: str) -> str:
    return (
        "You are a helpful AI assistant, collaborating with other assistants."
        " Use the provided tools to progress towards answering the question."
        " If you are unable to fully answer, that's OK, another assistant with different tools "
        " will help where you left off. Execute what you can to make progress."
        " If you or any of the other assistants have the final answer or deliverable,"
        " prefix your response with FINAL ANSWER so the team knows to stop."
        f"\n{suffix}"
    )


# === 에이전트 정의 ===
llm = ChatOpenAI(model=MODEL_NAME)

## Research Agent 생성
research_agent = create_react_agent(
    llm,
    tools=[tavily_tool],
    prompt=make_system_prompt(
        "You can only do research. You are working with a chart generator colleague."
    ),
)


# Research Agent 노드 정의
def research_node(state: MessagesState) -> MessagesState:
    result = research_agent.invoke(state)

    # 마지막 메시지를 HumanMessage 로 변환
    last_message = HumanMessage(
        content=result["messages"][-1].content, name="researcher"
    )
    return {
        # Research Agent 의 메시지 목록 반환
        "messages": [last_message],
    }


chart_generator_system_prompt = """
You can only generate charts. You are working with a researcher colleague.
Be sure to use the following font code in your code when generating charts.

##### 폰트 설정 #####
import platform

# OS 판단
current_os = platform.system()

if current_os == "Windows":
    # Windows 환경 폰트 설정
    font_path = "C:/Windows/Fonts/malgun.ttf"  # 맑은 고딕 폰트 경로
    fontprop = fm.FontProperties(fname=font_path, size=12)
    plt.rc("font", family=fontprop.get_name())
elif current_os == "Darwin":  # macOS
    # Mac 환경 폰트 설정
    plt.rcParams["font.family"] = "AppleGothic"
else:  # Linux 등 기타 OS
    # 기본 한글 폰트 설정 시도
    try:
        plt.rcParams["font.family"] = "NanumGothic"
    except:
        print("한글 폰트를 찾을 수 없습니다. 시스템 기본 폰트를 사용합니다.")

##### 마이너스 폰트 깨짐 방지 #####
plt.rcParams["axes.unicode_minus"] = False  # 마이너스 폰트 깨짐 방지
"""

# Chart Generator Agent 생성
chart_agent = create_react_agent(
    llm,
    [python_repl_tool],
    prompt=make_system_prompt(chart_generator_system_prompt),
)


# === 상태 노드 정의 ===
def research_node(state: MessagesState) -> MessagesState:
    result = research_agent.invoke(state)
    return {
        "messages": [
            HumanMessage(content=result["messages"][-1].content, name="researcher")
        ]
    }


def chart_node(state: MessagesState) -> MessagesState:
    result = chart_agent.invoke(state)

    # 마지막 메시지를 HumanMessage 로 변환
    last_message = HumanMessage(
        content=result["messages"][-1].content, name="chart_generator"
    )
    return {
        # share internal message history of chart agent with other agents
        "messages": [last_message],
    }


## 라우터 정의
def router(state: MessagesState):
    # This is the router
    messages = state["messages"]
    last_message = messages[-1]
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        print("Final answer received, ending workflow.")
        return END
    else:
        print("Continuing workflow, no final answer yet.")
        print(last_message.content)
    return "continue"


# === app 반환 함수 ===
def build_chart_research_app():
    workflow = StateGraph(MessagesState)

    workflow.add_node("researcher", research_node)
    workflow.add_node("chart_generator", chart_node)

    workflow.add_conditional_edges(
        "researcher", router, {"continue": "chart_generator", END: END}
    )
    workflow.add_conditional_edges(
        "chart_generator", router, {"continue": "researcher", END: END}
    )

    workflow.add_edge(START, "researcher")

    # Checkpointer는 Memory 기반
    app = workflow.compile(checkpointer=MemorySaver())
    return app


if __name__ == "__main__":
    from langchain_core.runnables import RunnableConfig
    from langchain_teddynote.messages import random_uuid, invoke_graph

    app = build_chart_research_app()
    config = RunnableConfig(
        recursion_limit=10, configurable={"thread_id": random_uuid()}
    )

    # 질문 입력
    inputs = {
        "messages": [
            HumanMessage(
                content="2010년 ~ 2024년까지의 대한민국의 1인당 GDP 추이를 그래프로 시각화 해주세요."
            )
        ],
    }

    # 그래프 실행
    invoke_graph(
        app, inputs, config, node_names=["researcher", "chart_generator", "agent"]
    )
