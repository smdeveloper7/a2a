import logging
import os

import click
import httpx

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, InMemoryPushNotifier
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from agent.researcher import ResearchAgent
from agent_executor import LangGraphAgentExecutor
from dotenv import load_dotenv


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""


@click.command()
@click.option("--host", default="localhost", help="Host to bind.")
@click.option("--port", default=10001, help="Port to run the server on.")
def main(host, port):
    """Starts the Web Search Agent server."""
    try:
        if not os.getenv("GOOGLE_API_KEY") or not os.getenv("TAVILY_API_KEY"):
            raise MissingAPIKeyError("GOOGLE_API_KEY or TAVILY_API_KEY not set in .env")

        agent = ResearchAgent()

        capabilities = AgentCapabilities(streaming=True, pushNotifications=True)

        skills = [
              AgentSkill(
                id="fitcloud_marketplace_recommendation",
                name="FitCloud 마켓플레이스 서비스 추천",
                description="고객의 요구사항을 분석하여 FitCloud 마켓플레이스에서 최적의 서비스를 추천합니다",
                tags=["마켓플레이스", "서비스추천", "FitCloud", "클라우드서비스"],
                examples=[
                    "데이터 분석 솔루션을 찾고 있습니다",
                    "중소기업에 적합한 클라우드 마이그레이션 서비스가 있나요?",
                    "AI/ML 서비스를 도입하고 싶은데 어떤 것이 좋을까요?",
                    "보안 강화를 위한 서비스를 추천해주세요",
                    "비용 효율적인 스토리지 솔루션을 찾고 있습니다",
                ],
            ),
            AgentSkill(
                id="service_comparison",
                name="서비스 비교 및 분석",
                description="여러 FitCloud 서비스들을 비교하여 고객 상황에 가장 적합한 옵션을 제시합니다",
                tags=["서비스비교", "분석", "추천"],
                examples=[
                    "여러 데이터베이스 서비스 중 어떤 것이 좋을까요?",
                    "스타트업과 대기업에 적합한 서비스 차이점을 알려주세요",
                    "비용과 성능을 고려한 최적의 서비스 조합을 추천해주세요",
                ],
            ),
            AgentSkill(
                id="business_consultation",
                name="비즈니스 상담",
                description="고객의 비즈니스 상황과 목표에 맞는 디지털 전환 방안을 제시합니다",
                tags=["비즈니스상담", "디지털전환", "컨설팅"],
                examples=[
                    "온라인 쇼핑몰을 시작하려는데 필요한 서비스가 뭔가요?",
                    "기존 시스템을 클라우드로 이전하려면 어떻게 해야 하나요?",
                    "디지털 마케팅을 위한 도구들을 추천해주세요",
                ],
            ),
        ]
        # skill = AgentSkill(
        #     id="web_search",
        #     name="Web Search Tool",
        #     description="Finds real-time information from the web using Tavily",
        #     tags=["search", "web", "news"],
        #     examples=[
        #         "Find recent news about OpenAI",
        #         "What is the latest on AI regulation?",
        #     ],
        # )
        agent_card = AgentCard(
            name="FitCloud 마켓플레이스 추천 어시스턴트",
            description="고객의 요구사항을 분석하여 FitCloud 마켓플레이스에서 최적의 서비스를 찾아 추천하는 AI 어시스턴트입니다. 클라우드 서비스, AI/ML 솔루션, 데이터 분석 도구 등 다양한 IT 서비스를 고객의 비즈니스 상황에 맞게 추천해드립니다.",
            url=f"http://{host}:{port}/",
            version="1.0.0",
            defaultInputModes=agent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=agent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=skills,
        )

        httpx_client = httpx.AsyncClient()
        request_handler = DefaultRequestHandler(
            agent_executor=LangGraphAgentExecutor(agent),
            task_store=InMemoryTaskStore(),
            push_notifier=InMemoryPushNotifier(httpx_client),
        )

        server = A2AStarletteApplication(
            agent_card=agent_card, http_handler=request_handler
        )

        import uvicorn

        uvicorn.run(server.build(), host=host, port=port)

    except MissingAPIKeyError as e:
        logger.error(f"Missing API key: {e}")
        exit(1)
    except Exception as e:
        logger.exception("Error during WebSearchAgent startup")
        exit(1)


if __name__ == "__main__":
    main()
