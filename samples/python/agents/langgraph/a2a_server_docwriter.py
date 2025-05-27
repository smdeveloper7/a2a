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
from agent.doc_write import DocumentWriterAgent
from agent_executor import LangGraphAgentExecutor
from dotenv import load_dotenv


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""


@click.command()
@click.option("--host", default="localhost", help="Host to bind.")
@click.option("--port", default=10002, help="Port to run the server on.")
def main(host, port):
    """Starts the Web Search Agent server."""
    try:
        if not os.getenv("GOOGLE_API_KEY") or not os.getenv("TAVILY_API_KEY"):
            raise MissingAPIKeyError("GOOGLE_API_KEY or TAVILY_API_KEY not set in .env")

        agent = DocumentWriterAgent()

        capabilities = AgentCapabilities(streaming=True, pushNotifications=True)

        skill = AgentSkill(
            id="document_writer",
            name="Document Writer Tool",
            description="Creates, reads, and edits text documents using a structured toolset. Useful for writing research papers or notes collaboratively.",
            tags=["document", "write", "edit", "read", "arxiv", "research"],
            examples=[
                "Create a document named 'summary.txt' and write a summary of today's meeting.",
                "Edit the document 'draft.txt' by inserting a title at line 1.",
                "Read the contents of 'report.txt' from line 10 to 30.",
            ],
        )
        agent_card = AgentCard(
            name="Document Writer Agent",
            description="A collaborative assistant capable of creating, reading, and editing documents. Designed for tasks such as writing arXiv-style research papers using Gemini.",
            url=f"http://{host}:{port}/",
            version="1.0.0",
            defaultInputModes=agent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=agent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
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
