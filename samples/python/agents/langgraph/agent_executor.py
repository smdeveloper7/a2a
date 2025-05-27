import logging

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import Event, EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InternalError,
    InvalidParamsError,
    Part,
    Task,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import (
    new_agent_text_message,
    new_task,
)
from a2a.utils.errors import ServerError
from typing import Any, AsyncIterable, Dict
from typing_extensions import override


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LangGraphAgentExecutor(AgentExecutor):
    def __init__(self, agent: Any, artifact_name: str = "result"):
        self.agent = agent
        self.artifact_name = artifact_name

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        if not context.get_user_input():
            raise ServerError(error=InvalidParamsError())

        query = context.get_user_input()
        task = context.current_task or new_task(context.message)
        event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.contextId)

        try:
            async for item in self.agent.stream(query, task.contextId):
                print("query")
                is_task_complete = item.get("is_task_complete", False)
                require_user_input = item.get("require_user_input", False)
                content = item.get("content", "")

                if not is_task_complete and not require_user_input:
                    updater.update_status(
                        TaskState.working,
                        new_agent_text_message(content, task.contextId, task.id),
                    )
                elif require_user_input:
                    updater.update_status(
                        TaskState.input_required,
                        new_agent_text_message(content, task.contextId, task.id),
                        final=True,
                    )
                    break
                else:
                    updater.add_artifact(
                        [Part(root=TextPart(text=content))],
                        name=self.artifact_name,
                    )
                    updater.complete()
                    break
        except Exception as e:
            logger.exception("Agent execution failed")
            raise ServerError(error=InternalError()) from e

    async def cancel(
        self, request: RequestContext, event_queue: EventQueue
    ) -> Task | None:
        raise ServerError(error=UnsupportedOperationError())
