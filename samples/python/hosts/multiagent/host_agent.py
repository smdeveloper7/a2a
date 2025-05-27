import base64
import json
import uuid

from typing import List

import httpx

from a2a.client import A2ACardResolver
from a2a.types import (
    AgentCard,
    DataPart,
    Message,
    MessageSendConfiguration,
    MessageSendParams,
    Part,
    Task,
    TaskState,
    TextPart,
)
from google.adk import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from .remote_agent_connection import RemoteAgentConnections, TaskUpdateCallback


class HostAgent:
    """The host agent.

    This is the agent responsible for choosing which remote agents to send
    tasks to and coordinate their work.
    """

    def __init__(
        self,
        remote_agent_addresses: list[str],
        http_client: httpx.AsyncClient,
        task_callback: TaskUpdateCallback | None = None,
    ):
        self.task_callback = task_callback
        self.httpx_client = http_client
        self.remote_agent_connections: dict[str, RemoteAgentConnections] = {}
        self.cards: dict[str, AgentCard] = {}
        for address in remote_agent_addresses:
            card_resolver = A2ACardResolver(http_client, address)
            card = card_resolver.get_agent_card()
            print(card)
            remote_connection = RemoteAgentConnections(http_client, card)
            self.remote_agent_connections[card.name] = remote_connection
            self.cards[card.name] = card
        agent_info = []
        for ra in self.list_remote_agents():
            agent_info.append(json.dumps(ra))
        self.agents = '\n'.join(agent_info)

    def register_agent_card(self, card: AgentCard):
        remote_connection = RemoteAgentConnections(self.httpx_client, card)
        self.remote_agent_connections[card.name] = remote_connection
        self.cards[card.name] = card
        agent_info = []
        for ra in self.list_remote_agents():
            agent_info.append(json.dumps(ra))
        self.agents = '\n'.join(agent_info)

    def create_agent(self) -> Agent:
        return Agent(
            model='gemini-2.0-flash-001',
            name='host_agent',
            instruction=self.root_instruction,
            before_model_callback=self.before_model_callback,
            description=(
                'This agent orchestrates the decomposition of the user request into'
                ' tasks that can be performed by the child agents.'
            ),
            tools=[
                self.list_remote_agents,
                self.send_message,
            ],
        )

    def root_instruction(self, context: ReadonlyContext) -> str:
        current_agent = self.check_state(context)
        return f"""당신은 사용자의 요청을 적절한 원격 에이전트(remote agent)에게 위임할 수 있는 전문가입니다.

🔍 원격 에이전트 탐색:
- 필요한 경우 `list_remote_agents` 도구를 사용하여 사용 가능한 원격 에이전트 목록을 확인할 수 있습니다.
- 각 에이전트의 설명을 참고하여 어떤 작업을 위임할 수 있을지 판단하세요.

🛠️ 작업 실행:
- 요청을 직접 처리하기 어려운 경우, `send_message` 도구를 사용하여 적절한 원격 에이전트에 작업을 위임하세요.
- 간단한 질문이나 명확한 답이 있는 경우에는 직접 응답해도 됩니다.

💬 사용자 응답 시 유의사항:
- 도구를 사용하여 원격 에이전트에 작업을 위임한 경우, 응답에 해당 에이전트 이름을 반드시 포함하세요.
- 정보를 추측하여 만들어내지 말고, 확실하지 않은 경우에는 사용자에게 추가 정보를 요청하세요.
- 대화의 가장 최근 내용을 중심으로 판단하여 응답하세요.

📋 사용 가능한 에이전트 목록:
{self.agents}

📌 현재 활성 에이전트: {current_agent['active_agent']}
"""

    def check_state(self, context: ReadonlyContext):
        state = context.state
        if (
            'context_id' in state
            and 'session_active' in state
            and state['session_active']
            and 'agent' in state
        ):
            return {'active_agent': f'{state["agent"]}'}
        return {'active_agent': 'None'}

    def before_model_callback(
        self, callback_context: CallbackContext, llm_request
    ):
        state = callback_context.state
        if 'session_active' not in state or not state['session_active']:
            state['session_active'] = True

    def list_remote_agents(self):
        """List the available remote agents you can use to delegate the task."""
        if not self.remote_agent_connections:
            return []

        remote_agent_info = []
        for card in self.cards.values():
            remote_agent_info.append(
                {'name': card.name, 'description': card.description}
            )
        return remote_agent_info

    async def send_message(
        self, agent_name: str, message: str, tool_context: ToolContext
    ):
        """Sends a task either streaming (if supported) or non-streaming.

        This will send a message to the remote agent named agent_name.

        Args:
          agent_name: The name of the agent to send the task to.
          message: The message to send to the agent for the task.
          tool_context: The tool context this method runs in.

        Yields:
          A dictionary of JSON data.
        """
        if agent_name not in self.remote_agent_connections:
            raise ValueError(f'Agent {agent_name} not found')
        state = tool_context.state
        state['agent'] = agent_name
        client = self.remote_agent_connections[agent_name]
        if not client:
            raise ValueError(f'Client not available for {agent_name}')
        taskId = state.get('task_id', None)
        contextId = state.get('context_id', None)
        messageId = state.get('message_id', None)
        task: Task
        if not messageId:
            messageId = str(uuid.uuid4())
        request: MessageSendParams = MessageSendParams(
            id=str(uuid.uuid4()),
            message=Message(
                role='user',
                parts=[TextPart(text=message)],
                messageId=messageId,
                contextId=contextId,
                taskId=taskId,
            ),
            configuration=MessageSendConfiguration(
                acceptedOutputModes=['text', 'text/plain', 'image/png'],
            ),
        )
        response = await client.send_message(request, self.task_callback)
        if isinstance(response, Message):
            return await convert_parts(task.parts, tool_context)
        task: Task = response
        # Assume completion unless a state returns that isn't complete
        state['session_active'] = task.status.state not in [
            TaskState.completed,
            TaskState.canceled,
            TaskState.failed,
            TaskState.unknown,
        ]
        if task.contextId:
            state['context_id'] = task.contextId
        state['task_id'] = task.id
        if task.status.state == TaskState.input_required:
            # Force user input back
            tool_context.actions.skip_summarization = True
            tool_context.actions.escalate = True
        elif task.status.state == TaskState.canceled:
            # Open question, should we return some info for cancellation instead
            raise ValueError(f'Agent {agent_name} task {task.id} is cancelled')
        elif task.status.state == TaskState.failed:
            # Raise error for failure
            raise ValueError(f'Agent {agent_name} task {task.id} failed')
        response = []
        if task.status.message:
            # Assume the information is in the task message.
            response.extend(
                await convert_parts(task.status.message.parts, tool_context)
            )
        if task.artifacts:
            for artifact in task.artifacts:
                response.extend(
                    await convert_parts(artifact.parts, tool_context)
                )
        return response


async def convert_parts(parts: list[Part], tool_context: ToolContext):
    rval = []
    for p in parts:
        rval.append(await convert_part(p, tool_context))
    return rval


async def convert_part(part: Part, tool_context: ToolContext):
    if part.root.kind == 'text':
        return part.root.text
    elif part.root.kind == 'data':
        return part.root.data
    elif part.root.kind == 'file':
        # Repackage A2A FilePart to google.genai Blob
        # Currently not considering plain text as files
        file_id = part.root.file.name
        file_bytes = base64.b64decode(part.root.file.bytes)
        file_part = types.Part(
            inline_data=types.Blob(
                mime_type=part.root.file.mimeType, data=file_bytes
            )
        )
        await tool_context.save_artifact(file_id, file_part)
        tool_context.actions.skip_summarization = True
        tool_context.actions.escalate = True
        return DataPart(data={'artifact-file-id': file_id})
    return f'Unknown type: {part.kind}'
