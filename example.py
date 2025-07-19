import asyncio
import uuid
from typing import Dict, Any
from google.genai import types
from pydantic import BaseModel

from google.adk.agents.llm_agent import Agent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.events.event import Event
from google.adk.runners import InMemoryRunner
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from dotenv import load_dotenv

load_dotenv()

import agentops
from agentops.sdk.decorators import tool, trace

agentops.init(tags=["math-agent", "google-adk", "calculator"], trace_name="Math Agent ADK Example")

class CalculatorRequest(BaseModel):
    operation: str
    a: float
    b: float


class CalculatorResponse(BaseModel):
    operation: str
    a: float
    b: float
    result: float


@tool(name="Calculator", cost=0.01)
def calculator(args: Dict[str, Any], tool_context: ToolContext) -> Dict[str, Any]:
    request = CalculatorRequest(**args)
    
    result = 0
    if request.operation == "add":
        result = request.a + request.b
    elif request.operation == "subtract":
        result = request.a - request.b
    elif request.operation == "multiply":
        result = request.a * request.b
    elif request.operation == "divide":
        if request.b == 0:
            return {"error": "Division by zero is not allowed"}
        result = request.a / request.b
    else:
        return {"error": f"Unsupported operation: {request.operation}"}
    
    response = CalculatorResponse(
        operation=request.operation,
        a=request.a,
        b=request.b,
        result=result,
    )
    
    return response.model_dump()


def create_math_agent() -> Agent:
    calculator_tool = FunctionTool(func=calculator)
    
    agent = Agent(
        name="math_agent",
        description="An agent that can perform mathematical calculations",
        model="gemini-1.5-flash",
        instruction="""
        You are a helpful assistant that can perform mathematical calculations.
        
        When asked to perform a calculation, you MUST use the calculator tool to provide information.
        DO NOT generate code snippets or print statements. Instead, directly use the tool.
        
        For example, if someone asks "What is 5 + 3?", you should use the calculator tool
        with operation="add", a=5, b=3 and then present the results in a friendly way.
        
        If someone asks about multiple calculations, use the tool for each calculation.
        
        The calculator tool supports the following operations:
        - add: Adds two numbers
        - subtract: Subtracts the second number from the first
        - multiply: Multiplies two numbers
        - divide: Divides the first number by the second (division by zero is not allowed)
        """,
        tools=[calculator_tool],
    )
    
    return agent


@trace(name="AgentExecution", tags=["adk-execution"])
async def run_agent(
    agent: Agent,
    user_id: str,
    session_id: str,
    message: str,
) -> list[Event]:
    runner = InMemoryRunner(
        agent=agent,
        app_name="math-example",
    )
    
    session = runner.session_service.create_session(
        app_name="math-example",
        user_id=user_id,
        session_id=session_id,
    )
    
    content = types.Content(
        role="user",
        parts=[types.Part(text=message)],
    )
    
    run_config = RunConfig(
        streaming_mode=StreamingMode.NONE,
    )
    
    events = []
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
        run_config=run_config,
    ):
        events.append(event)
    
    return events


@trace(name="MainWorkflow", tags=["main-workflow", "math-demo"])
async def main() -> None:
    agent = create_math_agent()
    
    user_id = f"user-{uuid.uuid4().hex[:8]}"
    session_id = f"session-{uuid.uuid4().hex[:8]}"
    
    messages = [
        "What is 5 + 3?",
        "Can you calculate 10 * 7?",
        "What is 20 divided by 4?",
        "What is 15 - 8?",
    ]
    
    for i, message in enumerate(messages):
        events = await run_agent(
            agent=agent,
            user_id=user_id,
            session_id=session_id,
            message=message,
        )
        
        for event in events:
            if hasattr(event, 'content') and event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        print(part.text.strip())
        
        if i < len(messages) - 1:
            await asyncio.sleep(2)
    
    print("Example completed.")


if __name__ == "__main__":
    asyncio.run(main())