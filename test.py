# test_ollama.py
import asyncio
from typing import Any
from rich.console import Console

from pydantic import AnyHttpUrl
from pydantic_ai.models.openai import (
    OpenAIModel,
    OpenAIResponsesModel,
)
from pydantic_ai.providers.openai import OpenAIProvider

# ---------------- 配置 ----------------
LOCAL_MODEL_ENDPOINT: AnyHttpUrl = AnyHttpUrl("http://localhost:11434/v1")
LOCAL_MODEL_API_KEY = "ollama"
LOCAL_ORCHESTRATOR_MODEL_ID = "gemma3:1b"
LOCAL_CYPHER_MODEL_ID = "gemma3:1b"

# 系统提示
LOCAL_CYPHER_SYSTEM_PROMPT = "你是一个Cypher数据库查询助手，请根据用户问题生成相应的Cypher查询。"

# ---------------- 主逻辑 ----------------
async def main():
    # 初始化模型
    llm = OpenAIModel(  # type: ignore
        LOCAL_CYPHER_MODEL_ID,
        provider=OpenAIProvider(
            api_key=LOCAL_MODEL_API_KEY,
            base_url=str(LOCAL_MODEL_ENDPOINT),
        ),
    )

    # 准备问题和消息历史（可修改为测试用例）
    question= "请帮我生成查询所有电影节点的Cypher语句。"
    message_history = []

    # 直接与模型交互
    response = await llm.acall(
        messages=[
            {"role": "system", "content": LOCAL_CYPHER_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
    )

    print("模型输出:\n", response)


async def run_with_cancellation(
    console: Console, coro: Any, timeout: float | None = None
) -> Any:
    """Run a coroutine with proper Ctrl+C cancellation that doesn't exit the program."""
    task = asyncio.create_task(coro)

    try:
        return await asyncio.wait_for(task, timeout=timeout) if timeout else await task
    except TimeoutError:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        console.print(
            f"\n[bold yellow]Operation timed out after {timeout} seconds.[/bold yellow]"
        )
        return {"cancelled": True, "timeout": True}
    except (asyncio.CancelledError, KeyboardInterrupt):
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        console.print("\n[bold yellow]Thinking cancelled.[/bold yellow]")
        return {"cancelled": True}


# ---------------- 运行 ----------------
if __name__ == "__main__":
    asyncio.run(main())
