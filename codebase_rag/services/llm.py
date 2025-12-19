from typing import cast, Any

from loguru import logger
from pydantic_ai import Agent, Tool
from pydantic_ai.models.gemini import GeminiModel, GeminiModelSettings
from pydantic_ai.models.openai import (
    OpenAIModel,
    OpenAIChatModel,
    OpenAIResponsesModel,
)
from pydantic_ai.providers.google_gla import GoogleGLAProvider
from pydantic_ai.providers.google_vertex import GoogleVertexProvider, VertexAiRegion
from pydantic_ai.providers.openai import OpenAIProvider
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings

from codebase_rag.config import detect_provider_from_model, settings
from codebase_rag.prompts import (
    CYPHER_SYSTEM_PROMPT,
    LOCAL_CYPHER_SYSTEM_PROMPT,
    RAG_ORCHESTRATOR_SYSTEM_PROMPT,
    SEMANTIC_EXTRACTION_SYSTEM_PROMPT, GRAPH_EXTRACTION_SYSTEM_PROMPT, GRAPH_EXTRACTION_RAG_ORCHESTRATOR_SYSTEM_PROMPT,
    ANTIPATTERN_RELEVANCE_SYSTEM_PROMPT,
    CODE2TEXT_SYSTEM_PROMPT
)


class LLMGenerationError(Exception):
    """Custom exception for LLM generation failures."""

    pass


def _clean_cypher_response(response_text: str) -> str:
    """Utility to clean up common LLM formatting artifacts from a Cypher query."""
    query = response_text.strip().replace("`", "")
    if query.startswith("cypher"):
        query = query[6:].strip()
    if not query.endswith(";"):
        query += ";"
    return query


class CypherGenerator:
    """Generates Cypher queries from natural language."""

    def __init__(self) -> None:
        try:
            model_settings = None

            # Get active cypher model and detect its provider
            cypher_model_id = settings.active_cypher_model
            cypher_provider = detect_provider_from_model(cypher_model_id)

            # Configure model based on detected provider
            if cypher_provider == "gemini":
                if settings.GEMINI_PROVIDER == "vertex":
                    provider = GoogleVertexProvider(
                        project_id=settings.GCP_PROJECT_ID,
                        region=cast(VertexAiRegion, settings.GCP_REGION),
                        service_account_file=settings.GCP_SERVICE_ACCOUNT_FILE,
                    )
                else:
                    provider = GoogleGLAProvider(api_key=settings.GEMINI_API_KEY)  # type: ignore

                if settings.GEMINI_THINKING_BUDGET is not None:
                    model_settings = GeminiModelSettings(
                        gemini_thinking_config={
                            "thinking_budget": int(settings.GEMINI_THINKING_BUDGET)
                        }
                    )

                llm = GeminiModel(
                    cypher_model_id,
                    provider=provider,
                )
                system_prompt = CYPHER_SYSTEM_PROMPT
            elif cypher_provider == "openai":
                llm = OpenAIResponsesModel(
                    cypher_model_id,
                    provider=OpenAIProvider(
                        api_key=settings.OPENAI_API_KEY,
                    ),
                )
                system_prompt = GRAPH_EXTRACTION_SYSTEM_PROMPT
            else:  # local
                llm = OpenAIModel(  # type: ignore
                    cypher_model_id,
                    provider=OpenAIProvider(
                        api_key=settings.OPENAI_API_KEY,
                        base_url=str(settings.OPENAI_MODEL_ENDPOINT),
                    ),
                )
                system_prompt = LOCAL_CYPHER_SYSTEM_PROMPT
            self.agent = Agent(
                model=llm,
                system_prompt=system_prompt,
                output_type=str,
                model_settings=model_settings,
            )
        except Exception as e:
            raise LLMGenerationError(
                f"Failed to initialize CypherGenerator: {e}"
            ) from e

    async def generate(self, natural_language_query: str) -> str:
        logger.info(
            f"  [CypherGenerator] Generating query for: '{natural_language_query}'"
        )
        try:
            result = await self.agent.run(natural_language_query)
            if (
                    not isinstance(result.output, str)
                    or "MATCH" not in result.output.upper()
            ):
                raise LLMGenerationError(
                    f"LLM did not generate a valid query. Output: {result.output}"
                )

            query = _clean_cypher_response(result.output)
            logger.info(f"  [CypherGenerator] Generated Cypher: {query}")
            return query
        except Exception as e:
            logger.error(f"  [CypherGenerator] Error: {e}")
            raise LLMGenerationError(f"Cypher generation failed: {e}") from e


def create_rag_orchestrator(tools: list[Tool]) -> Agent:
    """Factory function to create the main RAG orchestrator agent."""
    try:
        model_settings = None

        # Get active orchestrator model and detect its provider
        orchestrator_model_id = settings.active_orchestrator_model
        orchestrator_provider = detect_provider_from_model(orchestrator_model_id)

        if orchestrator_provider == "gemini":
            if settings.GEMINI_PROVIDER == "vertex":
                provider = GoogleVertexProvider(
                    project_id=settings.GCP_PROJECT_ID,
                    region=cast(VertexAiRegion, settings.GCP_REGION),
                    service_account_file=settings.GCP_SERVICE_ACCOUNT_FILE,
                )
            else:
                provider = GoogleGLAProvider(api_key=settings.GEMINI_API_KEY)  # type: ignore

            if settings.GEMINI_THINKING_BUDGET is not None:
                model_settings = GeminiModelSettings(
                    gemini_thinking_config={
                        "thinking_budget": int(settings.GEMINI_THINKING_BUDGET)
                    }
                )

            llm = GeminiModel(
                orchestrator_model_id,
                provider=provider,
            )
        elif orchestrator_provider == "local":
            llm = OpenAIModel(  # type: ignore
                orchestrator_model_id,
                provider=OpenAIProvider(
                    api_key=settings.LOCAL_MODEL_API_KEY,
                    base_url=str(settings.LOCAL_MODEL_ENDPOINT),
                ),
            )
        else:  # openai provider
            llm = OpenAIResponsesModel(
                orchestrator_model_id,
                provider=OpenAIProvider(
                    api_key=settings.OPENAI_API_KEY,
                    base_url=str(settings.OPENAI_MODEL_ENDPOINT),
                ),
            )

        return Agent(
            model=llm,
            system_prompt=GRAPH_EXTRACTION_RAG_ORCHESTRATOR_SYSTEM_PROMPT,
            tools=tools,
            model_settings=model_settings,
        )  # type: ignore
    except Exception as e:
        raise LLMGenerationError(f"Failed to initialize RAG Orchestrator: {e}") from e


def create_semantic_model() -> Any:
    try:
        """
        返回用于生成语义摘要的 LLM client，支持 Gemini / OpenAI / local。
        """
        model_settings = None
        model_id = settings.active_cypher_model
        provider = detect_provider_from_model(model_id)

        if provider == "gemini":
            if settings.GEMINI_PROVIDER == "vertex":
                prov = GoogleVertexProvider(
                    project_id=settings.GCP_PROJECT_ID,
                    region=settings.GCP_REGION,
                    service_account_file=settings.GCP_SERVICE_ACCOUNT_FILE,
                )
            else:
                prov = GoogleGLAProvider(api_key=settings.GEMINI_API_KEY)
            llm = GeminiModel(model_id, provider=prov)

        elif provider == "local":
            llm = OpenAIChatModel(
                model_id,
                provider=OpenAIProvider(
                    api_key=settings.LOCAL_MODEL_API_KEY,
                    base_url=str(settings.LOCAL_MODEL_ENDPOINT),
                ),
            )
        else:  # openai
            llm = OpenAIResponsesModel(
                model_id,
                provider=OpenAIProvider(
                    api_key=settings.OPENAI_API_KEY,
                    base_url=str(settings.OPENAI_MODEL_ENDPOINT),
                ),
            )
        return Agent(
            model=llm,
            system_prompt=SEMANTIC_EXTRACTION_SYSTEM_PROMPT,
            model_settings=model_settings,
        )  # type: ignore
    except Exception as e:
        raise LLMGenerationError(f"Failed to initialize RAG Orchestrator: {e}") from e


def create_relevance_model() -> Any:
    try:
        """
        返回用于生成语义摘要的 LLM client，支持 Gemini / OpenAI / local。
        """
        model_settings = None
        model_id = settings.active_cypher_model
        provider = detect_provider_from_model(model_id)

        if provider == "gemini":
            if settings.GEMINI_PROVIDER == "vertex":
                prov = GoogleVertexProvider(
                    project_id=settings.GCP_PROJECT_ID,
                    region=settings.GCP_REGION,
                    service_account_file=settings.GCP_SERVICE_ACCOUNT_FILE,
                )
            else:
                prov = GoogleGLAProvider(api_key=settings.GEMINI_API_KEY)
            llm = GeminiModel(model_id, provider=prov)

        elif provider == "local":
            llm = OpenAIChatModel(
                model_id,
                provider=OpenAIProvider(
                    api_key=settings.LOCAL_MODEL_API_KEY,
                    base_url=str(settings.LOCAL_MODEL_ENDPOINT),
                ),
            )
        else:  # openai
            llm = OpenAIResponsesModel(
                model_id,
                provider=OpenAIProvider(
                    api_key=settings.OPENAI_API_KEY,
                    base_url=str(settings.OPENAI_MODEL_ENDPOINT),
                ),
            )
        return Agent(
            model=llm,
            system_prompt=ANTIPATTERN_RELEVANCE_SYSTEM_PROMPT,
            model_settings=model_settings,
        )  # type: ignore
    except Exception as e:
        raise LLMGenerationError(f"Failed to initialize RAG Orchestrator: {e}") from e


def create_graph_extract_query_model() -> Any:
    try:
        """
        返回用于生成语义摘要的 LLM client，支持 Gemini / OpenAI / local。
        """
        model_settings = None
        model_id = settings.active_cypher_model
        provider = detect_provider_from_model(model_id)

        if provider == "gemini":
            if settings.GEMINI_PROVIDER == "vertex":
                prov = GoogleVertexProvider(
                    project_id=settings.GCP_PROJECT_ID,
                    region=settings.GCP_REGION,
                    service_account_file=settings.GCP_SERVICE_ACCOUNT_FILE,
                )
            else:
                prov = GoogleGLAProvider(api_key=settings.GEMINI_API_KEY)
            llm = GeminiModel(model_id, provider=prov)

        elif provider == "local":
            llm = OpenAIChatModel(
                model_id,
                provider=OpenAIProvider(
                    api_key=settings.LOCAL_MODEL_API_KEY,
                    base_url=str(settings.LOCAL_MODEL_ENDPOINT),
                ),
            )
        else:  # openai
            llm = OpenAIResponsesModel(
                model_id,
                provider=OpenAIProvider(
                    api_key=settings.OPENAI_API_KEY,
                    base_url=str(settings.OPENAI_MODEL_ENDPOINT),
                ),
            )
        return Agent(
            model=llm,
            system_prompt=CYPHER_SYSTEM_PROMPT,
            model_settings=model_settings,
        )  # type: ignore
    except Exception as e:
        raise LLMGenerationError(f"Failed to initialize RAG Orchestrator: {e}") from e


def create_repair_code_model(system_prompt: str) -> Any:
    try:
        """
        返回自定义 system_prompt 代码修复的 LLM client，支持 Gemini / OpenAI / local。
        """
        model_settings = None
        model_id = settings.active_cypher_model
        provider = detect_provider_from_model(model_id)

        if provider == "gemini":
            if settings.GEMINI_PROVIDER == "vertex":
                prov = GoogleVertexProvider(
                    project_id=settings.GCP_PROJECT_ID,
                    region=settings.GCP_REGION,
                    service_account_file=settings.GCP_SERVICE_ACCOUNT_FILE,
                )
            else:
                prov = GoogleGLAProvider(api_key=settings.GEMINI_API_KEY)
            llm = GeminiModel(model_id, provider=prov)

        elif provider == "local":
            llm = OpenAIChatModel(
                model_id,
                provider=OpenAIProvider(
                    api_key=settings.OPENAI_API_KEY,
                    base_url=str(settings.OPENAI_MODEL_ENDPOINT),
                ),
            )
        else:  # openai
            llm = OpenAIResponsesModel(
                model_id,
                provider=OpenAIProvider(
                    api_key=settings.OPENAI_API_KEY,
                    base_url=str(settings.OPENAI_MODEL_ENDPOINT),
                ),
            )
        return Agent(
            model=llm,
            system_prompt=system_prompt,
            model_settings=model_settings,
        )  # type: ignore
    except Exception as e:
        raise LLMGenerationError(f"Failed to initialize RAG Orchestrator: {e}") from e


def create_embedding_model() -> Embeddings:
    """
    Factory function to create the embedding model based on configured provider.
    Supports Gemini, OpenAI, and local models.
    """
    provider_type = detect_provider_from_model(settings.active_embedding_model)

    if provider_type == "gemini":
        # 如果未来有 Gemini 的 embedding 支持，可以在这里添加
        raise NotImplementedError("Gemini embeddings not yet implemented.")
    elif provider_type == "local":
        # 例如使用 Ollama 本地服务，映射到 OpenAIEmbeddings 接口
        llm_api_key = settings.LOCAL_MODEL_API_KEY
        base_url = str(settings.LOCAL_MODEL_ENDPOINT)
        return OpenAIEmbeddings(
            model=settings.active_embedding_model,
            openai_api_key=llm_api_key,
            openai_api_base=base_url
        )
    else:  # 默认使用 OpenAI
        return OpenAIEmbeddings(
            model=settings.active_embedding_model,
            openai_api_key=settings.OPENAI_API_KEY
        )


def create_code2text_model() -> Any:
    try:
        """
        返回用于生成语义摘要的 LLM client，支持 Gemini / OpenAI / local。
        """
        model_settings = None
        model_id = settings.active_cypher_model
        provider = detect_provider_from_model(model_id)

        if provider == "gemini":
            if settings.GEMINI_PROVIDER == "vertex":
                prov = GoogleVertexProvider(
                    project_id=settings.GCP_PROJECT_ID,
                    region=settings.GCP_REGION,
                    service_account_file=settings.GCP_SERVICE_ACCOUNT_FILE,
                )
            else:
                prov = GoogleGLAProvider(api_key=settings.GEMINI_API_KEY)
            llm = GeminiModel(model_id, provider=prov)

        elif provider == "local":
            llm = OpenAIChatModel(
                model_id,
                provider=OpenAIProvider(
                    api_key=settings.LOCAL_MODEL_API_KEY,
                    base_url=str(settings.LOCAL_MODEL_ENDPOINT),
                ),
            )
        else:  # openai
            llm = OpenAIResponsesModel(
                model_id,
                provider=OpenAIProvider(
                    api_key=settings.OPENAI_API_KEY,
                    base_url=str(settings.OPENAI_MODEL_ENDPOINT),
                ),
            )
        return Agent(
            model=llm,
            system_prompt=CODE2TEXT_SYSTEM_PROMPT,
            model_settings=model_settings,
        )  # type: ignore
    except Exception as e:
        raise LLMGenerationError(f"Failed to initialize RAG Orchestrator: {e}") from e


if __name__ == "__main__":
    model = create_repair_code_model("hello")
    result = model.run_sync("hello")
    print(result)
