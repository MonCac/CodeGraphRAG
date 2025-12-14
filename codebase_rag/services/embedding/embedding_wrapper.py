from typing import List, Union


class BaseEmbeddingWrapper:
    """
    Base wrapper providing shared embedding logic.
    Subclasses only need to define `encode()` according to model behavior.
    """

    def __init__(self, hf_model):
        self.model = hf_model
        if not hasattr(hf_model, "_client"):
            raise RuntimeError(f"{hf_model.__class__.__name__} has no _client attribute")
        self._client = hf_model._client

    def embed_documents(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_emb = self.encode(batch_texts, is_query=False)
            if hasattr(batch_emb, "tolist"):
                batch_emb = batch_emb.tolist()
            embeddings.extend(batch_emb)
        return embeddings

    def embed_query(self, text_or_texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        is_single = isinstance(text_or_texts, str)
        texts = [text_or_texts] if is_single else text_or_texts
        emb = self.encode(texts, is_query=True)
        if hasattr(emb, "tolist"):
            emb = emb.tolist()
        return emb[0] if is_single else emb

    def encode(self, texts: List[str], is_query: bool = False):
        """子类实现：定义模型 encode 逻辑"""
        raise NotImplementedError("Subclasses must implement encode()")


class JinaCodeEmbeddingWrapper(BaseEmbeddingWrapper):
    """Wrapper for Jina models — supports task='code' and prompt_name='query'."""

    def encode(self, texts: List[str], is_query: bool = False):
        if is_query:
            return self._client.encode(texts)
        return self._client.encode(texts)


class QwenEmbeddingWrapper(BaseEmbeddingWrapper):
    """Wrapper for Qwen models — supports prompt_name='query', no task param."""

    def encode(self, texts: List[str], is_query: bool = False):
        if is_query:
            return self._client.encode(texts)
        return self._client.encode(texts)
