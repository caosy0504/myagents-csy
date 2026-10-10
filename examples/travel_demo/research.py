"""Optional, opt-in Tavily research for the local travel demo."""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Callable
from urllib.parse import urlparse

from dotenv import dotenv_values
from pydantic import BaseModel, Field, field_validator


PROJECT_ENV = Path(__file__).resolve().parents[2] / ".env"
CACHE_SECONDS = 60 * 60
_cache: dict[tuple[str, tuple[str, ...], str], tuple[float, dict]] = {}
_cache_lock = Lock()


class ResearchUnavailableError(ValueError):
    """Tavily is not configured for this local project."""


class ResearchError(RuntimeError):
    """An opted-in web search failed or returned no usable sources."""


class SearchSource(BaseModel):
    title: str = Field(max_length=160)
    url: str = Field(max_length=1000)
    snippet: str = Field(max_length=500)

    @field_validator("url")
    @classmethod
    def https_source(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("搜索来源必须是 HTTPS 链接")
        return value


class ResearchBundle(BaseModel):
    query: str = Field(max_length=200)
    sources: list[SearchSource] = Field(max_length=5)
    fetched_at: str = Field(max_length=40)
    cached: bool = False


def tavily_api_key() -> str:
    """Read only the Tavily key; the demo never reads cloud LLM settings."""
    return (os.getenv("TAVILY_API_KEY") or dotenv_values(PROJECT_ENV).get("TAVILY_API_KEY") or "").strip()


def tavily_available() -> bool:
    return bool(tavily_api_key())


def search_city(
    city: str,
    preferences: list[str],
    *,
    api_key: str | None = None,
    client_factory: Callable | None = None,
) -> dict:
    """Run at most one basic Tavily search per uncached city/preference query."""
    key = api_key or tavily_api_key()
    if not key:
        raise ResearchUnavailableError(
            "Tavily Key 未配置：请在 MyAgents-csy-main/.env 的 TAVILY_API_KEY= 后填写，重启服务再勾选联网搜索。"
        )
    query = f"{city} 旅游景点 {' '.join(preferences[:3])} 旅行攻略 官方景区".strip()
    cache_key = (city, tuple(preferences[:3]), key)
    now = time.monotonic()
    with _cache_lock:
        cached = _cache.get(cache_key)
        if cached and now - cached[0] < CACHE_SECONDS:
            return {**cached[1], "cached": True}

    if client_factory is None:
        try:
            from tavily import TavilyClient
        except ImportError as exc:
            raise ResearchUnavailableError(
                "Tavily 搜索依赖未安装：请在项目虚拟环境执行 pip install -e '.[search]'。"
            ) from exc
        client_factory = TavilyClient
    try:
        response = client_factory(api_key=key).search(
            query=query,
            search_depth="basic",
            max_results=5,
            include_answer=False,
            language="zh-cn",
        )
    except Exception as exc:
        raise ResearchError("Tavily 搜索失败，请检查 Key、网络和账号用量。") from exc

    sources = []
    for item in response.get("results", []) if isinstance(response, dict) else []:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "")
        if urlparse(url).scheme != "https":
            continue
        title = str(item.get("title") or "").strip()[:160]
        snippet = str(item.get("content") or "").strip()[:500]
        if title and snippet:
            sources.append({"title": title, "url": url[:1000], "snippet": snippet})
        if len(sources) == 5:
            break
    if not sources:
        raise ResearchError("Tavily 没有返回可用的来源；本次未生成行程，也不会把搜索摘要当作事实。")

    bundle = ResearchBundle(
        query=query,
        sources=sources,
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    ).model_dump(mode="json")
    with _cache_lock:
        _cache[cache_key] = (now, bundle)
    return bundle
