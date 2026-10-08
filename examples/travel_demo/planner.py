"""A small travel workflow using MyAgents' ToolRegistry and SimpleAgent."""

from __future__ import annotations

import json
import re
from datetime import date, timedelta
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from my_agents import MyAgentsLLM, SimpleAgent, ToolRegistry

from .data import CITIES


class PlanGenerationError(ValueError):
    """The local model did not produce a usable, fixture-grounded selection."""


class TravelRequest(BaseModel):
    city: str
    start_date: date
    days: int = Field(ge=1, le=3)
    preferences: list[str] = Field(default_factory=list, max_length=5)
    transportation: Literal["公共交通", "打车"] = "公共交通"
    accommodation: Literal["舒适型", "经济型"] = "舒适型"
    pace: Literal["轻松", "标准", "紧凑"] = "标准"
    budget_limit: int | None = Field(default=None, ge=100, le=100_000)
    notes: str = Field(default="", max_length=300)

    @field_validator("city")
    @classmethod
    def supported_city(cls, value: str) -> str:
        if value not in CITIES:
            raise ValueError(f"目前仅支持：{'、'.join(CITIES)}")
        return value


def _mock_tools() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_function(
        "mock_attractions", "读取指定城市的本地模拟景点",
        lambda city: json.dumps(CITIES[city]["attractions"], ensure_ascii=False),
    )
    registry.register_function(
        "mock_hotels", "读取指定城市的本地模拟酒店",
        lambda city: json.dumps(CITIES[city]["hotels"], ensure_ascii=False),
    )
    registry.register_function(
        "mock_weather", "读取指定城市的本地模拟天气基准",
        lambda city: json.dumps({"temperature": CITIES[city]["temperature"]}, ensure_ascii=False),
    )
    return registry


def _json_object(text: str) -> dict[str, Any]:
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        value = json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", clean)
        if not match:
            raise ValueError("模型没有返回 JSON 对象") from None
        value = json.loads(match.group())
    if not isinstance(value, dict):
        raise ValueError("模型返回的顶层结构不是对象")
    return value


def _validate_selection(
    value: dict[str, Any], count: int, valid_ids: set[str], pace: str
) -> list[dict[str, Any]]:
    days = value.get("days")
    if not isinstance(days, list) or len(days) != count:
        raise ValueError(f"需要恰好 {count} 天的行程")
    minimum, maximum = {"轻松": (2, 2), "标准": (2, 3), "紧凑": (3, 3)}[pace]
    seen: set[str] = set()
    for index, item in enumerate(days, 1):
        if not isinstance(item, dict):
            raise ValueError(f"第 {index} 天不是对象")
        ids = item.get("attractions")
        if not isinstance(ids, list) or not minimum <= len(ids) <= maximum:
            raise ValueError(f"第 {index} 天需要 {minimum} 至 {maximum} 个景点 ID")
        if any(not isinstance(poi_id, str) or poi_id not in valid_ids for poi_id in ids):
            raise ValueError(f"第 {index} 天包含数据集中不存在的景点 ID")
        if seen.intersection(ids) or len(set(ids)) != len(ids):
            raise ValueError("景点不能重复安排")
        seen.update(ids)
        if not isinstance(item.get("theme"), str) or not item["theme"].strip():
            raise ValueError(f"第 {index} 天缺少主题")
    return days


def _time_text(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _timeline(attractions: list[dict], meals: list[dict], pace: str) -> list[dict]:
    """Create illustrative time slots without claiming actual route durations."""
    events = [{
        "kind": "meal", "start": "07:00", "end": "07:40",
        "title": meals[0]["name"], "note": "模拟早餐",
    }]
    current = {"轻松": 9 * 60, "标准": 8 * 60 + 30, "紧凑": 8 * 60}[pace]
    for index, poi in enumerate(attractions):
        finish = current + poi["minutes"]
        events.append({
            "kind": "attraction", "poi_id": poi["id"],
            "start": _time_text(current), "end": _time_text(finish),
            "title": poi["name"], "note": f"建议游览 {poi['minutes']} 分钟（模拟）",
        })
        current = finish
        if index == 0:
            lunch_start = max(12 * 60, current + 20)
            events.append({
                "kind": "meal", "start": _time_text(lunch_start),
                "end": _time_text(lunch_start + 60), "title": meals[1]["name"],
                "note": "模拟午餐",
            })
            current = lunch_start + 80
        elif index < len(attractions) - 1:
            current += 30
    if not attractions:
        events.append({
            "kind": "meal", "start": "12:00", "end": "13:00",
            "title": meals[1]["name"], "note": "模拟午餐",
        })
        current = 13 * 60
    dinner_start = max(18 * 60, current + 45)
    events.append({
        "kind": "meal", "start": _time_text(dinner_start),
        "end": _time_text(dinner_start + 70), "title": meals[2]["name"],
        "note": "模拟晚餐",
    })
    return events


def build_plan_from_selection(
    request: TravelRequest,
    selection: list[dict[str, Any]],
    tips: str,
    *,
    model_attempts: int = 0,
    allow_empty_days: bool = False,
    edited: bool = False,
) -> dict[str, Any]:
    """Rebuild a plan after generation or editing, without another model call."""
    tools = _mock_tools()
    attractions = json.loads(tools.execute("mock_attractions", request.city))
    hotels = json.loads(tools.execute("mock_hotels", request.city))
    weather_base = json.loads(tools.execute("mock_weather", request.city))
    by_id = {item["id"]: item for item in attractions}
    if len(selection) != request.days:
        raise ValueError("行程天数与请求不一致")
    seen: set[str] = set()
    for index, day in enumerate(selection, 1):
        if not isinstance(day, dict):
            raise ValueError(f"第 {index} 天不是对象")
        ids = day.get("attractions")
        minimum = 0 if allow_empty_days else 2
        if not isinstance(ids, list) or not minimum <= len(ids) <= 3:
            raise ValueError(f"第 {index} 天需要 {minimum} 至 3 个景点")
        if any(not isinstance(poi_id, str) or poi_id not in by_id for poi_id in ids):
            raise ValueError(f"第 {index} 天包含数据集中不存在的景点 ID")
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError("景点不能重复安排")
        seen.update(ids)
        if not isinstance(day.get("theme"), str) or not day["theme"].strip():
            raise ValueError(f"第 {index} 天缺少主题")

    hotel = next(item for item in hotels if item["type"] == request.accommodation)
    meal_names = CITIES[request.city]["meals"]
    transport_per_day = 40 if request.transportation == "公共交通" else 100
    result_days = []
    weather = []
    for index, item in enumerate(selection):
        current_date = request.start_date + timedelta(days=index)
        day_attractions = [by_id[poi_id] for poi_id in item["attractions"]]
        day_meals = [
            {"type": kind, "name": name, "estimated_cost": cost}
            for kind, name, cost in zip(
                ("breakfast", "lunch", "dinner"), meal_names, (25, 55, 75), strict=True
            )
        ]
        day_budget = {
            "tickets": sum(poi["ticket"] for poi in day_attractions),
            "lodging": hotel["nightly"] if index < request.days - 1 else 0,
            "meals": sum(meal["estimated_cost"] for meal in day_meals),
            "transport": transport_per_day,
        }
        day_budget["total"] = sum(day_budget.values())
        result_days.append({
            "date": current_date.isoformat(), "day_index": index,
            "theme": item["theme"].strip()[:60],
            "transportation": request.transportation, "hotel": hotel,
            "attractions": day_attractions, "meals": day_meals,
            "timeline": _timeline(day_attractions, day_meals, request.pace),
            "budget": day_budget,
        })
        weather.append({
            "date": current_date.isoformat(),
            "label": ("晴", "多云", "阴")[index % 3],
            "temperature": weather_base["temperature"] + (index % 3) - 1,
        })

    budget = {
        key: sum(day["budget"][key] for day in result_days)
        for key in ("tickets", "lodging", "meals", "transport", "total")
    }
    return {
        "schema_version": 2,
        "edited": edited,
        "request": request.model_dump(mode="json"),
        "city": request.city,
        "start_date": request.start_date.isoformat(),
        "end_date": (request.start_date + timedelta(days=request.days - 1)).isoformat(),
        "days": result_days, "weather": weather, "catalog": attractions,
        "tips": tips.strip()[:300] or "行程、费用和天气均为演示数据。",
        "budget": budget,
        "budget_limit": request.budget_limit,
        "budget_remaining": (
            request.budget_limit - budget["total"] if request.budget_limit is not None else None
        ),
        "workflow": {
            "tools": tools.list_tools(),
            "model_attempts": model_attempts,
            "stages": ["读取本地模拟数据", "Ollama 挑选景点与主题", "校验景点 ID", "确定性排时间与计算预算"],
        },
        "data_notice": "景点资料、天气、价格、位置和时间均为本地演示数据，不可用于真实出行决策。",
        "model": "Ollama · qwen2.5:7b",
    }


def create_plan(request: TravelRequest, llm: Any | None = None) -> dict[str, Any]:
    """Build a verified plan; only Ollama generates the attraction selection and prose."""
    tools = _mock_tools()
    attractions = json.loads(tools.execute("mock_attractions", request.city))
    by_id = {item["id"]: item for item in attractions}

    agent = SimpleAgent(
        name="本地旅行规划助手",
        llm=llm or MyAgentsLLM(
            model="qwen2.5:7b", api_key="ollama", base_url="http://127.0.0.1:11434/v1",
            timeout=180,
        ),
        system_prompt=(
            "你是旅行行程规划助手。只能使用用户消息提供的模拟景点 ID。"
            "不得编造景点、票价、天气预报、营业时间或交通时长。只输出合法 JSON。"
        ),
    )
    target_count = {"轻松": "恰好2", "标准": "2至3", "紧凑": "恰好3"}[request.pace]
    prompt = (
        f"请为{request.city}安排{request.days}天的演示行程，"
        f"{request.pace}节奏每一天选{target_count}个景点，"
        "全程不能重复。优先考虑用户偏好与示意位置的接近程度。\n"
        f"偏好：{request.preferences or ['经典景点']}；交通：{request.transportation}；"
        f"住宿：{request.accommodation}；节奏：{request.pace}；"
        f"预算上限：{request.budget_limit if request.budget_limit is not None else '未设置'}元；"
        f"补充要求：{request.notes or '无'}。\n"
        "可用景点（x/y只是示意图坐标，票价仅供演示）：\n"
        + json.dumps(attractions, ensure_ascii=False)
        + '\n只输出 JSON 对象，包含 days 数组（每项有 attractions 景点 ID 数组和 theme 简短主题）'
        '与 tips 字符串。tips 请结合所选景点写一至两句建议，不要照抄字段说明。'
        'days 数量必须与天数一致。'
    )
    selection = None
    last_error = ""
    attempts = 0
    for attempt in range(2):
        attempts += 1
        reply = agent.run(prompt if attempt == 0 else f"上次输出无效：{last_error}。请重新按原要求输出完整 JSON。", max_tokens=1000)
        try:
            parsed = _json_object(reply)
            selection = _validate_selection(parsed, request.days, set(by_id), request.pace)
            tips = parsed.get("tips") if isinstance(parsed.get("tips"), str) else "行程与价格均为模拟数据。"
            break
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = str(exc)
    if selection is None:
        raise PlanGenerationError(f"本地模型未能生成有效行程：{last_error}")

    return build_plan_from_selection(request, selection, tips, model_attempts=attempts)
