"""Offline travel workflow checks; the model and external services are never called."""

import pytest

from examples.travel_demo.data import CITIES
from examples.travel_demo.planner import TravelRequest, _json_object, build_plan_from_selection, create_plan
from examples.travel_demo import research


class FakeLLM:
    provider = "fake"

    def __init__(self, *replies):
        self.replies = iter(replies)
        self.calls = 0

    def invoke(self, messages, **kwargs):
        self.calls += 1
        return next(self.replies)


def test_plan_uses_only_fixture_attractions_and_calculates_budget():
    request = TravelRequest(city="北京", start_date="2026-10-15", days=2)
    llm = FakeLLM('''{"days":[
      {"attractions":["bj_palace","bj_jingshan"],"theme":"古城"},
      {"attractions":["bj_temple","bj_museum"],"theme":"文化"}
    ],"tips":"按自己的节奏游览。"}''')

    plan = create_plan(request, llm)

    assert [item["name"] for item in plan["days"][0]["attractions"]] == ["故宫", "景山公园"]
    assert plan["budget"] == {
        "tickets": 100, "lodging": 380, "meals": 310, "transport": 80, "total": 870,
    }
    assert plan["days"][0]["timeline"][1]["poi_id"] == "bj_palace"
    assert plan["days"][0]["budget"]["total"] == 645
    assert plan["workflow"]["tools"] == ["mock_attractions", "mock_hotels", "mock_weather"]
    assert llm.calls == 1


def test_unknown_attraction_is_rejected_even_after_retry():
    request = TravelRequest(city="上海", start_date="2026-10-15", days=1)
    invalid = '{"days":[{"attractions":["made_up","sh_bund"],"theme":"虚构"}]}'
    llm = FakeLLM(invalid, invalid)

    with pytest.raises(ValueError, match="不存在的景点"):
        create_plan(request, llm)
    assert llm.calls == 2


def test_unsupported_city_is_rejected():
    with pytest.raises(ValueError, match="目前仅支持"):
        TravelRequest(city="未提供的城市", start_date="2026-10-15", days=1)


def test_edit_rebuilds_schedule_and_flags_budget_without_model_call():
    request = TravelRequest(
        city="北京", start_date="2026-10-15", days=2,
        pace="轻松", accommodation="经济型", budget_limit=500,
    )
    plan = build_plan_from_selection(
        request,
        [
            {"attractions": ["bj_palace", "bj_jingshan"], "theme": "古城"},
            {"attractions": ["bj_temple", "bj_museum"], "theme": "文化"},
        ],
        "模拟建议",
    )
    changed = build_plan_from_selection(
        request,
        [
            {"attractions": ["bj_jingshan"], "theme": "古城"},
            {"attractions": ["bj_temple", "bj_museum"], "theme": "文化"},
        ],
        "模拟建议", allow_empty_days=True, edited=True,
    )

    assert plan["budget_remaining"] < 0
    assert changed["budget"]["tickets"] == plan["budget"]["tickets"] - 60
    assert changed["days"][0]["timeline"][1]["start"] == "09:00"
    assert changed["days"][0]["timeline"][1]["title"] == "景山公园"
    assert changed["edited"] is True


def test_edit_cannot_duplicate_attraction_across_days():
    request = TravelRequest(city="上海", start_date="2026-10-15", days=2)
    with pytest.raises(ValueError, match="重复"):
        build_plan_from_selection(
            request,
            [
                {"attractions": ["sh_bund"], "theme": "海滨"},
                {"attractions": ["sh_bund"], "theme": "夜景"},
            ],
            "模拟建议", allow_empty_days=True,
        )


def test_relaxed_pace_rejects_three_attractions_from_model():
    request = TravelRequest(city="杭州", start_date="2026-10-15", days=1, pace="轻松")
    invalid = '{"days":[{"attractions":["hz_westlake","hz_lingyin","hz_xixi"],"theme":"湖山"}]}'
    with pytest.raises(ValueError, match="2 至 2 个景点"):
        create_plan(request, FakeLLM(invalid, invalid))


def test_every_offline_city_has_enough_unique_attractions_for_three_dense_days():
    assert len(CITIES) == 29
    for city, data in CITIES.items():
        ids = [attraction["id"] for attraction in data["attractions"]]
        assert len(ids) >= 9, city
        assert len(ids) == len(set(ids)), city
        selection = [
            {"attractions": ids[index:index + 3], "theme": f"第{index // 3 + 1}天"}
            for index in (0, 3, 6)
        ]
        plan = build_plan_from_selection(
            TravelRequest(city=city, start_date="2026-10-15", days=3, pace="紧凑"),
            selection,
            "模拟建议",
        )
        assert len(plan["days"]) == 3
        assert sum(len(day["attractions"]) for day in plan["days"]) == 9


def test_tavily_is_opt_in_and_one_basic_search_is_cached():
    calls = []

    class FakeTavily:
        def __init__(self, api_key):
            assert api_key == "test-key"

        def search(self, **kwargs):
            calls.append(kwargs)
            return {"results": [
                {"title": "示例旅游指南", "url": "https://example.org/guide", "content": "示例城市的景点资料"},
                {"title": "无效来源", "url": "http://example.org/unsafe", "content": "不采用"},
            ]}

    first = research.search_city("测试城", ["自然风光"], api_key="test-key", client_factory=FakeTavily)
    second = research.search_city("测试城", ["自然风光"], api_key="test-key", client_factory=FakeTavily)

    assert len(calls) == 1
    assert calls[0]["search_depth"] == "basic"
    assert calls[0]["max_results"] == 5
    assert first["sources"][0]["url"] == "https://example.org/guide"
    assert len(first["sources"]) == 1
    assert second["cached"] is True


def test_web_research_keeps_sources_and_uses_existing_city_catalog():
    request = TravelRequest(city="北京", start_date="2026-10-15", days=1, web_research=True)
    source = {
        "query": "北京 旅游景点", "sources": [
            {"title": "北京景点", "url": "https://example.org/beijing", "snippet": "故宫和景山公园"},
        ], "fetched_at": "2026-10-10T00:00:00+00:00", "cached": False,
    }
    llm = FakeLLM('{"days":[{"attractions":["bj_palace","bj_jingshan"],"theme":"古城"}],"tips":"模拟建议"}')
    plan = create_plan(request, llm, researcher=lambda city, prefs: source)
    selection = [{"attractions": [item["id"] for item in plan["days"][0]["attractions"]], "theme": "古城"}]
    rebuilt = build_plan_from_selection(
        request, selection, plan["tips"], research=plan["research"], edited=True,
    )

    assert plan["research"]["sources"][0]["url"] == "https://example.org/beijing"
    assert "tavily_search" in plan["workflow"]["tools"]
    assert rebuilt["research"] == plan["research"]
    assert llm.calls == 1


def test_model_json_parser_accepts_one_object_with_trailing_explanation():
    response = '```json\n{"days": [], "tips": "ok"}\n```\n补充说明'
    assert _json_object(response)["tips"] == "ok"
