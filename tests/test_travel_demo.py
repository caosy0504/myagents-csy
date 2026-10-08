"""Offline travel workflow checks; the model and external services are never called."""

import pytest

from examples.travel_demo.planner import TravelRequest, build_plan_from_selection, create_plan


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
