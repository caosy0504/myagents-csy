from my_agents import CalculatorTool, ToolRegistry


def test_calculator_and_registry():
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    assert registry.execute("calculator", "(25 + 15) * 3 - 8") == "112"
    assert registry.execute("calculator", "sqrt(16) + 2 * 3") == "10.0"


def test_unknown_tool_returns_error():
    registry = ToolRegistry()
    assert "未找到工具" in registry.execute("missing", "input")


def test_calculator_rejects_code_and_extreme_exponents():
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    assert "执行失败" in registry.execute("calculator", "__import__('os')")
    assert "执行失败" in registry.execute("calculator", "2 ** 1000")
