"""AST-based calculator without eval()."""

import ast
import math
import operator
from typing import Callable

from .base import Tool


class CalculatorTool(Tool):
    MAX_EXPRESSION_LENGTH = 200
    MAX_EXPONENT = 100

    OPERATORS: dict[type[ast.operator | ast.unaryop], Callable] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    FUNCTIONS = {
        "abs": abs,
        "round": round,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
    }
    CONSTANTS = {"pi": math.pi, "e": math.e}

    def __init__(self) -> None:
        super().__init__(
            name="calculator",
            description="计算数学表达式，例如 (25 + 15) * 3 - 8 或 sqrt(16)",
        )

    def run(self, input_text: str) -> str:
        expression = input_text.strip()
        if not expression:
            return "错误：计算表达式不能为空。"
        if len(expression) > self.MAX_EXPRESSION_LENGTH:
            raise ValueError("表达式过长")
        node = ast.parse(expression, mode="eval")
        return str(self._evaluate(node.body))

    def _evaluate(self, node: ast.AST):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("只允许数字常量")
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            base = self._evaluate(node.left)
            exponent = self._evaluate(node.right)
            if abs(exponent) > self.MAX_EXPONENT:
                raise ValueError(f"指数绝对值不能超过 {self.MAX_EXPONENT}")
            return operator.pow(base, exponent)
        if isinstance(node, ast.BinOp) and type(node.op) in self.OPERATORS:
            return self.OPERATORS[type(node.op)](
                self._evaluate(node.left), self._evaluate(node.right)
            )
        if isinstance(node, ast.UnaryOp) and type(node.op) in self.OPERATORS:
            return self.OPERATORS[type(node.op)](self._evaluate(node.operand))
        if isinstance(node, ast.Name) and node.id in self.CONSTANTS:
            return self.CONSTANTS[node.id]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            function = self.FUNCTIONS.get(node.func.id)
            if function is None:
                raise ValueError(f"不支持函数 {node.func.id}")
            return function(*(self._evaluate(argument) for argument in node.args))
        raise ValueError(f"不支持的表达式：{ast.dump(node)}")
