"""Bounded, read-only code exploration with precise paths and line ranges."""

import fnmatch
import json
import os
from pathlib import Path

from .base import Tool


class TerminalTool(Tool):
    """List, glob, read line ranges, and search without executing shell commands."""

    _SKIP_DIRS = {"__pycache__", "node_modules", "build", "dist", ".git", ".venv"}

    def __init__(self, workspace: str | Path, max_output_chars: int = 12000) -> None:
        super().__init__(
            "terminal",
            '只读文件探索，JSON 参数。action=list/glob/read/search；read 支持 start_line/end_line；'
            'search/glob 支持 limit/offset。示例 {"action":"read","path":"README.md","start_line":1,"end_line":30}',
        )
        self.workspace = Path(workspace).expanduser().resolve()
        if not self.workspace.is_dir():
            raise ValueError("工作目录不存在")
        if max_output_chars < 256:
            raise ValueError("max_output_chars 至少为 256")
        self.max_output_chars = max_output_chars
        gitignore = self.workspace / ".gitignore"
        self._ignore_patterns = [
            line.strip() for line in gitignore.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith(("#", "!"))
        ] if gitignore.is_file() else []

    def function_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name, "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["list", "glob", "read", "search"]},
                        "path": {"type": "string"}, "pattern": {"type": "string"},
                        "query": {"type": "string"}, "start_line": {"type": "integer"},
                        "end_line": {"type": "integer"}, "start_column": {"type": "integer"},
                        "limit": {"type": "integer"},
                        "offset": {"type": "integer"},
                    },
                    "required": ["action"], "additionalProperties": False,
                },
            },
        }

    def run_arguments(self, arguments: dict) -> str:
        return self.run(json.dumps(arguments, ensure_ascii=False))

    def _resolve(self, relative: str) -> Path:
        if not isinstance(relative, str):
            raise ValueError("path 必须是字符串")
        target = (self.workspace / relative).resolve()
        if not target.is_relative_to(self.workspace):
            raise ValueError("不允许访问工作目录之外的路径")
        if any(part.startswith(".") and part not in {".", ".."} for part in Path(relative).parts):
            raise ValueError("不允许访问隐藏文件或目录")
        return target

    def _ignored(self, path: Path) -> bool:
        if not path.resolve().is_relative_to(self.workspace):
            return True
        relative = path.relative_to(self.workspace)
        if any(part.startswith(".") or part in self._SKIP_DIRS for part in relative.parts):
            return True
        text = relative.as_posix()
        return any(
            fnmatch.fnmatch(text, pattern.rstrip("/"))
            or fnmatch.fnmatch(path.name, pattern.rstrip("/"))
            or (pattern.endswith("/") and text.startswith(pattern))
            for pattern in self._ignore_patterns
        )

    @staticmethod
    def _page(request: dict) -> tuple[int, int]:
        offset = request.get("offset", 0)
        limit = request.get("limit", 30)
        if not isinstance(offset, int) or offset < 0 or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("offset 必须非负，limit 必须在 1 到 100 之间")
        return offset, limit

    def _file_text(self, path: Path) -> str:
        if self._ignored(path):
            raise ValueError("文件不允许访问")
        if not path.is_file():
            raise ValueError("文件不存在")
        if path.stat().st_size > 1_000_000:
            raise ValueError("文件超过 1 MB，请缩小搜索范围")
        return path.read_text(encoding="utf-8", errors="replace")

    def _walk_files(self, root: Path):
        for current, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = sorted(name for name in dirs if not self._ignored(Path(current) / name))
            for name in sorted(files):
                path = Path(current) / name
                if not self._ignored(path) and path.is_file() and path.stat().st_size <= 1_000_000:
                    yield path

    def _paged(self, action: str, items: list, offset: int, limit: int) -> str:
        page = items[offset:offset + limit]
        next_offset = offset + len(page) if offset + len(page) < len(items) else None
        result = {"action": action, "items": page, "next_offset": next_offset}
        while page and len(json.dumps(result, ensure_ascii=False)) > self.max_output_chars:
            page.pop()
            result["next_offset"] = offset + len(page)
        if not page and items[offset:]:
            raise ValueError("输出预算太小，无法返回一条结果")
        return json.dumps(result, ensure_ascii=False)

    def run(self, input_text: str) -> str:
        request = json.loads(input_text)
        if not isinstance(request, dict):
            raise ValueError("输入必须是 JSON 对象")
        action = request.get("action")
        path = self._resolve(request.get("path", "."))
        if action in {"list", "glob", "search"} and not path.is_dir():
            raise ValueError("目录不存在")
        if action == "list":
            offset, limit = self._page(request)
            items = [
                {"path": str(child.relative_to(self.workspace)), "type": "directory" if child.is_dir() else "file"}
                for child in sorted(path.iterdir(), key=lambda child: child.name)
                if not self._ignored(child)
            ]
            return self._paged(action, items, offset, limit)
        if action == "glob":
            offset, limit = self._page(request)
            pattern = request.get("pattern")
            if not isinstance(pattern, str) or not pattern or ".." in Path(pattern).parts or pattern.startswith("/"):
                raise ValueError("glob 需要工作目录内的 pattern")
            items = [
                str(file.relative_to(self.workspace)) for file in self._walk_files(path)
                if fnmatch.fnmatch(str(file.relative_to(path)), pattern)
                or fnmatch.fnmatch(file.name, pattern)
            ]
            return self._paged(action, items, offset, limit)
        if action == "read":
            lines = self._file_text(path).splitlines()
            start = request.get("start_line", 1)
            start_column = request.get("start_column", 0)
            end = request.get("end_line", min(len(lines), start + 49) if isinstance(start, int) else 50)
            if (not isinstance(start, int) or not isinstance(end, int) or start < 1
                or end < start or end - start > 199 or not isinstance(start_column, int)
                or start_column < 0):
                raise ValueError("行号无效；一次最多读取 200 行")
            selected = [
                f"{number}: {lines[number - 1][start_column if number == start else 0:]}"
                for number in range(start, min(end, len(lines)) + 1)
            ]
            content = "\n".join(selected)
            headroom = max(self.max_output_chars - 180, 1)
            truncated = len(content) > headroom
            next_line = min(end, len(lines)) + 1 if end < len(lines) else None
            next_column = 0 if next_line is not None else None
            if truncated:
                position = 0
                for number, rendered in zip(range(start, min(end, len(lines)) + 1), selected):
                    if position + len(rendered) >= headroom:
                        prefix_length = len(f"{number}: ")
                        used = max(0, headroom - position - prefix_length)
                        next_line = number
                        next_column = (start_column if number == start else 0) + used
                        break
                    position += len(rendered) + 1
                content = content[:headroom]
            return json.dumps({
                "action": action, "path": str(path.relative_to(self.workspace)),
                "start_line": start, "end_line": min(end, len(lines)),
                "total_lines": len(lines), "content": content,
                "truncated": truncated, "next_line": next_line, "next_column": next_column,
            }, ensure_ascii=False)
        if action == "search":
            query = request.get("query")
            if not isinstance(query, str) or not query:
                raise ValueError("search 需要非空 query")
            offset, limit = self._page(request)
            matches = []
            for file in self._walk_files(path):
                for number, line in enumerate(self._file_text(file).splitlines(), 1):
                    if query.casefold() in line.casefold():
                        matches.append({
                            "path": str(file.relative_to(self.workspace)),
                            "line": number, "text": line[:240],
                        })
                        if len(matches) >= offset + limit + 1:
                            break
                if len(matches) >= offset + limit + 1:
                    break
            return self._paged(action, matches, offset, limit)
        raise ValueError("action 必须是 list/glob/read/search")
