"""Persistent Markdown notes for long-running tasks."""

import json
import os
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from math import log
from pathlib import Path
from uuid import UUID, uuid4

from .base import Tool


def _tokens(text: str) -> list[str]:
    """English words and overlapping Chinese bigrams avoid single-character noise."""
    result = []
    for part in re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", text.casefold()):
        if "\u4e00" <= part[0] <= "\u9fff" and len(part) > 1:
            result.extend(part[index:index + 2] for index in range(len(part) - 1))
        else:
            result.append(part)
    return result


class NoteTool(Tool):
    """Seven JSON actions: create, read, update, search, list, delete, summary."""

    def __init__(self, workspace: str | Path, project_id: str = "default") -> None:
        super().__init__(
            "note",
            '持久化项目笔记。输入 JSON，如 {"action":"search","query":"测试"}；'
            '或 {"action":"create","title":"进度","content":"已完成...","tags":["任务"]}',
        )
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.project_id = project_id

    def function_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name, "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["create", "read", "update", "search", "list", "delete", "summary"],
                        },
                        "id": {"type": "string"}, "title": {"type": "string"},
                        "content": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}},
                        "query": {"type": "string"}, "task_id": {"type": "string"},
                        "status": {"type": "string"}, "source": {"type": "string"},
                        "tag": {"type": "string"}, "limit": {"type": "integer"}, "offset": {"type": "integer"},
                    },
                    "required": ["action"], "additionalProperties": False,
                },
            },
        }

    def run_arguments(self, arguments: dict) -> str:
        return self.run(json.dumps(arguments, ensure_ascii=False))

    def _path(self, note_id: str) -> Path:
        try:
            safe_id = str(UUID(note_id))
        except (ValueError, AttributeError) as exc:
            raise ValueError("无效的笔记 ID") from exc
        path = self.workspace / f"{safe_id}.md"
        if not path.resolve().is_relative_to(self.workspace):
            raise ValueError("不允许访问笔记目录之外的路径")
        return path

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _write(self, record: dict) -> None:
        path = self._path(record["id"])
        keys = (
            "id", "title", "tags", "created_at", "updated_at",
            "project_id", "task_id", "status", "source", "verified_at",
        )
        frontmatter = "\n".join(f"{key}: {json.dumps(record[key], ensure_ascii=False)}" for key in keys)
        data = f"---\n{frontmatter}\n---\n{record['content']}\n"
        fd, temporary = tempfile.mkstemp(prefix=".note-", dir=self.workspace, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(data)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _read(self, note_id: str) -> dict:
        path = self._path(note_id)
        if not path.is_file():
            raise ValueError("笔记不存在")
        data = path.read_text(encoding="utf-8")
        parts = data.split("\n---\n", 1)
        if len(parts) != 2 or not parts[0].startswith("---\n"):
            raise ValueError("笔记格式错误")
        record = {}
        for line in parts[0][4:].splitlines():
            key, value = line.split(": ", 1)
            record[key] = json.loads(value)
        record["content"] = parts[1].rstrip("\n")
        record.setdefault("project_id", "default")
        record.setdefault("task_id", None)
        record.setdefault("status", "active")
        record.setdefault("source", None)
        record.setdefault("verified_at", None)
        return record

    def _all(self) -> list[dict]:
        records = []
        for path in self.workspace.glob("*.md"):
            try:
                record = self._read(path.stem)
                if record["project_id"] == self.project_id:
                    records.append(record)
            except (ValueError, KeyError, json.JSONDecodeError):
                continue
        return sorted(records, key=lambda item: item["updated_at"], reverse=True)

    @staticmethod
    def _brief(record: dict) -> dict:
        return {key: record[key] for key in (
            "id", "title", "tags", "project_id", "task_id", "status",
            "source", "verified_at", "updated_at",
        )}

    @staticmethod
    def _optional_text(request: dict, key: str):
        value = request.get(key)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{key} 必须是字符串或 null")
        return value

    @staticmethod
    def _excerpt(content: str, terms: set[str], width: int = 180) -> str:
        lowered = content.casefold()
        positions = [lowered.find(term) for term in terms if lowered.find(term) >= 0]
        start = max(0, min(positions) - 40) if positions else 0
        return content[start:start + width]

    def _search(self, request: dict) -> dict:
        query = request.get("query", "")
        if not isinstance(query, str):
            raise ValueError("query 必须是字符串")
        limit, offset = request.get("limit", 10), request.get("offset", 0)
        if not isinstance(limit, int) or not 1 <= limit <= 50 or not isinstance(offset, int) or offset < 0:
            raise ValueError("limit 必须在 1 到 50 之间，offset 必须非负")
        task_id = self._optional_text(request, "task_id")
        status = self._optional_text(request, "status")
        tag = self._optional_text(request, "tag")
        notes = [
            note for note in self._all()
            if (task_id is None or note["task_id"] == task_id)
            and (status is None or note["status"] == status)
            and (tag is None or tag in note["tags"])
        ]
        terms = set(_tokens(query))
        if not terms:
            ranked = [(0.0, note, set()) for note in notes]
        else:
            documents = [
                Counter(_tokens(note["title"]) * 3 + _tokens(" ".join(note["tags"])) * 2 + _tokens(note["content"]))
                for note in notes
            ]
            frequencies = Counter(term for doc in documents for term in doc)
            average_length = sum(sum(doc.values()) for doc in documents) / max(len(documents), 1)
            ranked = []
            for note, doc in zip(notes, documents):
                length = sum(doc.values())
                score = 0.0
                matched = set()
                for term in terms:
                    frequency = doc.get(term, 0)
                    if not frequency:
                        continue
                    matched.add(term)
                    idf = log(1 + (len(notes) - frequencies[term] + 0.5) / (frequencies[term] + 0.5))
                    score += idf * frequency * 2.2 / (
                        frequency + 1.2 * (0.25 + 0.75 * length / max(average_length, 1))
                    )
                if query.casefold() in (note["title"] + " " + note["content"]).casefold():
                    score += 2.0
                if score > 0:
                    ranked.append((score, note, matched))
            ranked.sort(key=lambda item: (item[0], item[1]["updated_at"]), reverse=True)
        items = [
            self._brief(note) | {
                "score": round(score, 4), "matched_terms": sorted(matched),
                "excerpt": self._excerpt(note["content"], matched),
                "source_path": self._path(note["id"]).name,
            }
            for score, note, matched in ranked[offset:offset + limit]
        ]
        next_offset = offset + len(items) if offset + len(items) < len(ranked) else None
        return {"items": items, "total": len(ranked), "next_offset": next_offset}

    def run(self, input_text: str) -> str:
        request = json.loads(input_text)
        if not isinstance(request, dict):
            raise ValueError("输入必须是 JSON 对象")
        action = request.get("action")
        if action == "create":
            title, content = request.get("title"), request.get("content")
            tags = request.get("tags", [])
            if not isinstance(title, str) or not title.strip() or not isinstance(content, str):
                raise ValueError("create 需要 title 和 content 字符串")
            if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
                raise ValueError("tags 必须是字符串数组")
            timestamp = self._now()
            record = {
                "id": str(uuid4()), "title": title.strip(), "content": content,
                "tags": tags, "created_at": timestamp, "updated_at": timestamp,
                "project_id": self.project_id,
                "task_id": self._optional_text(request, "task_id"),
                "status": request.get("status", "active"),
                "source": self._optional_text(request, "source"),
                "verified_at": self._optional_text(request, "verified_at"),
            }
            if record["status"] not in {"active", "blocked", "done", "reference"}:
                raise ValueError("status 必须是 active/blocked/done/reference")
            self._write(record)
            result = self._brief(record)
        elif action == "read":
            result = self._read(request.get("id", ""))
            if result["project_id"] != self.project_id:
                raise ValueError("笔记不属于当前项目")
        elif action == "update":
            record = self._read(request.get("id", ""))
            if record["project_id"] != self.project_id:
                raise ValueError("笔记不属于当前项目")
            for key in ("title", "content", "tags", "task_id", "status", "source", "verified_at"):
                if key in request:
                    value = request[key]
                    if key == "tags" and (not isinstance(value, list) or not all(isinstance(x, str) for x in value)):
                        raise ValueError("tags 必须是字符串数组")
                    if key in {"title", "content", "status"} and not isinstance(value, str):
                        raise ValueError(f"{key} 必须是字符串")
                    if key in {"task_id", "source", "verified_at"} and value is not None and not isinstance(value, str):
                        raise ValueError(f"{key} 必须是字符串或 null")
                    if key == "status" and value not in {"active", "blocked", "done", "reference"}:
                        raise ValueError("无效的 status")
                    record[key] = value
            record["updated_at"] = self._now()
            self._write(record)
            result = self._brief(record)
        elif action == "delete":
            path = self._path(request.get("id", ""))
            if not path.is_file():
                raise ValueError("笔记不存在")
            if self._read(path.stem)["project_id"] != self.project_id:
                raise ValueError("笔记不属于当前项目")
            path.unlink()
            result = {"deleted": path.stem}
        elif action in {"list", "search"}:
            if action == "search":
                result = self._search(request)
            else:
                result = [self._brief(note) for note in self._all()[:20]]
        elif action == "summary":
            records = self._all()
            result = {"count": len(records), "recent": [self._brief(note) for note in records[:10]]}
        else:
            raise ValueError("action 必须是 create/read/update/search/list/delete/summary")
        return json.dumps(result, ensure_ascii=False)
