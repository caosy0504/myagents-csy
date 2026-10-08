"""Convert supported files into normalized Markdown text."""

from pathlib import Path


class DocumentConverter:
    TEXT_EXTENSIONS = {
        ".md",
        ".markdown",
        ".txt",
        ".py",
        ".js",
        ".ts",
        ".json",
        ".csv",
        ".yaml",
        ".yml",
    }

    def __init__(self, max_file_size_mb: int = 50) -> None:
        if max_file_size_mb <= 0:
            raise ValueError("max_file_size_mb 必须大于 0")
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

    def convert(self, file_path: str | Path) -> str:
        path = Path(file_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"文档不存在：{path}")
        if path.stat().st_size > self.max_file_size_bytes:
            max_mb = self.max_file_size_bytes // (1024 * 1024)
            raise ValueError(f"文档超过 {max_mb} MB 大小限制：{path.name}")

        if path.suffix.lower() in self.TEXT_EXTENSIONS:
            text = path.read_text(encoding="utf-8", errors="replace")
        else:
            try:
                from markitdown import MarkItDown
            except ImportError as exc:
                raise RuntimeError(
                    "该文件格式需要 MarkItDown，请安装：pip install -e '.[rag]'"
                ) from exc
            # 插件默认关闭，减少处理不可信文档时的额外执行面。
            result = MarkItDown(enable_plugins=False).convert(str(path))
            text = getattr(result, "text_content", "") or ""

        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized:
            raise ValueError(f"文档转换后没有可用文本：{path}")
        return normalized
