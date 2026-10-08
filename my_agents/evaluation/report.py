"""Small machine-readable and human-readable reports for chapter 12."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def render_report(report: dict[str, Any]) -> str:
    kind = report.get("benchmark", "evaluation")
    lines = [f"# {kind} 评估报告", "", f"生成时间（UTC）：{datetime.now(timezone.utc).isoformat()}", ""]
    if report.get("official_score") is False:
        lines.extend(["**性质：本地评估，不是官方榜单成绩。**", ""])
    lines.extend(["## 汇总", ""])
    for key, value in report.items():
        if key in {"samples", "rows", "plan", "stages", "final_test"}:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            lines.append(f"- {key}: {value}")
        elif key.endswith("_ci95") and isinstance(value, dict):
            lines.append(f"- {key}: [{value['lower']:.3f}, {value['upper']:.3f}]")
    lines.append("")
    if "categories" in report:
        lines.extend(["## 类别", "", "| 类别 | 样本数 | 工具选择 | 参数匹配 |",
                      "| --- | ---: | ---: | ---: |"])
        for name, stats in report["categories"].items():
            lines.append(
                f"| {name} | {stats['count']} | {stats['selection_accuracy']:.1%} | "
                f"{stats['argument_accuracy']:.1%} |"
            )
        lines.append("")
    if "levels" in report:
        lines.extend(["## 难度", "", "| Level | 已评分 | 准确率 |",
                      "| --- | ---: | ---: |"])
        for level, stats in report["levels"].items():
            lines.append(f"| {level} | {stats['count']} | {stats['accuracy']:.1%} |")
        lines.append("")
    if "suites" in report:
        lines.extend(["## 固定评估组", ""])
        for name, item in report["suites"].items():
            lines.append(f"### {name}")
            lines.append("")
            for key, value in item["metrics"].items():
                lines.append(f"- {key}: {value}")
            lines.append("")
    if "judge" in report:
        judge = report["judge"]
        lines.extend([
            "## 生成数据质量", "",
            f"- Judge 有效评分：{judge['valid_count']}/{judge['sample_count']}",
            f"- Judge 平均分：{judge['mean_score']}",
        ])
    if "pairwise" in report:
        pair = report["pairwise"]
        lines.extend([
            "- 双顺序成对比较：胜 "
            f"{pair['wins']}、负 {pair['losses']}、平 {pair['ties']}",
            f"- 顺序矛盾：{pair.get('disagreement_count', 0)}",
            f"- 胜率：{pair['win_rate']}",
        ])
    if "human_review" in report:
        lines.append("- 人工复核：等待填写 review_queue.jsonl；未自动批准任何题目。")
    if any(key in report for key in ("judge", "pairwise", "human_review")):
        lines.append("")
    if "calibration" in report:
        audit = report["calibration"]
        lines.extend([
            "## Judge 与人工复核对照", "",
            f"- 可比较题目：{audit['matched_count']}",
            f"- 通过判定一致率：{audit['agreement']}",
            f"- 分维度平均绝对误差：{audit['dimension_mae']}",
            f"- 不一致题目：{audit['disagreements']}", "",
        ])
    errors = [row for row in report.get("samples", report.get("rows", [])) if row.get("error")]
    if errors:
        lines.extend(["## 错误", ""])
        for row in errors[:20]:
            lines.append(f"- {row.get('id', row.get('task_id', row.get('problem_id', '?')))}: {row['error']}")
        lines.append("")
    return "\n".join(lines) + "\n"


def save_report(report: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "report.json"
    markdown_path = output / "report.md"
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError(f"Evaluation report already exists in {output}")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
                         encoding="utf-8")
    markdown_path.write_text(render_report(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}
