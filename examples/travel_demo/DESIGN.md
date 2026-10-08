# 第 13 章本地旅行助手实战

这个案例把 MyAgents CSY 的 `SimpleAgent`、`MyAgentsLLM` 和 `ToolRegistry` 组合成一个可操作的网页应用。模型固定连接本机 Ollama `qwen2.5:7b`；没有外部地图、图片、天气或云端模型请求。

## 功能

- 北京、上海、杭州，1–3 日行程；支持偏好、节奏、交通、住宿、补充要求与预算上限。轻松节奏每天 2 个景点、标准 2–3 个、紧凑 3 个。
- 模型只选择模拟景点 ID、每日主题和文字建议；程序验证城市、天数、景点 ID、每日数量及跨日重复。
- Python 代码根据模拟时长生成早餐、景点、午餐、晚餐时间线；根据模拟票价、住宿、餐饮、交通计算每日及总预算。超预算时只提示差额。
- 浏览器显示 SVG 景点分布示意图、模拟天气、流程说明；支持添加、移除、排序、跨日移动景点。编辑调用 `/api/rebuild`，不再调用 Ollama。
- 当前浏览器自动保存草稿；可导出 JSON 或通过打印保存为 PDF。

## 数据流

```mermaid
flowchart LR
    UI[原生 HTML/CSS/JS 页面] --> API[Python 本地 HTTP 服务]
    API --> REQ[TravelRequest 校验]
    REQ --> TOOL[ToolRegistry: 三个本地模拟工具]
    TOOL --> AGENT[SimpleAgent + 本机 Ollama]
    AGENT --> CHECK[JSON 与景点 ID 校验]
    CHECK --> BUILD[确定性时间线与预算]
    BUILD --> UI
    UI -->|编辑景点| REBUILD[/api/rebuild: 重新计算]
    REBUILD --> BUILD
```

| 文件 | 作用 |
|---|---|
| `data.py` | 模拟城市、景点、酒店、餐饮与示意坐标 |
| `planner.py` | 请求校验、模拟工具、模型选择、结果验证、时间线和预算 |
| `server.py` | 只监听 `127.0.0.1:8765` 的本地 API 与静态文件服务 |
| `index.html` / `style.css` / `app.js` | 表单、行程、示意图、编辑与导出 |
| `../../tests/test_travel_demo.py` | 使用 Fake LLM 的离线测试 |

## 启动

在 `MyAgents-csy-main` 目录执行：

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .
ollama pull qwen2.5:7b
# 若 Ollama 尚未在后台运行，在另一个终端执行：ollama serve
.venv/bin/python -m examples.travel_demo.server
```

浏览器打开 <http://127.0.0.1:8765>。主要接口：`POST /api/plan` 首次生成；`POST /api/rebuild` 根据已选景点重新计算；`GET /api/health` 检查本地服务。测试命令：`.venv/bin/python -m pytest tests/test_travel_demo.py -q`（先安装 `.[dev]`）。

## 边界

这里的景点资料、酒店、天气、价格、位置和时间均为**模拟数据**；示意图不是地图，道路线和交通耗时没有计算。模型选择结果需要满足程序校验；失败会返回错误，不会把编造的景点呈现为事实。当前案例是单 Agent 工作流，没有实现第 13 章原项目的多 Agent 调度或真实地图接入。
