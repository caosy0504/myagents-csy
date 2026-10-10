const DRAFT_KEY = 'myagents-csy-travel-demo-v3';
const PREFERENCES = ['历史文化', '自然风光', '博物馆', '城市漫步', '美食购物', '艺术街区'];
const DAY_COLORS = ['#248866', '#d18d4d', '#597cb3'];

const form = document.getElementById('trip-form');
const results = document.getElementById('results');
const empty = document.getElementById('empty');
const errorBox = document.getElementById('error');
const formStatus = document.getElementById('form-status');
const generateButton = document.getElementById('generate');
const chipContainer = document.getElementById('preference-chips');
const citySelect = document.getElementById('city');
let selectedPreferences = new Set(['历史文化', '美食购物']);
let plan = null;
let editing = false;

function el(tag, className = '', text = '') {
  const node = document.createElement(tag);
  node.className = className;
  node.textContent = text;
  return node;
}

function svgEl(tag, attrs = {}) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
  return node;
}

function money(value) {
  return `¥${Number(value).toLocaleString('zh-CN')}`;
}

function dateText(value) {
  const date = new Date(`${value}T00:00:00`);
  return `${date.getMonth() + 1}月${date.getDate()}日`;
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function clearError() {
  errorBox.hidden = true;
  errorBox.textContent = '';
}

async function api(path, body) {
  const response = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `请求失败 (${response.status})`);
  return data;
}

function renderPreferenceChips() {
  chipContainer.replaceChildren();
  PREFERENCES.forEach(preference => {
    const chip = el('button', 'pref-chip', preference);
    chip.type = 'button';
    chip.setAttribute('aria-pressed', selectedPreferences.has(preference) ? 'true' : 'false');
    chip.addEventListener('click', () => {
      if (selectedPreferences.has(preference)) selectedPreferences.delete(preference);
      else if (selectedPreferences.size < 5) selectedPreferences.add(preference);
      else { showError('最多选择 5 项旅行偏好。'); return; }
      clearError();
      renderPreferenceChips();
    });
    chipContainer.append(chip);
  });
}

function defaultDate() {
  const date = new Date();
  date.setDate(date.getDate() + 7);
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  document.getElementById('start-date').value = `${date.getFullYear()}-${month}-${day}`;
}

async function loadCities() {
  try {
    const response = await fetch('/api/cities');
    if (!response.ok) return;
    const data = await response.json();
    if (!Array.isArray(data.cities) || !data.cities.length) return;
    const selected = citySelect.value;
    citySelect.replaceChildren();
    data.cities.sort((a, b) => a.localeCompare(b, 'zh-CN'))
      .forEach(city => citySelect.append(new Option(city, city)));
    citySelect.value = data.cities.includes(selected) ? selected : data.cities[0];
    document.getElementById('city-count').textContent = `◌ ${data.cities.length} 座演示城市`;
    const researchInput = document.getElementById('web-research');
    researchInput.disabled = !data.tavily_available;
    document.getElementById('research-help').textContent = data.tavily_available
      ? '仅勾选后才联网搜索；一次基础搜索通常消耗 1 个 Tavily API credit。'
      : '如需联网搜索，请先在本项目 .env 填写 TAVILY_API_KEY 并重启服务。';
  } catch (error) {
    // Keep the three HTML fallback options if the local service is unavailable.
  }
}

function requestFromForm() {
  const rawBudget = document.getElementById('budget-limit').value.trim();
  return {
    city: document.getElementById('city').value,
    start_date: document.getElementById('start-date').value,
    days: Number(document.getElementById('days').value),
    preferences: [...selectedPreferences],
    transportation: document.getElementById('transport').value,
    accommodation: document.getElementById('hotel').value,
    pace: document.getElementById('pace').value,
    budget_limit: rawBudget ? Number(rawBudget) : null,
    web_research: document.getElementById('web-research').checked,
    notes: document.getElementById('notes').value.trim(),
  };
}

function formFromRequest(request) {
  document.getElementById('city').value = request.city;
  document.getElementById('start-date').value = request.start_date;
  document.getElementById('days').value = String(request.days);
  document.getElementById('transport').value = request.transportation;
  document.getElementById('hotel').value = request.accommodation;
  document.getElementById('pace').value = request.pace;
  document.getElementById('budget-limit').value = request.budget_limit ?? '';
  document.getElementById('web-research').checked = Boolean(request.web_research);
  document.getElementById('notes').value = request.notes;
  selectedPreferences = new Set(request.preferences);
  renderPreferenceChips();
}

form.addEventListener('submit', async event => {
  event.preventDefault();
  clearError();
  generateButton.disabled = true;
  generateButton.firstChild.textContent = '本地模型正在规划… ';
  formStatus.textContent = document.getElementById('web-research').checked
    ? '正在调用 Tavily 搜索和本机 Ollama；本次搜索会消耗 Tavily API credit。'
    : '正在读取演示数据并请求本机 Ollama，首次生成可能需要几十秒。';
  try {
    plan = await api('/api/plan', requestFromForm());
    localStorage.setItem(DRAFT_KEY, JSON.stringify(plan));
    render();
    formStatus.textContent = '已生成并保存在此浏览器。';
    results.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (error) {
    showError(error.message);
    formStatus.textContent = '';
  } finally {
    generateButton.disabled = false;
    generateButton.firstChild.textContent = '重新生成演示行程 ';
  }
});

function button(label, onClick, className = 'ghost-button') {
  const node = el('button', className, label);
  node.type = 'button';
  node.addEventListener('click', onClick);
  return node;
}

function downloadJson() {
  const blob = new Blob([JSON.stringify(plan, null, 2)], { type: 'application/json;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `模拟行程-${plan.city}-${plan.start_date}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function clearDraft() {
  localStorage.removeItem(DRAFT_KEY);
  plan = null;
  results.classList.remove('visible');
  results.replaceChildren();
  empty.hidden = false;
  formStatus.textContent = '已清除浏览器中的演示草稿。';
}

function renderToolbar() {
  const toolbar = el('div', 'toolbar');
  toolbar.append(el('span', 'toolbar-status', '草稿自动保存在此浏览器'));
  toolbar.append(button('导出 JSON', downloadJson));
  toolbar.append(button('打印 / 保存 PDF', () => window.print()));
  toolbar.append(button('清除草稿', clearDraft));
  return toolbar;
}

function renderOverview() {
  const card = el('section', 'card overview');
  const top = el('div', 'topline');
  const heading = el('div');
  heading.append(
    el('h2', '', `${plan.city} · ${plan.days.length} 日行程`),
    el('div', 'sub', `${dateText(plan.start_date)} — ${dateText(plan.end_date)} · ${plan.model} · ${plan.request.pace}节奏`),
  );
  top.append(heading, el('span', 'badge', plan.research ? '联网来源 + 模拟数据' : '本地模拟数据'));
  card.append(top);

  const metrics = el('div', 'metrics');
  const attractionCount = plan.days.reduce((total, day) => total + day.attractions.length, 0);
  [
    ['景点', `${attractionCount} 处`],
    ['住宿', `${Math.max(plan.days.length - 1, 0)} 晚`],
    ['模拟总预算', money(plan.budget.total)],
  ].forEach(([label, value]) => {
    const metric = el('div', 'metric');
    metric.append(el('span', '', label), el('b', '', value));
    metrics.append(metric);
  });
  card.append(metrics);

  const budget = el('div', 'budget-grid');
  [
    ['门票', plan.budget.tickets], ['住宿', plan.budget.lodging],
    ['餐饮', plan.budget.meals], ['交通', plan.budget.transport],
  ].forEach(([label, value]) => {
    const item = el('div', 'budget-item');
    item.append(el('span', '', label), el('b', '', money(value)));
    budget.append(item);
  });
  card.append(budget);

  if (plan.budget_limit !== null) {
    const over = plan.budget_remaining < 0;
    const limit = el('div', over ? 'budget-limit over' : 'budget-limit');
    limit.append(el('strong', '', over
      ? `模拟预算超出 ${money(-plan.budget_remaining)}`
      : `模拟预算剩余 ${money(plan.budget_remaining)}`));
    limit.append(el('div', '', `预算上限 ${money(plan.budget_limit)} · 当前估算 ${money(plan.budget.total)}`));
    const track = el('div', 'budget-track');
    const fill = el('div', 'budget-fill');
    fill.style.width = `${Math.min(100, plan.budget.total / plan.budget_limit * 100)}%`;
    track.append(fill);
    limit.append(track);
    card.append(limit);
  }

  const weather = el('div', 'weather-strip');
  plan.weather.forEach(item => weather.append(el('span', 'weather-chip',
    `${dateText(item.date)} · 模拟${item.label} ${item.temperature}°C`)));
  card.append(weather);
  return card;
}

function renderMap() {
  const card = el('section', 'card');
  card.append(el('span', 'section-number', '02 / 路线示意'), el('h2', '', '景点分布与每日路线'));
  const layout = el('div', 'map-layout');
  const svg = svgEl('svg', { class: 'map', viewBox: '0 0 100 100', role: 'img', 'aria-label': '模拟景点分布图' });
  for (let n = 10; n < 100; n += 10) {
    svg.append(svgEl('line', { x1: n, y1: 0, x2: n, y2: 100, stroke: '#dce8dc', 'stroke-width': '.23' }));
    svg.append(svgEl('line', { x1: 0, y1: n, x2: 100, y2: n, stroke: '#dce8dc', 'stroke-width': '.23' }));
  }
  plan.days.forEach((day, dayIndex) => {
    const color = DAY_COLORS[dayIndex];
    if (day.attractions.length > 1) {
      svg.append(svgEl('polyline', {
        points: day.attractions.map(poi => `${poi.x},${poi.y}`).join(' '),
        fill: 'none', stroke: color, 'stroke-width': '.9', 'stroke-dasharray': '1.8 1.2',
      }));
    }
    day.attractions.forEach((poi, index) => {
      svg.append(svgEl('circle', { cx: poi.x, cy: poi.y, r: 2.4, fill: color, stroke: 'white', 'stroke-width': '.7' }));
      const label = svgEl('text', { x: poi.x + 3, y: poi.y + 1, fill: '#264c3b', 'font-size': '3', 'font-weight': '700' });
      label.textContent = `${dayIndex + 1}.${index + 1} ${poi.name}`;
      svg.append(label);
    });
  });
  layout.append(svg);
  const legend = el('div', 'map-legend');
  plan.days.forEach((day, index) => {
    const item = el('span', '', `● 第 ${index + 1} 天 · ${day.attractions.length} 处`);
    item.style.color = DAY_COLORS[index];
    legend.append(item);
  });
  layout.append(legend);
  card.append(layout, el('p', 'small-note', '手工示意坐标仅用于查看景点排列，路线连线不表示真实道路或交通时间。'));
  return card;
}

function selection() {
  return plan.days.map(day => ({
    attractions: day.attractions.map(poi => poi.id), theme: day.theme,
  }));
}

async function edit(mutator) {
  if (editing) return;
  clearError();
  const days = selection();
  try {
    mutator(days);
  } catch (error) {
    showError(error.message);
    return;
  }
  editing = true;
  results.style.pointerEvents = 'none';
  formStatus.textContent = '正在本地重算时间与预算，不会再次调用 Ollama。';
  try {
    plan = await api('/api/rebuild', {
      request: plan.request, days, tips: plan.tips,
      model_attempts: plan.workflow.model_attempts,
      research: plan.research,
    });
    localStorage.setItem(DRAFT_KEY, JSON.stringify(plan));
    render();
    formStatus.textContent = '已更新行程和预算。';
  } catch (error) {
    showError(error.message);
    formStatus.textContent = '';
  } finally {
    editing = false;
    results.style.pointerEvents = '';
  }
}

function renderEvent(event, dayIndex) {
  const row = el('div', `event ${event.kind}`);
  row.append(el('span', 'event-time', `${event.start}–${event.end}`));
  const main = el('div', 'event-main');
  main.append(el('div', 'event-title', event.title), el('div', 'event-note', event.note));
  if (event.kind === 'attraction') {
    const poi = plan.days[dayIndex].attractions.find(item => item.id === event.poi_id);
    const index = plan.days[dayIndex].attractions.findIndex(item => item.id === event.poi_id);
    main.append(el('div', 'event-note', `${poi.category} · 模拟门票 ${money(poi.ticket)}`));
    const actions = el('div', 'event-actions');
    const up = button('上移', () => edit(days => {
      [days[dayIndex].attractions[index - 1], days[dayIndex].attractions[index]] =
        [days[dayIndex].attractions[index], days[dayIndex].attractions[index - 1]];
    }), 'mini-button');
    up.disabled = index === 0;
    const down = button('下移', () => edit(days => {
      [days[dayIndex].attractions[index + 1], days[dayIndex].attractions[index]] =
        [days[dayIndex].attractions[index], days[dayIndex].attractions[index + 1]];
    }), 'mini-button');
    down.disabled = index === plan.days[dayIndex].attractions.length - 1;
    actions.append(up, down, button('移除', () => edit(days => {
      days[dayIndex].attractions.splice(index, 1);
    }), 'mini-button'));
    if (plan.days.length > 1) {
      const move = el('select', 'move-select');
      move.setAttribute('aria-label', `移动${poi.name}到其他日期`);
      move.append(new Option('移至…', ''));
      plan.days.forEach((day, targetIndex) => {
        if (targetIndex !== dayIndex) move.append(new Option(`第 ${targetIndex + 1} 天`, String(targetIndex)));
      });
      move.addEventListener('change', () => {
        const target = Number(move.value);
        if (plan.days[target].attractions.length >= 3) {
          showError(`第 ${target + 1} 天最多安排 3 个景点。`);
          move.value = '';
          return;
        }
        edit(days => {
          days[dayIndex].attractions.splice(index, 1);
          days[target].attractions.push(poi.id);
        });
      });
      actions.append(move);
    }
    main.append(actions);
  }
  row.append(main);
  return row;
}

function renderDay(day, dayIndex) {
  const card = el('section', 'card day-card');
  const header = el('div', 'day-head');
  const title = el('div');
  title.append(el('h3', '', `第 ${dayIndex + 1} 天 · ${day.theme}`), el('p', '', dateText(day.date)));
  header.append(title, el('span', 'day-budget', `当日模拟费用 ${money(day.budget.total)}`));
  card.append(header);
  const summary = el('div', 'day-summary');
  summary.append(el('span', '', `🚏 ${day.transportation}`));
  summary.append(el('span', '', day.budget.lodging
    ? `🏨 ${day.hotel.name} · ${money(day.hotel.nightly)}/晚`
    : `🏨 住宿参考：${day.hotel.name}（当日未计住宿）`));
  summary.append(el('span', '', `${day.attractions.length} 个景点`));
  card.append(summary);

  const timeline = el('div', 'timeline');
  day.timeline.forEach(event => timeline.append(renderEvent(event, dayIndex)));
  card.append(timeline);

  const used = new Set(plan.days.flatMap(item => item.attractions.map(poi => poi.id)));
  const available = plan.catalog.filter(poi => !used.has(poi.id));
  const addRow = el('div', 'add-row');
  addRow.append(el('span', '', '加入景点'));
  const select = el('select', 'add-select');
  select.setAttribute('aria-label', `向第 ${dayIndex + 1} 天加入景点`);
  available.forEach(poi => select.append(new Option(`${poi.name} · ${poi.category}`, poi.id)));
  select.disabled = day.attractions.length >= 3 || available.length === 0;
  const addButton = button('添加并重算', () => edit(days => {
    if (days[dayIndex].attractions.length >= 3) throw new Error('每天最多安排 3 个景点。');
    days[dayIndex].attractions.push(select.value);
  }), 'mini-button');
  addButton.disabled = select.disabled;
  addRow.append(select, addButton);
  card.append(addRow);
  card.append(el('div', 'day-foot', `餐饮：${day.meals.map(meal => `${meal.name} ${money(meal.estimated_cost)}`).join(' · ')}。时间线包含模拟缓冲，并非实际交通安排。`));
  return card;
}

function renderNotes() {
  const card = el('section', 'card');
  card.append(el('span', 'section-number', '04 / 说明与执行过程'), el('h2', '', '建议与系统流程'));
  card.append(el('div', 'tip-box', plan.tips));
  if (plan.edited) card.append(el('p', 'small-note', '模型建议来自初始行程；景点编辑后的时间线和预算已重新计算，请以当前行程为准。'));
  const details = el('details');
  const summary = el('summary', '', '查看这份行程怎样生成');
  details.append(summary);
  const list = el('ol', 'workflow-list');
  plan.workflow.stages.forEach(stage => list.append(el('li', '', stage)));
  details.append(list, el('p', 'workflow-tools',
    `工具：${plan.workflow.tools.join('、')} · 本地模型调用 ${plan.workflow.model_attempts} 次。编辑后只重新计算，不再调用模型。`));
  card.append(details, el('p', 'small-note', plan.data_notice));
  if (plan.research && Array.isArray(plan.research.sources)) {
    const sources = el('div', 'research-sources');
    sources.append(el('h3', '', 'Tavily 搜索参考来源'));
    plan.research.sources.forEach((source, index) => {
      try {
        const url = new URL(source.url);
        if (url.protocol !== 'https:') return;
        const item = document.createElement('a');
        item.href = url.href;
        item.target = '_blank';
        item.rel = 'noopener noreferrer';
        item.textContent = `${index + 1}. ${source.title}`;
        sources.append(item);
      } catch (error) { /* Ignore a malformed saved source URL. */ }
    });
    sources.append(el('p', 'small-note', '搜索摘要只用于辅助在本地景点目录中选点；请以景区官方信息核对实时价格与开放情况。'));
    card.append(sources);
  }
  return card;
}

function render() {
  if (!plan) return;
  empty.hidden = true;
  results.classList.add('visible');
  results.replaceChildren(renderToolbar(), renderOverview(), renderMap());
  results.append(el('span', 'section-number', '03 / 每日行程'));
  plan.days.forEach((day, index) => results.append(renderDay(day, index)));
  results.append(renderNotes());
}

defaultDate();
renderPreferenceChips();
try {
  const draft = JSON.parse(localStorage.getItem(DRAFT_KEY));
  if (draft && draft.schema_version === 3 && draft.request && Array.isArray(draft.days)) {
    plan = draft;
    formFromRequest(draft.request);
    render();
    formStatus.textContent = '已恢复此浏览器保存的演示草稿。';
  }
} catch (error) {
  localStorage.removeItem(DRAFT_KEY);
}
loadCities();
