"""Illustrative data for the local travel demo; none of it is live travel data."""

CITIES = {
    "北京": {
        "attractions": [
            {"id": "bj_palace", "name": "故宫", "category": "历史文化", "minutes": 180, "ticket": 60, "x": 49, "y": 34},
            {"id": "bj_jingshan", "name": "景山公园", "category": "自然风光", "minutes": 90, "ticket": 10, "x": 52, "y": 25},
            {"id": "bj_temple", "name": "天坛", "category": "历史文化", "minutes": 150, "ticket": 30, "x": 55, "y": 64},
            {"id": "bj_summer", "name": "颐和园", "category": "自然风光", "minutes": 180, "ticket": 30, "x": 22, "y": 22},
            {"id": "bj_798", "name": "798艺术区", "category": "艺术街区", "minutes": 120, "ticket": 0, "x": 79, "y": 28},
            {"id": "bj_hutong", "name": "南锣鼓巷", "category": "城市漫步", "minutes": 100, "ticket": 0, "x": 59, "y": 27},
            {"id": "bj_museum", "name": "中国国家博物馆", "category": "博物馆", "minutes": 150, "ticket": 0, "x": 52, "y": 45},
            {"id": "bj_olympic", "name": "奥林匹克森林公园", "category": "自然风光", "minutes": 120, "ticket": 0, "x": 52, "y": 9},
            {"id": "bj_qianmen", "name": "前门大街", "category": "美食购物", "minutes": 90, "ticket": 0, "x": 50, "y": 55},
        ],
        "hotels": [
            {"name": "京城漫游酒店（模拟）", "type": "舒适型", "nightly": 380},
            {"name": "京城轻旅客栈（模拟）", "type": "经济型", "nightly": 240},
        ],
        "meals": ["京味早餐（模拟）", "胡同家常菜（模拟）", "京味晚餐（模拟）"],
        "temperature": 22,
    },
    "上海": {
        "attractions": [
            {"id": "sh_bund", "name": "外滩", "category": "城市漫步", "minutes": 100, "ticket": 0, "x": 62, "y": 40},
            {"id": "sh_yuyuan", "name": "豫园", "category": "历史文化", "minutes": 120, "ticket": 40, "x": 57, "y": 54},
            {"id": "sh_museum", "name": "上海博物馆", "category": "博物馆", "minutes": 150, "ticket": 0, "x": 47, "y": 44},
            {"id": "sh_lujiazui", "name": "陆家嘴", "category": "城市地标", "minutes": 100, "ticket": 0, "x": 74, "y": 40},
            {"id": "sh_tianzifang", "name": "田子坊", "category": "艺术街区", "minutes": 100, "ticket": 0, "x": 41, "y": 64},
            {"id": "sh_wukang", "name": "武康路", "category": "城市漫步", "minutes": 90, "ticket": 0, "x": 28, "y": 49},
            {"id": "sh_expo", "name": "世博文化公园", "category": "自然风光", "minutes": 120, "ticket": 0, "x": 66, "y": 77},
            {"id": "sh_1933", "name": "1933老场坊", "category": "艺术街区", "minutes": 90, "ticket": 0, "x": 67, "y": 23},
            {"id": "sh_nanjing", "name": "南京路步行街", "category": "美食购物", "minutes": 90, "ticket": 0, "x": 50, "y": 35},
        ],
        "hotels": [
            {"name": "海派漫游酒店（模拟）", "type": "舒适型", "nightly": 420},
            {"name": "海派轻旅客栈（模拟）", "type": "经济型", "nightly": 260},
        ],
        "meals": ["海派早餐（模拟）", "本帮风味午餐（模拟）", "夜上海晚餐（模拟）"],
        "temperature": 24,
    },
    "杭州": {
        "attractions": [
            {"id": "hz_westlake", "name": "西湖", "category": "自然风光", "minutes": 180, "ticket": 0, "x": 44, "y": 48},
            {"id": "hz_lingyin", "name": "灵隐寺", "category": "历史文化", "minutes": 150, "ticket": 45, "x": 24, "y": 42},
            {"id": "hz_xixi", "name": "西溪湿地", "category": "自然风光", "minutes": 180, "ticket": 80, "x": 17, "y": 25},
            {"id": "hz_museum", "name": "浙江省博物馆", "category": "博物馆", "minutes": 120, "ticket": 0, "x": 50, "y": 36},
            {"id": "hz_hefang", "name": "河坊街", "category": "美食购物", "minutes": 100, "ticket": 0, "x": 59, "y": 64},
            {"id": "hz_longjing", "name": "龙井村", "category": "自然风光", "minutes": 130, "ticket": 0, "x": 26, "y": 63},
            {"id": "hz_canal", "name": "大运河杭州段", "category": "城市漫步", "minutes": 120, "ticket": 0, "x": 68, "y": 24},
            {"id": "hz_song", "name": "宋城", "category": "历史文化", "minutes": 150, "ticket": 90, "x": 31, "y": 79},
            {"id": "hz_liangzhu", "name": "良渚博物院", "category": "博物馆", "minutes": 130, "ticket": 0, "x": 56, "y": 12},
        ],
        "hotels": [
            {"name": "杭城漫游酒店（模拟）", "type": "舒适型", "nightly": 360},
            {"name": "杭城轻旅客栈（模拟）", "type": "经济型", "nightly": 230},
        ],
        "meals": ["杭式早餐（模拟）", "杭帮风味午餐（模拟）", "江南晚餐（模拟）"],
        "temperature": 23,
    },
}
