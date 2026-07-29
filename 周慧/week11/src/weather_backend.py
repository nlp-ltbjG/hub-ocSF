import httpx

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo 天气代码 → 中文描述映射
WEATHER_CODE_MAP = {
    0: "晴天",
    1: "大致晴朗",
    2: "局部多云",
    3: "阴天",
    45: "雾",
    48: "冻雾",
    51: "小毛毛雨",
    53: "中毛毛雨",
    55: "大毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    80: "小阵雨",
    81: "中阵雨",
    82: "大阵雨",
    95: "雷暴",
    96: "雷暴伴小冰雹",
    99: "雷暴伴大冰雹",
}


def get_coordinates_by_city(city: str) -> dict | None:
    """
    根据城市名称查询经纬度及相关地理信息。

    中国地名常有歧义：裸"宁德"会命中西藏那曲市的一个村（PPL），
    而宁德时代总部所在的福建宁德是地级市"宁德市"（PPLA2）。
    策略：先按用户输入查；若命中的只是低级行政点（feature_code 纯 PPL），
    且用户没带"市/县/区"后缀，就用 city+"市" 重查一次并优先采用。

    Args:
        city: 城市名称，支持中文，例如 "宁德"、"北京"、"上海"

    Returns:
        包含地理信息的字典：{"lat": 纬度, "lon": 经度, "city_name": 城市名,
                              "country": 国家, "admin1": 省/州级行政区}，
        未找到城市时返回 None
    """
    with httpx.Client(timeout=10.0) as client:

        def _geocode(name: str):
            resp = client.get(
                GEOCODING_URL,
                params={
                    "name": name,
                    "count": 10,
                    "language": "zh",
                    "format": "json",
                },
            )
            resp.raise_for_status()
            return resp.json().get("results") or []

        results = _geocode(city)
        is_low_admin = (
            all(
                str(r.get("feature_code", "")).startswith("PPL")
                and not str(r.get("feature_code", "")).startswith("PPLA")
                for r in results
            )
            if results
            else True
        )
        has_suffix = any(city.endswith(s) for s in ("市", "县", "区", "镇"))
        if is_low_admin and not has_suffix:
            retry = _geocode(city + "市")
            if retry:
                results = retry

        if not results:
            return None

        def _rank(r):
            fc = str(r.get("feature_code", ""))
            admin_priority = 1 if fc.startswith("PPLA") or fc.startswith("ADM") else 0
            pop = r.get("population") or 0
            return (admin_priority, pop)

        loc = max(results, key=_rank)
        return {
            "lat": loc["latitude"],
            "lon": loc["longitude"],
            "city_name": loc.get("name", city),
            "country": loc.get("country", ""),
            "admin1": loc.get("admin1", ""),
        }


def get_weather_by_coordinates(
    lat: float, lon: float, city_name: str, country: str = "", admin1: str = ""
) -> str:
    """
    根据经纬度查询当前天气及未来3天预报。

    Args:
        lat: 纬度
        lon: 经度
        city_name: 城市名称（用于格式化输出）
        country: 国家（用于格式化输出）
        admin1: 省/州级行政区（用于格式化输出）

    Returns:
        包含温度、湿度、风速、天气状况和3天预报的文字描述
    """
    with httpx.Client(timeout=10.0) as client:
        try:
            weather_resp = client.get(
                WEATHER_URL,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
                    "timezone": "Asia/Shanghai",
                    "forecast_days": 3,
                },
            )
            weather_resp.raise_for_status()
        except httpx.RequestError as e:
            return f"天气数据获取失败：{e}"

        data = weather_resp.json()
        cur = data["current"]
        daily = data["daily"]

        weather_desc = WEATHER_CODE_MAP.get(
            cur["weather_code"], f"代码{cur['weather_code']}"
        )
        location_str = f"{country} {admin1} {city_name}".strip()

        lines = [
            f"【{location_str}】天气报告",
            f"坐标：{lat:.2f}°N, {lon:.2f}°E",
            "",
            f"当前天气：{weather_desc}",
            f"  温度：{cur['temperature_2m']}°C",
            f"  相对湿度：{cur['relative_humidity_2m']}%",
            f"  风速：{cur['wind_speed_10m']} km/h",
            "",
            "未来3天预报：",
        ]
        for i in range(3):
            day_desc = WEATHER_CODE_MAP.get(daily["weather_code"][i], "")
            lines.append(
                f"  {daily['time'][i]}：{day_desc}，"
                f"{daily['temperature_2m_max'][i]}°C / {daily['temperature_2m_min'][i]}°C，"
                f"降水 {daily['precipitation_sum'][i]} mm"
            )

        return "\n".join(lines)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="天气查询工具")
    parser.add_argument("--city", help="城市名称，例如 '宁德'")
    parser.add_argument("--lat", type=float, help="纬度，与 --lon 配合使用")
    parser.add_argument("--lon", type=float, help="经度，与 --lat 配合使用")
    parser.add_argument("--name", help="地点名称（仅经纬度模式下使用）")
    args = parser.parse_args()

    if args.lat is not None and args.lon is not None:
        city_name = args.name or f"{args.lat:.2f}°N, {args.lon:.2f}°E"
        print(get_weather_by_coordinates(args.lat, args.lon, city_name))
    elif args.city is None and args.lat is None:
        parser.print_help()
