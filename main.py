import os
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from anthropic import Anthropic

app = Flask(__name__)
CORS(app)

TWELVE_DATA_KEY = os.environ.get("TWELVE_DATA_KEY")
ANTHROPIC_KEY   = os.environ.get("ANTHROPIC_API_KEY")
client          = Anthropic(api_key=ANTHROPIC_KEY)

# ─── الأزواج المدعومة ────────────────────────────────────────────
PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF",
    "AUD/USD", "USD/CAD", "XAU/USD", "EUR/GBP",
    "EUR/JPY", "GBP/JPY", "NZD/USD", "USD/TRY"
]

# ─── جلب السعر الحالي من Twelve Data ────────────────────────────
def get_price(symbol: str) -> dict:
    url = "https://api.twelvedata.com/price"
    r   = requests.get(url, params={"symbol": symbol, "apikey": TWELVE_DATA_KEY}, timeout=8)
    return r.json()

# ─── جلب بيانات الشموع (OHLCV) ──────────────────────────────────
def get_candles(symbol: str, interval: str = "1h", outputsize: int = 50) -> list:
    url = "https://api.twelvedata.com/time_series"
    r   = requests.get(url, params={
        "symbol":     symbol,
        "interval":   interval,
        "outputsize": outputsize,
        "apikey":     TWELVE_DATA_KEY
    }, timeout=10)
    data = r.json()
    return data.get("values", [])

# ─── حساب RSI يدوياً ─────────────────────────────────────────────
def calc_rsi(candles: list, period: int = 14) -> float:
    closes = [float(c["close"]) for c in candles]
    if len(closes) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(1, period + 1):
        diff = closes[i - 1] - closes[i]   # الأحدث أولاً في القائمة
        (gains if diff > 0 else losses).append(abs(diff))
    avg_gain = sum(gains) / period if gains else 0
    avg_loss = sum(losses) / period if losses else 1e-9
    rs  = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)

# ─── حساب EMA ────────────────────────────────────────────────────
def calc_ema(prices: list, period: int) -> float:
    if len(prices) < period:
        return prices[0] if prices else 0
    k   = 2 / (period + 1)
    ema = sum(prices[-period:]) / period
    for p in reversed(prices[:-period]):
        ema = p * k + ema * (1 - k)
    return round(ema, 5)

# ─── مستويات الدعم والمقاومة ─────────────────────────────────────
def get_support_resistance(candles: list) -> dict:
    highs  = [float(c["high"])  for c in candles[:20]]
    lows   = [float(c["low"])   for c in candles[:20]]
    closes = [float(c["close"]) for c in candles[:20]]
    return {
        "resistance": round(max(highs), 5),
        "support":    round(min(lows), 5),
        "avg_close":  round(sum(closes) / len(closes), 5)
    }

# ─── بناء سياق التحليل الكامل ───────────────────────────────────
def build_analysis_context(symbol: str, interval: str) -> dict:
    candles = get_candles(symbol, interval, 50)
    if not candles:
        return {}

    closes  = [float(c["close"]) for c in candles]
    current = closes[0]
    rsi     = calc_rsi(candles)
    ema20   = calc_ema(closes, 20)
    ema50   = calc_ema(closes, 50)
    sr      = get_support_resistance(candles)
    change  = round(current - closes[1], 5)
    change_pct = round((change / closes[1]) * 100, 3)

    return {
        "symbol":      symbol,
        "interval":    interval,
        "price":       round(current, 5),
        "change":      change,
        "change_pct":  change_pct,
        "rsi":         rsi,
        "ema20":       ema20,
        "ema50":       ema50,
        "support":     sr["support"],
        "resistance":  sr["resistance"],
        "avg_close":   sr["avg_close"],
        "candles_used": len(candles)
    }

# ─── بناء البرومبت لـ Claude ─────────────────────────────────────
def build_prompt(ctx: dict, extra_pairs: list) -> str:
    corr_text = "\n".join(
        f"- {p['symbol']}: {p['price']} ({'+' if p['change'] >= 0 else ''}{p['change']})"
        for p in extra_pairs
    )
    return f"""أنت أفضل محلل فوركس في العالم. تجمع بين التحليل الفني الدقيق، التحليل الأساسي، وارتباطات الأسواق.

═══ البيانات الحقيقية اللحظية ═══
الزوج:        {ctx['symbol']}
الإطار:       {ctx['interval']}
السعر:        {ctx['price']}
التغيير:      {ctx['change']} ({ctx['change_pct']}%)
RSI (14):     {ctx['rsi']}
EMA 20:       {ctx['ema20']}
EMA 50:       {ctx['ema50']}
مقاومة:       {ctx['resistance']}
دعم:          {ctx['support']}
المتوسط:      {ctx['avg_close']}

═══ أسواق مرتبطة (لحظية) ═══
{corr_text}

═══ مهمتك ═══
قدّم تحليلاً احترافياً منظماً يشمل:

1. الإشارة: [BUY / SELL / NEUTRAL]
2. دقة الإشارة: [نسبة %]
3. نقطة الدخول: [سعر]
4. Stop Loss: [سعر]
5. Take Profit 1: [سعر]
6. Take Profit 2: [سعر]
7. تحليل RSI: ماذا يقول؟
8. تحليل EMA: تقاطع؟ اتجاه؟
9. الدعم والمقاومة: قريب من أيهما؟
10. تأثير الارتباطات
11. ملخص الصفقة في جملتين

الإجابة بالعربية، منظمة، لا تتجاوز 350 كلمة.
"""

# ════════════════════════════════════════════════════════════════
#  API Endpoints
# ════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return jsonify({"status": "ForexAI Backend — Running ✅"})


@app.route("/api/prices")
def prices():
    """يجلب أسعار جميع الأزواج دفعة واحدة"""
    results = {}
    for pair in PAIRS:
        try:
            data  = get_price(pair)
            results[pair] = {"price": float(data.get("price", 0)), "ok": True}
        except Exception as e:
            results[pair] = {"price": 0, "ok": False, "error": str(e)}
    return jsonify(results)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """التحليل الكامل لزوج محدد"""
    body     = request.json or {}
    symbol   = body.get("symbol", "EUR/USD")
    interval = body.get("interval", "1h")

    # بيانات الزوج الرئيسي
    ctx = build_analysis_context(symbol, interval)
    if not ctx:
        return jsonify({"error": "فشل جلب البيانات — تحقق من مفتاح Twelve Data"}), 500

    # بيانات أسواق مرتبطة للسياق
    related = ["XAU/USD", "EUR/USD", "USD/JPY", "GBP/USD"]
    extra   = []
    for r in related:
        if r != symbol:
            try:
                d = get_price(r)
                c = get_candles(r, interval, 3)
                ch = round(float(c[0]["close"]) - float(c[1]["close"]), 5) if len(c) >= 2 else 0
                extra.append({"symbol": r, "price": float(d.get("price", 0)), "change": ch})
            except:
                pass

    # إرسال لـ Claude
    prompt = build_prompt(ctx, extra)
    try:
        message = client.messages.create(
            model      = "claude-opus-4-5",
            max_tokens = 1024,
            messages   = [{"role": "user", "content": prompt}]
        )
        ai_text = message.content[0].text
    except Exception as e:
        return jsonify({"error": f"Claude API Error: {str(e)}"}), 500

    # استخراج الإشارة والدقة تلقائياً
    import re
    signal_match = re.search(r'\b(BUY|SELL|NEUTRAL)\b', ai_text, re.IGNORECASE)
    conf_match   = re.search(r'(\d{2,3})\s*%', ai_text)

    return jsonify({
        "symbol":     symbol,
        "interval":   interval,
        "market_data": ctx,
        "related":    extra,
        "analysis":   ai_text,
        "signal":     signal_match.group(1).upper() if signal_match else "NEUTRAL",
        "confidence": int(conf_match.group(1)) if conf_match else 65,
    })


@app.route("/api/quick/<symbol>")
def quick(symbol):
    """بيانات سريعة لزوج واحد بدون تحليل AI"""
    symbol = symbol.replace("-", "/")
    ctx    = build_analysis_context(symbol, "1h")
    return jsonify(ctx)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
