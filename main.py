import os
import re
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

TWELVE_DATA_KEY = os.environ.get("TWELVE_DATA_KEY")
GROQ_API_KEY    = os.environ.get("GROQ_API_KEY")

PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF",
    "AUD/USD", "USD/CAD", "XAU/USD", "EUR/GBP",
    "EUR/JPY", "GBP/JPY", "NZD/USD", "USD/TRY"
]

def get_price(symbol):
    r = requests.get("https://api.twelvedata.com/price",
        params={"symbol": symbol, "apikey": TWELVE_DATA_KEY}, timeout=8)
    return r.json()

def get_candles(symbol, interval="1h", outputsize=50):
    r = requests.get("https://api.twelvedata.com/time_series",
        params={"symbol": symbol, "interval": interval,
                "outputsize": outputsize, "apikey": TWELVE_DATA_KEY}, timeout=10)
data = r.json()
if "values" not in data:
    print(f"Twelve Data Error: {data}")
return data.get("values", [])

def calc_rsi(candles, period=14):
    closes = [float(c["close"]) for c in candles]
    if len(closes) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(1, period + 1):
        diff = closes[i - 1] - closes[i]
        (gains if diff > 0 else losses).append(abs(diff))
    avg_gain = sum(gains) / period if gains else 0
    avg_loss = sum(losses) / period if losses else 1e-9
    return round(100 - (100 / (1 + avg_gain / avg_loss)), 2)

def calc_ema(prices, period):
    if len(prices) < period:
        return prices[0] if prices else 0
    k = 2 / (period + 1)
    ema = sum(prices[-period:]) / period
    for p in reversed(prices[:-period]):
        ema = p * k + ema * (1 - k)
    return round(ema, 5)

def get_sr(candles):
    highs  = [float(c["high"])  for c in candles[:20]]
    lows   = [float(c["low"])   for c in candles[:20]]
    closes = [float(c["close"]) for c in candles[:20]]
    return {
        "resistance": round(max(highs), 5),
        "support":    round(min(lows), 5),
        "avg_close":  round(sum(closes) / len(closes), 5)
    }

def build_context(symbol, interval):
    candles = get_candles(symbol, interval, 50)
    if not candles:
        return {}
    closes = [float(c["close"]) for c in candles]
    current = closes[0]
    sr = get_sr(candles)
    return {
        "symbol": symbol, "interval": interval,
        "price": round(current, 5),
        "change": round(current - closes[1], 5),
        "change_pct": round((current - closes[1]) / closes[1] * 100, 3),
        "rsi": calc_rsi(candles),
        "ema20": calc_ema(closes, 20),
        "ema50": calc_ema(closes, 50),
        "support": sr["support"],
        "resistance": sr["resistance"],
        "avg_close": sr["avg_close"],
    }

def ask_groq(prompt):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.3
    }
    r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                      headers=headers, json=body, timeout=30)
    return r.json()["choices"][0]["message"]["content"]

def build_prompt(ctx, extra):
    corr = "\n".join(
        f"- {p['symbol']}: {p['price']} ({'+' if p['change']>=0 else ''}{p['change']})"
        for p in extra
    )
    return f"""أنت أفضل محلل فوركس في العالم. تجمع بين التحليل الفني الدقيق والأساسي وارتباطات الأسواق.

البيانات الحقيقية اللحظية:
الزوج: {ctx['symbol']} | الإطار: {ctx['interval']}
السعر: {ctx['price']} | التغيير: {ctx['change']} ({ctx['change_pct']}%)
RSI(14): {ctx['rsi']} | EMA20: {ctx['ema20']} | EMA50: {ctx['ema50']}
مقاومة: {ctx['resistance']} | دعم: {ctx['support']}

أسواق مرتبطة:
{corr}

قدم تحليلاً منظماً يشمل:
1. الإشارة: BUY او SELL او NEUTRAL
2. دقة الإشارة: نسبة مئوية
3. نقطة الدخول
4. Stop Loss
5. Take Profit 1
6. Take Profit 2
7. تحليل RSI
8. تحليل EMA
9. الدعم والمقاومة
10. ملخص الصفقة

الإجابة بالعربية ولا تتجاوز 350 كلمة.
"""

@app.route("/")
def index():
    return jsonify({"status": "ForexAI Backend - Running"})

@app.route("/api/prices")
def prices():
    results = {}
    for pair in PAIRS:
        try:
            data = get_price(pair)
            results[pair] = {"price": float(data.get("price", 0)), "ok": True}
        except Exception as e:
            results[pair] = {"price": 0, "ok": False, "error": str(e)}
    return jsonify(results)

@app.route("/api/analyze", methods=["POST"])
def analyze():
    body     = request.json or {}
    symbol   = body.get("symbol", "EUR/USD")
    interval = body.get("interval", "1h")

    ctx = build_context(symbol, interval)
    if not ctx:
        return jsonify({"error": "فشل جلب البيانات"}), 500

    extra = []
    for rel in ["XAU/USD", "EUR/USD", "USD/JPY", "GBP/USD"]:
        if rel != symbol:
            try:
                d = get_price(rel)
                c = get_candles(rel, interval, 3)
                ch = round(float(c[0]["close"]) - float(c[1]["close"]), 5) if len(c) >= 2 else 0
                extra.append({"symbol": rel, "price": float(d.get("price", 0)), "change": ch})
            except:
                pass

    try:
        ai_text = ask_groq(build_prompt(ctx, extra))
    except Exception as e:
        return jsonify({"error": f"Groq Error: {str(e)}"}), 500

    signal_match = re.search(r'\b(BUY|SELL|NEUTRAL)\b', ai_text, re.IGNORECASE)
    conf_match   = re.search(r'(\d{2,3})\s*%', ai_text)

    return jsonify({
        "symbol": symbol, "interval": interval,
        "market_data": ctx, "related": extra,
        "analysis": ai_text,
        "signal": signal_match.group(1).upper() if signal_match else "NEUTRAL",
        "confidence": int(conf_match.group(1)) if conf_match else 65,
    })

@app.route("/api/quick/<symbol>")
def quick(symbol):
    symbol = symbol.replace("-", "/")
    return jsonify(build_context(symbol, "1h"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
