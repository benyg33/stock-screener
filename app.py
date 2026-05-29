from flask import Flask, render_template, request, jsonify
import os
from screener import screen_tickers, SP500_TICKERS, format_market_cap, get_stock_history, get_stock_news

app = Flask(__name__)
DEFAULT_TICKERS = SP500_TICKERS[:25]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/screen", methods=["POST"])
def api_screen():
    data = request.get_json() or {}
    use_sp500 = data.get("use_sp500", True)
    custom_tickers = [t.strip().upper() for t in data.get("custom_tickers", []) if t.strip()]
    limit = int(data.get("limit", 50))
    sector_filter = data.get("sector", "All")

    if use_sp500:
        tickers = list(dict.fromkeys(DEFAULT_TICKERS[:limit] + custom_tickers))
    else:
        tickers = custom_tickers if custom_tickers else DEFAULT_TICKERS[:20]

    results = screen_tickers(tickers)

    for r in results:
        r["market_cap_fmt"] = format_market_cap(r.get("market_cap"))

    # Compute sectors from ALL results before any filtering
    sectors = sorted(set(r["sector"] for r in results if r.get("sector") not in ("N/A", None, "")))

    # Filter for buys/sells/count from all results
    buys = [r for r in results if r["overall_rec"] in ("Strong Buy", "Buy")][:5]
    sells = [r for r in results if r["overall_rec"] in ("Strong Sell", "Sell")][:5]

    return jsonify({
        "results": results,
        "top_buys": buys,
        "top_sells": sells,
        "sectors": sectors,
        "total": len(results),
        "screened": len(tickers),
    })

@app.route("/api/detail/<ticker>")
def api_detail(ticker):
    results = screen_tickers([ticker.upper()])
    if results:
        r = results[0]
        r["market_cap_fmt"] = format_market_cap(r.get("market_cap"))
        return jsonify(r)
    return jsonify({"error": "Not found"}), 404

@app.route("/api/history/<ticker>")
def api_history(ticker):
    period = request.args.get("period", "6mo")
    data = get_stock_history(ticker.upper(), period)
    if data:
        return jsonify(data)
    return jsonify({"error": "No data"}), 404

@app.route("/api/news/<ticker>")
def api_news(ticker):
    articles = get_stock_news(ticker.upper())
    return jsonify({"news": articles})

@app.route("/api/test")
def api_test():
    import traceback
    try:
        import yfinance as yf
        ticker = yf.Ticker("AAPL")
        info = ticker.info
        hist = ticker.history(period="5d")
        return jsonify({
            "status": "ok",
            "price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "name": info.get("shortName"),
            "hist_rows": len(hist),
            "info_keys": list(info.keys())[:10]
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e), "trace": traceback.format_exc()})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port)
