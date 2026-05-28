from flask import Flask, render_template, request, jsonify
import os
from screener import screen_tickers, SP500_TICKERS, format_market_cap

app = Flask(__name__)

DEFAULT_TICKERS = SP500_TICKERS[:50]  # Start with top 50 for speed

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

    # Sector filter
    if sector_filter and sector_filter != "All":
        results = [r for r in results if r.get("sector") == sector_filter]

    # Format market cap
    for r in results:
        r["market_cap_fmt"] = format_market_cap(r.get("market_cap"))

    # Top picks
    buys = [r for r in results if r["overall_rec"] in ("Strong Buy", "Buy")][:5]
    sells = [r for r in results if r["overall_rec"] in ("Strong Sell", "Sell")][:5]

    # Sector breakdown
    sectors = sorted(set(r["sector"] for r in results if r["sector"] != "N/A"))

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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port)
