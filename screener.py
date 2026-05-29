import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import time
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use a browser-like session to avoid Yahoo Finance blocking
_session = requests.Session()
_session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
})

SP500_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","AVGO","JPM",
    "LLY","V","UNH","XOM","MA","COST","HD","PG","WMT","JNJ","ORCL","ABBV",
    "BAC","CRM","MRK","CVX","NFLX","AMD","PEP","KO","TMO","ADBE","ACN","LIN",
    "CSCO","MCD","ABT","GE","DHR","TXN","CAT","NEE","QCOM","INTU","IBM","SPGI",
    "AMGN","MS","GS","HON","RTX","BKNG","ISRG","VRTX","SYK","T","UBER","BLK",
    "NOW","DE","ELV","PFE","MDT","CI","CB","SCHW","ADP","SO","DUK","MO","CL",
    "MMC","EOG","SLB","BSX","WM","ZTS","NOC","ITW","CME","USB","PNC","GD","AON",
    "APD","F","GM","PANW","KLAC","LRCX","MRVL","SNPS","CDNS","FTNT","CRWD","MU",
    "INTC","TGT","LOW","NKE","SBUX","DIS","PYPL","SQ","SHOP","ABNB","PLTR","COIN"
]

def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def score_technical(hist):
    if hist is None or len(hist) < 50:
        return 50, {}

    close = hist["Close"].squeeze()
    volume = hist["Volume"].squeeze()
    signals = {}
    score = 50

    # RSI
    rsi_series = compute_rsi(close)
    rsi = rsi_series.iloc[-1]
    signals["rsi"] = round(float(rsi), 1)
    if rsi < 30:
        score += 20
        signals["rsi_signal"] = "Oversold (Bullish)"
    elif rsi < 45:
        score += 10
        signals["rsi_signal"] = "Approaching Oversold"
    elif rsi > 70:
        score -= 20
        signals["rsi_signal"] = "Overbought (Bearish)"
    elif rsi > 55:
        score -= 5
        signals["rsi_signal"] = "Slightly Overbought"
    else:
        signals["rsi_signal"] = "Neutral"

    # MACD
    macd_line, signal_line = compute_macd(close)
    macd_val = macd_line.iloc[-1]
    signal_val = signal_line.iloc[-1]
    prev_macd = macd_line.iloc[-2]
    prev_signal = signal_line.iloc[-2]
    signals["macd"] = round(float(macd_val), 3)
    if macd_val > signal_val and prev_macd <= prev_signal:
        score += 15
        signals["macd_signal"] = "Bullish Crossover"
    elif macd_val > signal_val:
        score += 8
        signals["macd_signal"] = "Bullish"
    elif macd_val < signal_val and prev_macd >= prev_signal:
        score -= 15
        signals["macd_signal"] = "Bearish Crossover"
    elif macd_val < signal_val:
        score -= 8
        signals["macd_signal"] = "Bearish"
    else:
        signals["macd_signal"] = "Neutral"

    # Moving averages
    ma20 = close.rolling(20).mean().iloc[-1]
    ma50 = close.rolling(50).mean().iloc[-1]
    current = close.iloc[-1]
    signals["ma20"] = round(float(ma20), 2)
    signals["ma50"] = round(float(ma50), 2)
    signals["price"] = round(float(current), 2)
    if current > ma20 > ma50:
        score += 10
        signals["ma_signal"] = "Strong Uptrend"
    elif current > ma50:
        score += 5
        signals["ma_signal"] = "Above MA50"
    elif current < ma20 < ma50:
        score -= 10
        signals["ma_signal"] = "Strong Downtrend"
    elif current < ma50:
        score -= 5
        signals["ma_signal"] = "Below MA50"
    else:
        signals["ma_signal"] = "Neutral"

    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper_bb = (sma20 + 2 * std20).iloc[-1]
    lower_bb = (sma20 - 2 * std20).iloc[-1]
    signals["bb_upper"] = round(float(upper_bb), 2)
    signals["bb_lower"] = round(float(lower_bb), 2)
    if current < lower_bb:
        score += 12
        signals["bb_signal"] = "Below Lower Band (Bullish)"
    elif current > upper_bb:
        score -= 12
        signals["bb_signal"] = "Above Upper Band (Bearish)"
    elif current < float(sma20.iloc[-1]):
        score += 2
        signals["bb_signal"] = "Lower Half (Neutral)"
    else:
        signals["bb_signal"] = "Upper Half (Neutral)"

    # Momentum (20-day)
    momentum_20 = ((current - close.iloc[-20]) / close.iloc[-20]) * 100
    signals["momentum_20d"] = round(float(momentum_20), 2)
    if momentum_20 > 10:
        score += 8
    elif momentum_20 > 3:
        score += 4
    elif momentum_20 < -10:
        score -= 8
    elif momentum_20 < -3:
        score -= 4

    # Volume surge
    avg_vol = volume.rolling(20).mean().iloc[-1]
    recent_vol = volume.iloc[-1]
    vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1
    signals["volume_ratio"] = round(float(vol_ratio), 2)
    if vol_ratio > 2:
        signals["volume_signal"] = "High Volume Surge"
    elif vol_ratio > 1.3:
        signals["volume_signal"] = "Above Avg Volume"
    else:
        signals["volume_signal"] = "Normal Volume"

    # 52-week position
    high_52 = close.rolling(252).max().iloc[-1] if len(close) >= 252 else close.max()
    low_52 = close.rolling(252).min().iloc[-1] if len(close) >= 252 else close.min()
    pos_52w = ((current - low_52) / (high_52 - low_52)) * 100 if high_52 != low_52 else 50
    signals["pos_52w"] = round(float(pos_52w), 1)
    signals["high_52w"] = round(float(high_52), 2)
    signals["low_52w"] = round(float(low_52), 2)

    return max(0, min(100, score)), signals

def score_fundamental(info):
    if not info:
        return 50, {}

    score = 50
    signals = {}

    # P/E ratio
    pe = info.get("trailingPE") or info.get("forwardPE")
    if pe and pe > 0:
        signals["pe_ratio"] = round(float(pe), 1)
        if pe < 12:
            score += 15; signals["pe_signal"] = "Undervalued"
        elif pe < 20:
            score += 8; signals["pe_signal"] = "Fair Value"
        elif pe < 30:
            score -= 3; signals["pe_signal"] = "Slightly Elevated"
        elif pe < 50:
            score -= 10; signals["pe_signal"] = "Overvalued"
        else:
            score -= 18; signals["pe_signal"] = "Very Overvalued"

    # Forward P/E vs trailing
    fwd_pe = info.get("forwardPE")
    trail_pe = info.get("trailingPE")
    if fwd_pe and trail_pe and fwd_pe > 0 and trail_pe > 0:
        signals["forward_pe"] = round(float(fwd_pe), 1)
        if fwd_pe < trail_pe * 0.9:
            score += 8; signals["fwd_pe_signal"] = "Earnings Growing"
        elif fwd_pe > trail_pe * 1.1:
            score -= 5; signals["fwd_pe_signal"] = "Earnings Declining"

    # Revenue growth
    rev_growth = info.get("revenueGrowth")
    if rev_growth is not None:
        signals["revenue_growth"] = f"{round(rev_growth * 100, 1)}%"
        if rev_growth > 0.20: score += 12
        elif rev_growth > 0.10: score += 7
        elif rev_growth > 0.05: score += 3
        elif rev_growth < 0: score -= 10

    # Earnings growth
    eps_growth = info.get("earningsGrowth")
    if eps_growth is not None:
        signals["eps_growth"] = f"{round(eps_growth * 100, 1)}%"
        if eps_growth > 0.25: score += 12
        elif eps_growth > 0.10: score += 6
        elif eps_growth < 0: score -= 8

    # Profit margin
    margin = info.get("profitMargins")
    if margin is not None:
        signals["profit_margin"] = f"{round(margin * 100, 1)}%"
        if margin > 0.20: score += 8
        elif margin > 0.10: score += 4
        elif margin < 0: score -= 10

    # Return on equity
    roe = info.get("returnOnEquity")
    if roe is not None:
        signals["roe"] = f"{round(roe * 100, 1)}%"
        if roe > 0.20: score += 8
        elif roe > 0.10: score += 4
        elif roe < 0: score -= 8

    # Debt to equity
    de = info.get("debtToEquity")
    if de is not None:
        signals["debt_to_equity"] = round(float(de), 2)
        if de < 30: score += 6
        elif de < 80: score += 2
        elif de > 200: score -= 8

    # Analyst target upside
    target = info.get("targetMeanPrice")
    current = info.get("currentPrice") or info.get("regularMarketPrice")
    if target and current and current > 0:
        upside = ((target - current) / current) * 100
        signals["analyst_target"] = round(float(target), 2)
        signals["analyst_upside"] = f"{round(upside, 1)}%"
        if upside > 20: score += 12
        elif upside > 10: score += 6
        elif upside < -10: score -= 8

    # Analyst recommendation
    rec = info.get("recommendationKey", "").lower()
    signals["analyst_rec"] = rec.replace("_", " ").title() if rec else "N/A"
    if rec in ["strong_buy", "buy"]: score += 5
    elif rec in ["sell", "strong_sell"]: score -= 5

    # Earnings date
    try:
        earnings_ts = info.get("earningsTimestamp")
        if earnings_ts:
            ed = datetime.fromtimestamp(int(earnings_ts), tz=timezone.utc)
            now = datetime.now(tz=timezone.utc)
            days = (ed - now).days
            if 0 <= days <= 7:
                signals["earnings_date"] = f"In {days} days ⚡"
                signals["earnings_alert"] = True
            elif 0 <= days <= 45:
                signals["earnings_date"] = f"In {days} days"
            elif days < 0 and days > -14:
                signals["earnings_date"] = f"{abs(days)} days ago"
    except Exception:
        pass

    return max(0, min(100, score)), signals

def get_recommendation(score):
    if score >= 75: return "Strong Buy", "success"
    elif score >= 62: return "Buy", "primary"
    elif score >= 50: return "Hold", "warning"
    elif score >= 38: return "Sell", "danger"
    else: return "Strong Sell", "dark"

def _fetch_with_retry(fn, retries=3, delay=2):
    for i in range(retries):
        try:
            result = fn()
            if result is not None:
                return result
        except Exception as e:
            logger.warning(f"Fetch attempt {i+1} failed: {e}")
            if i < retries - 1:
                time.sleep(delay)
    return None

def screen_tickers(tickers, period="6mo"):
    results = []

    # Try batch download first
    raw = None
    if len(tickers) > 1:
        try:
            raw = yf.download(
                tickers, period=period, progress=False,
                group_by="ticker", threads=False, auto_adjust=True
            )
            if raw is None or raw.empty:
                raw = None
                logger.warning("Batch download returned empty data")
        except Exception as e:
            logger.warning(f"Batch download failed: {e}")
            raw = None

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)

            # Get info with retry
            info = _fetch_with_retry(lambda s=stock: s.info) or {}
            if not info or len(info) < 5:
                logger.warning(f"{ticker}: info empty, retrying...")
                time.sleep(1)
                stock2 = yf.Ticker(ticker)
                info = stock2.info or {}

            # Get history
            hist = None
            if raw is not None and len(tickers) > 1:
                try:
                    if ticker in raw.columns.get_level_values(0):
                        hist = raw[ticker].dropna()
                        if hist.empty:
                            hist = None
                except Exception:
                    hist = None

            if hist is None or (hasattr(hist, '__len__') and len(hist) < 10):
                hist = _fetch_with_retry(lambda s=stock: s.history(period=period, auto_adjust=True))

            logger.info(f"{ticker}: info keys={len(info)}, hist rows={len(hist) if hist is not None else 0}")

            tech_score, tech_signals = score_technical(hist)
            fund_score, fund_signals = score_fundamental(info)
            composite = round(tech_score * 0.40 + fund_score * 0.60, 1)

            short_rec, short_color = get_recommendation(tech_score)
            long_rec, long_color = get_recommendation(fund_score)
            overall_rec, overall_color = get_recommendation(composite)

            current_price = (info.get("currentPrice") or info.get("regularMarketPrice") or tech_signals.get("price"))

            results.append({
                "ticker": ticker,
                "name": info.get("shortName", ticker),
                "sector": info.get("sector", "N/A"),
                "price": round(float(current_price), 2) if current_price else "N/A",
                "market_cap": info.get("marketCap"),
                "tech_score": tech_score,
                "fund_score": fund_score,
                "composite_score": composite,
                "short_rec": short_rec, "short_color": short_color,
                "long_rec": long_rec, "long_color": long_color,
                "overall_rec": overall_rec, "overall_color": overall_color,
                "tech_signals": tech_signals,
                "fund_signals": fund_signals,
                "pe_ratio": fund_signals.get("pe_ratio", "N/A"),
                "revenue_growth": fund_signals.get("revenue_growth", "N/A"),
                "analyst_rec": fund_signals.get("analyst_rec", "N/A"),
                "analyst_upside": fund_signals.get("analyst_upside", "N/A"),
                "rsi": tech_signals.get("rsi", "N/A"),
                "momentum_20d": tech_signals.get("momentum_20d", "N/A"),
                "pos_52w": tech_signals.get("pos_52w", "N/A"),
                "bb_signal": tech_signals.get("bb_signal", "N/A"),
                "earnings_date": fund_signals.get("earnings_date", "N/A"),
                "earnings_alert": fund_signals.get("earnings_alert", False),
            })
        except Exception as e:
            results.append({
                "ticker": ticker, "name": ticker, "sector": "N/A",
                "price": "N/A", "market_cap": None,
                "tech_score": 50, "fund_score": 50, "composite_score": 50,
                "short_rec": "N/A", "short_color": "secondary",
                "long_rec": "N/A", "long_color": "secondary",
                "overall_rec": "N/A", "overall_color": "secondary",
                "tech_signals": {}, "fund_signals": {},
                "pe_ratio": "N/A", "revenue_growth": "N/A",
                "analyst_rec": "N/A", "analyst_upside": "N/A",
                "rsi": "N/A", "momentum_20d": "N/A", "pos_52w": "N/A",
                "bb_signal": "N/A", "earnings_date": "N/A", "earnings_alert": False,
                "error": str(e),
            })

    return sorted(results, key=lambda x: x["composite_score"], reverse=True)

def get_stock_history(ticker, period="6mo"):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period, auto_adjust=True)
    if hist.empty:
        return None
    close = hist["Close"].squeeze()
    volume = hist["Volume"].squeeze()
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    std20 = close.rolling(20).std()
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20

    def clean(series):
        return [None if (v != v or v == 0) else round(float(v), 2) for v in series]

    return {
        "dates": hist.index.strftime("%Y-%m-%d").tolist(),
        "closes": clean(close),
        "volumes": [int(v) for v in volume],
        "ma20": clean(sma20),
        "ma50": clean(sma50),
        "bb_upper": clean(upper_bb),
        "bb_lower": clean(lower_bb),
    }

def get_stock_news(ticker):
    try:
        stock = yf.Ticker(ticker)
        raw_news = stock.news or []
        articles = []
        for item in raw_news[:10]:
            # Handle both old and new yfinance news formats
            if "content" in item:
                content = item["content"]
                title = content.get("title", "")
                publisher = content.get("provider", {}).get("displayName", "") if isinstance(content.get("provider"), dict) else ""
                link = content.get("canonicalUrl", {}).get("url", "") if isinstance(content.get("canonicalUrl"), dict) else ""
                pub_time = content.get("pubDate", "")
            else:
                title = item.get("title", "")
                publisher = item.get("publisher", "")
                link = item.get("link", "")
                pub_time = item.get("providerPublishTime", 0)
                if pub_time:
                    try:
                        pub_time = datetime.fromtimestamp(int(pub_time)).strftime("%b %d, %Y")
                    except Exception:
                        pub_time = ""
            if title:
                articles.append({"title": title, "publisher": publisher, "link": link, "time": pub_time})
        return articles
    except Exception:
        return []

def format_market_cap(val):
    if not val: return "N/A"
    val = float(val)
    if val >= 1e12: return f"${val/1e12:.1f}T"
    elif val >= 1e9: return f"${val/1e9:.1f}B"
    elif val >= 1e6: return f"${val/1e6:.1f}M"
    return f"${val:,.0f}"
