import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import time
import gc
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

SECTOR_MAP = {
    "AAPL":"Technology","MSFT":"Technology","NVDA":"Technology","AVGO":"Technology",
    "ORCL":"Technology","CRM":"Technology","AMD":"Technology","ADBE":"Technology",
    "ACN":"Technology","CSCO":"Technology","TXN":"Technology","QCOM":"Technology",
    "INTU":"Technology","IBM":"Technology","NOW":"Technology","PANW":"Technology",
    "KLAC":"Technology","LRCX":"Technology","MRVL":"Technology","SNPS":"Technology",
    "CDNS":"Technology","FTNT":"Technology","CRWD":"Technology","MU":"Technology",
    "INTC":"Technology","SHOP":"Technology","PLTR":"Technology","ADP":"Technology",
    "AMZN":"Consumer Cyclical","TSLA":"Consumer Cyclical","HD":"Consumer Cyclical",
    "MCD":"Consumer Cyclical","BKNG":"Consumer Cyclical","LOW":"Consumer Cyclical",
    "NKE":"Consumer Cyclical","SBUX":"Consumer Cyclical","F":"Consumer Cyclical",
    "GM":"Consumer Cyclical","ABNB":"Consumer Cyclical","TGT":"Consumer Defensive",
    "COST":"Consumer Defensive","PG":"Consumer Defensive","WMT":"Consumer Defensive",
    "PEP":"Consumer Defensive","KO":"Consumer Defensive","MO":"Consumer Defensive",
    "CL":"Consumer Defensive",
    "GOOGL":"Communication Services","META":"Communication Services",
    "NFLX":"Communication Services","T":"Communication Services",
    "DIS":"Communication Services","CMCSA":"Communication Services",
    "JPM":"Financial Services","BRK-B":"Financial Services","V":"Financial Services",
    "MA":"Financial Services","BAC":"Financial Services","GS":"Financial Services",
    "MS":"Financial Services","SPGI":"Financial Services","BLK":"Financial Services",
    "SCHW":"Financial Services","CB":"Financial Services","CME":"Financial Services",
    "USB":"Financial Services","PNC":"Financial Services","AON":"Financial Services",
    "MMC":"Financial Services","PYPL":"Financial Services","SQ":"Financial Services",
    "COIN":"Financial Services",
    "LLY":"Healthcare","UNH":"Healthcare","JNJ":"Healthcare","ABBV":"Healthcare",
    "MRK":"Healthcare","TMO":"Healthcare","ABT":"Healthcare","DHR":"Healthcare",
    "AMGN":"Healthcare","ISRG":"Healthcare","VRTX":"Healthcare","SYK":"Healthcare",
    "ELV":"Healthcare","PFE":"Healthcare","MDT":"Healthcare","CI":"Healthcare",
    "BSX":"Healthcare","ZTS":"Healthcare",
    "XOM":"Energy","CVX":"Energy","EOG":"Energy","SLB":"Energy",
    "GE":"Industrials","CAT":"Industrials","HON":"Industrials","RTX":"Industrials",
    "DE":"Industrials","NOC":"Industrials","ITW":"Industrials","WM":"Industrials",
    "GD":"Industrials","UBER":"Industrials",
    "NEE":"Utilities","SO":"Utilities","DUK":"Utilities",
    "LIN":"Basic Materials","APD":"Basic Materials",
}

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

def generate_reason(ticker, overall_rec, tech_score, fund_score, tech_signals, fund_signals):
    tech_points = []
    fund_points = []

    # --- Technical ---
    rsi = tech_signals.get("rsi")
    if rsi is not None:
        if rsi < 30:   tech_points.append(f"RSI is oversold at {rsi:.0f}")
        elif rsi > 70: tech_points.append(f"RSI is overbought at {rsi:.0f}")

    macd = tech_signals.get("macd_signal", "")
    if "Bullish Crossover" in macd:   tech_points.append("MACD just flashed a bullish crossover")
    elif "Bearish Crossover" in macd: tech_points.append("MACD just flashed a bearish crossover")
    elif macd == "Bullish":           tech_points.append("MACD is bullish")
    elif macd == "Bearish":           tech_points.append("MACD is bearish")

    ma = tech_signals.get("ma_signal", "")
    if "Strong Uptrend" in ma:    tech_points.append("price is in a strong uptrend above both MA20 and MA50")
    elif "Strong Downtrend" in ma: tech_points.append("price is in a strong downtrend below both moving averages")
    elif "Above MA50" in ma:       tech_points.append("price is trading above its 50-day moving average")
    elif "Below MA50" in ma:       tech_points.append("price is below its 50-day moving average")

    bb = tech_signals.get("bb_signal", "")
    if "Below Lower" in bb: tech_points.append("price is below the lower Bollinger Band (oversold)")
    elif "Above Upper" in bb: tech_points.append("price is above the upper Bollinger Band (overbought)")

    mom = tech_signals.get("momentum_20d")
    if mom is not None:
        if mom > 10:    tech_points.append(f"strong 20-day momentum of +{mom:.1f}%")
        elif mom < -10: tech_points.append(f"weak 20-day momentum of {mom:.1f}%")

    # --- Fundamental ---
    pe = fund_signals.get("pe_ratio")
    pe_sig = fund_signals.get("pe_signal", "")
    if pe:
        if "Undervalued" in pe_sig:      fund_points.append(f"P/E of {pe} looks undervalued")
        elif "Very Overvalued" in pe_sig: fund_points.append(f"P/E of {pe} looks very overvalued")
        elif "Overvalued" in pe_sig:      fund_points.append(f"P/E of {pe} is elevated")
        elif "Fair Value" in pe_sig:      fund_points.append(f"P/E of {pe} is at fair value")

    rev = fund_signals.get("revenue_growth", "")
    if rev and rev != "N/A":
        try:
            v = float(rev.replace("%", ""))
            if v > 20:   fund_points.append(f"revenue growing strongly at {rev}")
            elif v > 5:  fund_points.append(f"revenue growing at {rev}")
            elif v < 0:  fund_points.append(f"revenue declining at {rev}")
        except Exception:
            pass

    eps = fund_signals.get("eps_growth", "")
    if eps and eps != "N/A":
        try:
            v = float(eps.replace("%", ""))
            if v > 20:  fund_points.append(f"earnings growing at {eps}")
            elif v < 0: fund_points.append(f"earnings declining at {eps}")
        except Exception:
            pass

    margin = fund_signals.get("profit_margin", "")
    if margin and margin != "N/A":
        try:
            v = float(margin.replace("%", ""))
            if v > 20: fund_points.append(f"healthy profit margin of {margin}")
            elif v < 0: fund_points.append(f"negative profit margin of {margin}")
        except Exception:
            pass

    upside = fund_signals.get("analyst_upside", "")
    target = fund_signals.get("analyst_target")
    if upside and upside != "N/A" and target:
        try:
            v = float(upside.replace("%", ""))
            if v > 15:    fund_points.append(f"analyst target of ${target} implies {upside} upside")
            elif v < -10: fund_points.append(f"analyst target of ${target} implies {upside} downside")
        except Exception:
            pass

    arec = fund_signals.get("analyst_rec", "N/A")
    if arec not in ("N/A", "None", "", "Hold"):
        if any(w in arec.lower() for w in ("buy", "strong buy")):
            fund_points.append(f"Wall Street rates it a {arec}")
        elif any(w in arec.lower() for w in ("sell", "strong sell")):
            fund_points.append(f"Wall Street rates it a {arec}")

    # --- Build sentence ---
    all_points = tech_points[:2] + fund_points[:3]  # cap at ~5 reasons
    if not all_points:
        return f"Not enough data available to generate a detailed explanation for {ticker}."

    rec_phrase = {
        "Strong Buy":  "is a Strong Buy",
        "Buy":         "is rated Buy",
        "Hold":        "is rated Hold — neither a clear buy nor a sell",
        "Sell":        "is rated Sell",
        "Strong Sell": "is a Strong Sell",
    }.get(overall_rec, f"has an {overall_rec} rating")

    if len(all_points) == 1:
        body = all_points[0]
    elif len(all_points) == 2:
        body = f"{all_points[0]} and {all_points[1]}"
    else:
        body = ", ".join(all_points[:-1]) + f", and {all_points[-1]}"

    score_note = ""
    if tech_score != fund_score:
        better = "technicals" if tech_score > fund_score else "fundamentals"
        score_note = f" Short-term score: {tech_score}/100, long-term: {fund_score}/100 — {better} are stronger."

    return f"{ticker} {rec_phrase} because {body}.{score_note}"


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

BATCH_SIZE = 10  # Small batches to stay under 512MB RAM on Render free tier

def screen_tickers(tickers, period="6mo"):
    results = []

    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        logger.info(f"Processing batch {i//BATCH_SIZE + 1}: {batch}")

        raw = None
        if len(batch) > 1:
            try:
                raw = yf.download(batch, period=period, progress=False,
                                  group_by="ticker", threads=False, auto_adjust=True)
                if raw is None or raw.empty:
                    raw = None
            except Exception as e:
                logger.warning(f"Batch download failed: {e}")
                raw = None

        for ticker in batch:
            try:
                stock = yf.Ticker(ticker)
                info = {}

                # fast_info uses chart API (reliable on cloud, not rate-limited)
                try:
                    fi = stock.fast_info
                    info["currentPrice"] = getattr(fi, "last_price", None)
                    info["marketCap"] = getattr(fi, "market_cap", None)
                    info["trailingPE"] = getattr(fi, "pe_ratio", None)
                    info["forwardPE"] = getattr(fi, "forward_pe", None)
                    info["fiftyTwoWeekHigh"] = getattr(fi, "year_high", None) or getattr(fi, "fifty_two_week_high", None)
                    info["fiftyTwoWeekLow"] = getattr(fi, "year_low", None) or getattr(fi, "fifty_two_week_low", None)
                except Exception as e:
                    logger.warning(f"{ticker} fast_info failed: {e}")

                # Try full info — may work sometimes, adds fundamentals
                try:
                    full_info = stock.info or {}
                    if len(full_info) > 10:
                        # Merge, preferring full_info values
                        for k, v in full_info.items():
                            if v is not None:
                                info[k] = v
                        logger.info(f"{ticker}: full info loaded ({len(full_info)} keys)")
                    else:
                        logger.warning(f"{ticker}: full info sparse ({len(full_info)} keys), using fast_info only")
                except Exception as e:
                    logger.warning(f"{ticker}: full info failed: {e}")

                # Try analyst targets separately
                if not info.get("targetMeanPrice"):
                    try:
                        apt = stock.analyst_price_targets
                        if apt is not None and hasattr(apt, 'get'):
                            info["targetMeanPrice"] = apt.get("mean")
                    except Exception:
                        pass

                # Try recommendations
                if not info.get("recommendationKey"):
                    try:
                        rec = stock.recommendations_summary
                        if rec is not None and not rec.empty:
                            latest = rec.iloc[0]
                            info["recommendationKey"] = str(latest.get("period", "")).lower()
                    except Exception:
                        pass

                hist = None
                if raw is not None:
                    try:
                        if ticker in raw.columns.get_level_values(0):
                            h = raw[ticker].dropna()
                            if len(h) >= 10:
                                hist = h
                    except Exception:
                        pass

                if hist is None:
                    try:
                        hist = stock.history(period=period, auto_adjust=True)
                    except Exception as e:
                        logger.warning(f"{ticker}: history failed: {e}")
                        hist = None

                logger.info(f"{ticker}: info={len(info)} keys, hist={len(hist) if hist is not None else 0} rows")

                tech_score, tech_signals = score_technical(hist)
                fund_score, fund_signals = score_fundamental(info)
                composite = round(tech_score * 0.40 + fund_score * 0.60, 1)

                short_rec, short_color = get_recommendation(tech_score)
                long_rec, long_color = get_recommendation(fund_score)
                overall_rec, overall_color = get_recommendation(composite)

                current_price = (info.get("currentPrice") or info.get("regularMarketPrice") or tech_signals.get("price"))

                sector = info.get("sector") or SECTOR_MAP.get(ticker, "N/A")
                reason = generate_reason(ticker, overall_rec, tech_score, fund_score, tech_signals, fund_signals)
                results.append({
                    "ticker": ticker,
                    "name": info.get("shortName", ticker),
                    "sector": sector,
                    "reason": reason,
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
                logger.error(f"{ticker} failed: {e}")
                results.append({
                    "ticker": ticker, "name": ticker, "sector": "N/A",
                    "reason": f"Data could not be loaded for {ticker}.",
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

        del raw
        gc.collect()

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
