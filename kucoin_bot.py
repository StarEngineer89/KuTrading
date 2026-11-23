#!/usr/bin/env python3
"""
KuCoin spot trend-following bot (Classic account, REST).

Strategy: long-only EMA(fast/slow) crossover on closed candles.
  - Enter when fast EMA crosses above slow EMA.
  - Exit when fast crosses below slow, or price hits ATR stop.
  - Size so that a stop-out loses ~RISK_PCT of equity.

Modes:
  python kucoin_bot.py backtest   # public data only, no keys needed
  python kucoin_bot.py paper      # live prices, simulated fills (default for run)
  python kucoin_bot.py live       # REAL orders; needs env vars + DRY_RUN=0

Env vars for live: KC_KEY, KC_SECRET, KC_PASSPHRASE (key version 2 assumed),
read from the environment or from a .env file next to this script.
Use an API key with Spot trade permission ONLY (no withdrawal), IP-restricted.
Live mode assumes the bot owns all of the base coin in the account: it uses the
coin balance to tell whether the exchange stop-loss has filled.
"""
import os, sys, time, hmac, hashlib, base64, json, uuid
from decimal import Decimal, ROUND_DOWN
from functools import lru_cache
import requests


def load_dotenv(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")):
    # Minimal .env reader: KEY=value lines; real environment variables take precedence.
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_dotenv()

BASE ="https://api.kucoin.com"
SYMBOL = "SOL-USDT"
TF = "4hour"
FAST, SLOW, ATR_N = 30, 80, 14
ATR_MULT = 2.0
RISK_PCT = 0.02        # risk 2% of equity per trade
MAX_ALLOC = 0.5        # never put more than 50% of equity in one position
FEE = 0.001            # 0.1% per side (check your actual tier)
SLIPPAGE = 0.0005
POLL_SEC = 60
STATE_FILE = {True: "state.json", False: "state_live.json"}   # keyed by paper; never share


# ---------- API helpers ----------
def pub(path, params=None):
    r = requests.get(BASE + path, params=params, timeout=10)
    r.raise_for_status()
    j = r.json()
    if j.get("code") != "200000":
        raise RuntimeError(j)
    return j["data"]


def signed(method, endpoint, body=None):
    key, secret, phrase = (os.environ[k] for k in ("KC_KEY", "KC_SECRET", "KC_PASSPHRASE"))
    ts = str(int(time.time() * 1000))
    payload = json.dumps(body, separators=(",", ":")) if body else ""
    sig = base64.b64encode(hmac.new(secret.encode(), (ts + method + endpoint + payload).encode(),
                                    hashlib.sha256).digest()).decode()
    pp = base64.b64encode(hmac.new(secret.encode(), phrase.encode(), hashlib.sha256).digest()).decode()
    headers = {"KC-API-KEY": key, "KC-API-SIGN": sig, "KC-API-TIMESTAMP": ts,
               "KC-API-PASSPHRASE": pp, "KC-API-KEY-VERSION": "2",
               "Content-Type": "application/json"}
    r = requests.request(method, BASE + endpoint, headers=headers, data=payload or None, timeout=10)
    j = r.json()
    if j.get("code") != "200000":
        raise RuntimeError(j)
    return j["data"]


TF_SEC = {"1min": 60, "5min": 300, "15min": 900, "30min": 1800, "1hour": 3600,
          "4hour": 14400, "1day": 86400}


def get_candles(symbol, tf, n=300):
    # rows: [time, open, close, high, low, volume, turnover], newest first.
    # Without startAt/endAt KuCoin returns only ~100 rows; page back 1500 at a time.
    step, end, rows = TF_SEC[tf], int(time.time()), []
    while len(rows) < n:
        page = pub("/api/v1/market/candles", {"symbol": symbol, "type": tf,
                                               "startAt": end - step * 1500, "endAt": end})
        if not page:
            break
        rows += page
        end = int(page[-1][0]) - 1
    rows = rows[:n][::-1]
    return [dict(t=int(r[0]), o=float(r[1]), c=float(r[2]), h=float(r[3]), l=float(r[4])) for r in rows]


def last_price(symbol):
    return float(pub("/api/v1/market/orderbook/level1", {"symbol": symbol})["price"])


# ---------- indicators ----------
def ema(vals, n):
    k, out = 2 / (n + 1), [vals[0]]
    for x in vals[1:]:
        out.append(x * k + out[-1] * (1 - k))
    return out


def atr(bars, n):
    trs = [max(b["h"] - b["l"], abs(b["h"] - p["c"]), abs(b["l"] - p["c"]))
           for p, b in zip(bars, bars[1:])]
    return sum(trs[-n:]) / n


def signal(bars):
    """Return 'buy', 'sell' or None based on the last two CLOSED bars."""
    c = [b["c"] for b in bars]
    f, s = ema(c, FAST), ema(c, SLOW)
    if f[-2] <= s[-2] and f[-1] > s[-1]:
        return "buy"
    if f[-2] >= s[-2] and f[-1] < s[-1]:
        return "sell"
    return None


def position_size(equity, price, stop):
    risk_qty = equity * RISK_PCT / max(price - stop, 1e-9)
    return min(risk_qty, equity * MAX_ALLOC / price)


# ---------- backtest ----------
def backtest():
    bars = get_candles(SYMBOL, TF, 1500)
    equity, start = 1000.0, 1000.0
    pos, trades, peak, mdd = None, 0, 1000.0, 0.0
    for i in range(SLOW + 2, len(bars) - 1):
        hist, nxt = bars[: i + 1], bars[i + 1]
        if pos:
            if nxt["l"] <= pos["stop"]:                         # stop hit
                px = pos["stop"] * (1 - SLIPPAGE)
                equity += pos["qty"] * px * (1 - FEE); pos = None; trades += 1
            elif signal(hist) == "sell":
                px = nxt["o"] * (1 - SLIPPAGE)
                equity += pos["qty"] * px * (1 - FEE); pos = None; trades += 1
        elif signal(hist) == "buy":
            px = nxt["o"] * (1 + SLIPPAGE)
            stop = px - ATR_MULT * atr(hist, ATR_N)
            qty = position_size(equity, px, stop)
            cost = qty * px * (1 + FEE)
            equity -= cost
            pos = dict(qty=qty, stop=stop)
        mark = equity + (pos["qty"] * nxt["c"] if pos else 0)
        peak = max(peak, mark); mdd = max(mdd, (peak - mark) / peak)
    final = equity + (pos["qty"] * bars[-1]["c"] if pos else 0)
    bh = bars[-1]["c"] / bars[SLOW + 2]["o"] - 1
    print(f"{SYMBOL} {TF}: {len(bars)} bars | trades {trades}")
    print(f"strategy return {final / start - 1:+.1%} | buy&hold {bh:+.1%} | max drawdown {mdd:.1%}")


# ---------- live / paper ----------
def load(paper):
    if os.path.exists(STATE_FILE[paper]):
        return json.load(open(STATE_FILE[paper]))
    return dict(pos=None, last_bar=0, paper_usdt=1000.0)


def save(st, paper):
    json.dump(st, open(STATE_FILE[paper], "w"))


BASE_CCY = SYMBOL.split("-")[0]


def balance(currency, field="available"):
    # Classic (non-HF) spot account: the one /api/v1/orders and /api/v1/stop-order trade from.
    # field "balance" also counts coins held by open orders, e.g. our stop-loss.
    rows = signed("GET", f"/api/v1/accounts?currency={currency}&type=trade")
    return float(rows[0][field]) if rows else 0.0


@lru_cache(maxsize=None)
def symbol_info():
    return pub(f"/api/v2/symbols/{SYMBOL}")


def to_inc(x, inc):
    """Round x down to a multiple of the exchange increment string `inc`, formatted for the API."""
    return format((Decimal(str(x)) / Decimal(inc)).to_integral_value(ROUND_DOWN) * Decimal(inc), "f")


def holding():
    """Whether the account still holds a sellable amount of the base coin."""
    return balance(BASE_CCY, "balance") >= float(symbol_info()["baseMinSize"])


def market_order(side, qty, price, paper):
    """Place a market order; return the base quantity actually traded."""
    if paper:
        return qty
    body = {"clientOid": uuid.uuid4().hex, "symbol": SYMBOL, "type": "market", "side": side}
    if side == "buy":
        body["funds"] = f"{qty * price:.2f}"
    else:
        # Fees can leave us holding slightly less than recorded; never sell more than we have.
        size = to_inc(min(qty, balance(BASE_CCY)), symbol_info()["baseIncrement"])
        if float(size) == 0:
            return 0.0
        body["size"] = size
    before = balance(BASE_CCY, "balance")
    res = signed("POST", "/api/v1/orders", body)
    print("ORDER", res)
    # The order is in: never raise from here on, or the caller loses track of the coins.
    for _ in range(5):
        time.sleep(1)                                   # let the market order settle
        try:
            fill = signed("GET", f"/api/v1/orders/{res['orderId']}")
            if not fill["isActive"]:
                return float(fill["dealSize"])
        except Exception as e:
            print("fill check failed:", e)
    try:
        return abs(balance(BASE_CCY, "balance") - before)
    except Exception as e:
        print("balance check failed, assuming a full fill:", e)
        return qty


def place_stop(stop):
    """Cover any free base coin with a market stop-loss sell on the exchange."""
    info = symbol_info()
    size = to_inc(balance(BASE_CCY), info["baseIncrement"])
    if float(size) < float(info["baseMinSize"]):
        return                                          # nothing free: already covered
    # A sell with stopPrice below the market triggers on the way down; there is no loss/entry flag.
    body = {"clientOid": uuid.uuid4().hex, "symbol": SYMBOL, "side": "sell", "type": "market",
            "stopPrice": to_inc(stop, info["priceIncrement"]), "size": size, "tradeType": "TRADE"}
    print("STOP ORDER", signed("POST", "/api/v1/stop-order", body))


def cancel_stops():
    try:
        signed("DELETE", f"/api/v1/stop-order/cancel?symbol={SYMBOL}")
    except Exception as e:
        print("stop cancel failed:", e)
    time.sleep(1)                                       # let the held coins free up


def exit_position(st, px, paper, why):
    pos = st["pos"]
    pos["exiting"] = why                                # retried every poll until done
    if not paper:
        cancel_stops()
        market_order("sell", pos["qty"], px, paper)
        if holding():
            print(f"{why} exit incomplete; retrying next poll")
            return
    st["paper_usdt"] += pos["qty"] * px * (1 - FEE); st["pos"] = None
    print(f"{why} exit @ {px}")


def run(paper=True):
    st = load(paper)
    print(f"{'PAPER' if paper else 'LIVE'} mode | {SYMBOL} {TF}")
    while True:
        try:
            px = last_price(SYMBOL)
            bars = get_candles(SYMBOL, TF, 300)[:-1]          # drop the still-forming candle
            pos = st["pos"]

            if pos and not paper and not pos.get("exiting") and not holding():
                st["pos"] = None
                print("exchange stop-loss filled; position closed")
            elif pos and (pos.get("exiting") or px <= pos["stop"]):
                exit_position(st, px, paper, pos.get("exiting") or "STOP")
            elif pos and not paper:
                place_stop(pos["stop"])                         # no-op while already covered

            if bars[-1]["t"] != st["last_bar"]:                 # new closed candle
                st["last_bar"] = bars[-1]["t"]
                sig = signal(bars)
                pos = st["pos"]
                if sig == "buy" and not pos:
                    stop = px - ATR_MULT * atr(bars, ATR_N)
                    equity = st["paper_usdt"] if paper else balance("USDT")
                    qty = position_size(equity, px, stop)
                    qty = market_order("buy", qty, px, paper)
                    st["paper_usdt"] -= qty * px * (1 + FEE)
                    st["pos"] = dict(qty=qty, entry=px, stop=stop)
                    save(st, paper)                             # record the coins before anything can fail
                    print(f"BUY {qty:.6f} @ {px} stop {stop:.2f}")
                    if not paper:
                        place_stop(stop)
                elif sig == "sell" and pos and not pos.get("exiting"):
                    exit_position(st, px, paper, "SELL")
        except Exception as e:
            print("error:", e)                                  # never crash-loop silently
        finally:
            save(st, paper)
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "paper"
    if mode == "backtest":
        backtest()
    elif mode == "live":
        if os.environ.get("DRY_RUN", "1") != "0":
            sys.exit("Refusing live mode: set DRY_RUN=0 to confirm real orders.")
        run(paper=False)
    else:
        run(paper=True)