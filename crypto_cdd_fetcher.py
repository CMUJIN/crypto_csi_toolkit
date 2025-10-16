
# -*- coding: utf-8 -*-
import argparse, time, sys, datetime as dt
import pandas as pd

def norm_symbol(sym: str) -> str:
    s = sym.strip().upper().replace('-', '/')
    if '/' in s:
        return s
    QUOTES = ['USDT','USDC','USD','BTC','ETH','BUSD','TUSD','FDUSD']
    for q in sorted(QUOTES, key=len, reverse=True):
        if s.endswith(q):
            base = s[:-len(q)]
            if base:
                return f"{base}/{q}"
    return s

def parse_args():
    ap = argparse.ArgumentParser(description="Fetch OHLCV via ccxt and save CSV (datetime, open, high, low, close, volume).")
    ap.add_argument("--exchange", default="binance")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--freq", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=1000)
    return ap.parse_args()

def ensure_ccxt():
    try:
        import ccxt  # noqa
        return True
    except Exception:
        print("[X] ccxt is required. pip install ccxt", file=sys.stderr)
        return False

def ms(ts: dt.datetime) -> int:
    return int(ts.timestamp() * 1000)

def iso(ts_ms: int) -> str:
    return dt.datetime.utcfromtimestamp(ts_ms/1000).strftime("%Y-%m-%d %H:%M:%S")

def main():
    args = parse_args()
    if not ensure_ccxt():
        sys.exit(2)
    import ccxt

    ex_id = args.exchange.lower()
    if not hasattr(ccxt, ex_id):
        sys.exit(f"[X] Unsupported exchange: {ex_id}")
    ex = getattr(ccxt, ex_id)({"enableRateLimit": True})

    symbol = norm_symbol(args.symbol)
    timeframe = args.freq
    try:
        ex.load_markets()
    except Exception as e:
        sys.exit(f"[X] load_markets failed: {e}")

    if symbol not in ex.symbols:
        alt = symbol.replace("/", ":")
        if alt in ex.symbols:
            symbol = alt
        else:
            sys.exit(f"[X] Symbol not found on {ex_id}: {symbol}")

    try:
        start_dt = dt.datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    except Exception:
        sys.exit("[X] --start must be YYYY-MM-DD")
    end_ms = None
    if args.end:
        try:
            end_dt = dt.datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
            end_ms = ms(end_dt)
        except Exception:
            sys.exit("[X] --end must be YYYY-MM-DD")

    since = ms(start_dt)
    limit = args.limit

    rows = []
    print(f"[*] Fetching {ex_id} {symbol} {timeframe} from {args.start} ...")
    while True:
        try:
            ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        except ccxt.RateLimitExceeded:
            time.sleep(ex.rateLimit / 1000.0 + 0.5); continue
        except Exception as e:
            print(f"[!] fetch_ohlcv error: {e}. Retrying in 1s ...", file=sys.stderr)
            time.sleep(1.0); continue

        if not ohlcv:
            break

        for t, o, h, l, c, v in ohlcv:
            if end_ms and t >= end_ms:
                break
            rows.append((iso(t), o, h, l, c, v))

        last_ts = ohlcv[-1][0]
        next_since = last_ts + 1
        if end_ms and next_since >= end_ms:
            break
        if next_since == since:
            break
        since = next_since

        if len(ohlcv) < limit:
            if not end_ms or since >= (end_ms or since):
                break

    if not rows:
        sys.exit("[!] No data fetched. Check symbol/timeframe/start.")

    df = pd.DataFrame(rows, columns=["datetime","open","high","low","close","volume"])
    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    df.to_csv(args.out, index=False)
    print(f"[\u2713] Saved -> {args.out}  rows={len(df)}")

if __name__ == "__main__":
    main()
