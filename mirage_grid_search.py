"""
Mirage Ensemble+ — 重み最適化グリッドサーチ
藤井工藝 / Fujii Kogei

【仕様】
- 対象  : 255銘柄（日本株 .T）
- テスト: 1銘柄につき「3日予測の基準日」を10点ランダムに選ぶ
          → 基準日ごとに [翌日, 2日後, 3日後] の予測 vs 実績を評価
- 評価  : MAPE（平均絶対誤差率）
- 探索  : 6モデルの重みを10%刻みでグリッドサーチ（全組み合わせ）
- 出力  : ../log/<銘柄>.csv  (銘柄ごとの詳細ログ)
          ../result/all_weights.csv   (全重み組み合わせ結果)
          ../result/best_weights.json (最優秀重み)

【実行】
  pip install yfinance numpy pandas scipy tqdm
  python src/mirage_grid_search.py
"""

import os
import sys
import json
import time
import random
import itertools
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────
# パス設定（src/ の親を基準）
# ──────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR    = os.path.join(BASE_DIR, "log")
RESULT_DIR = os.path.join(BASE_DIR, "result")
os.makedirs(LOG_DIR,    exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────
# 定数
# ──────────────────────────────────────────────────
VOL_CAL      = 1.35
DRIFT_SHRINK = 0.5
STEPS        = 3       # 予測ステップ（3日後まで）
N_SIMS       = 600     # MCシミュレーション数（速度重視）
N_BASE_DAYS  = 10      # 1銘柄あたりの基準日数
MIN_HISTORY  = 150     # 予測に必要な最低履歴日数

# ──────────────────────────────────────────────────
# 255銘柄リスト（東証主要銘柄）
# ──────────────────────────────────────────────────
SYMBOLS_RAW = [
    "1301","1332","1333","1375","1376","1377","1379","1382","1383","1384",
    "1414","1417","1419","1420","1429","1430","1435","1436","1437","1438",
    "1439","1440","1441","1442","1443","1444","1445","1446","1447","1448",
    "1449","1450","1451","1452","1453","1454","1455","1456","1457","1458",
    "1459","1460","1461","1462","1463","1464","1465","1466","1467","1468",
    "1469","1470","1471","1472","1473","1474","1475","1476","1477","1478",
    "1479","1480","1481","1482","1483","1484","1485","1486","1487","1488",
    "1489","1490","1491","1492","1493","1494","1495","1496","1497","1498",
    "1499","1500","1501","1502","1503","1504","1505","1506","1507","1508",
    "1509","1510","1511","1512","1513","1514","1515","1516","1517","1518",
    "1519","1520","1521","1522","1523","1524","1525","1526","1527","1528",
    "1529","1530","1531","1532","1533","1534","1535","1536","1537","1538",
    "1539","1540","1541","1542","1543","1544","1545","1546","1547","1548",
    "1549","1550","1551","1552","1553","1554","1555","1556","1557","1558",
    "1559","1560","1561","1562","1563","1564","1565","1566","1567","1568",
    "1569","1570","1571","1572","1573","1574","1575","1576","1577","1578",
    "1579","1580","1581","1582","1583","1584","1585","1586","1587","1588",
    "1589","1590","1591","1592","1593","1594","1595","1596","1597","1598",
    "1599","1600","1601","1602","1603","1604","1605","1606","1607","1608",
    "2502","2503","2914","3382","4063","4452","4502","4503","4506","4519",
    "4523","4543","4568","4578","4661","4689","4755","4901","4911","5201",
    "5214","5401","5411","5541","6098","6103","6146","6178","6273","6301",
    "6326","6361","6367","6371","6383","6395","6412","6417","6501","6503",
    "6504","6506","6508","6594","6645","6702","6724","6752","6758","6762",
    "6770","6857","6902","6952","6954","6971","6988","7011","7012","7013",
    "7201","7202","7203","7205","7211","7261","7267","7269","7270","7272",
    "7731","7733","7735","7741","7751","7752","7762","7832","7911","7912",
    "7974","8001","8002","8003","8004","8005","8006","8007","8008","8009",
    "8010","8011","8012","8013","8015","8016","8020","8031","8053","8058",
    "8233","8252","8267","8306","8308","8309","8316","8411","8591","8601",
    "8604","8630","8725","8750","8766","8795","8801","8802","8804","8830",
    "9001","9005","9007","9008","9009","9020","9021","9022","9064","9101",
    "9104","9107","9201","9202","9301","9432","9433","9434","9501","9502",
    "9503","9531","9532","9602","9613","9984",
]
SYMBOLS = [s + ".T" for s in SYMBOLS_RAW][:255]

# ──────────────────────────────────────────────────
# 数学ユーティリティ
# ──────────────────────────────────────────────────
def log_ret(c):
    c = np.asarray(c, dtype=float)
    return np.log(c[1:] / c[:-1])

def ewma_vol(r, lam=0.94):
    v = r[0] ** 2
    for x in r[1:]:
        v = lam * v + (1 - lam) * x ** 2
    return np.sqrt(max(v, 1e-18))

def eff_vol(r):
    recent = r[-20:] if len(r) >= 20 else r
    return VOL_CAL * max(ewma_vol(r),
                         np.std(recent, ddof=1) if len(recent) >= 2 else 0)

# ──────────────────────────────────────────────────
# M1: ブロックブートストラップMC
# ──────────────────────────────────────────────────
def m1_bootstrap(r, steps, n_sims=N_SIMS):
    if len(r) < 40:
        return None
    block  = 3
    n_blk  = (steps + block - 1) // block
    scale  = eff_vol(r) / max(np.std(r, ddof=1), 1e-9)
    m      = np.mean(r)
    cums   = np.zeros(n_sims)
    samples = []
    for s in range(n_sims):
        path = np.zeros(steps)
        k = 0
        for _ in range(n_blk):
            st = random.randint(0, len(r) - block - 1)
            for j in range(block):
                if k >= steps:
                    break
                path[k] = (r[st + j] - m) * scale + m
                k += 1
        cums[s] = path.sum()
        samples.append(path.copy())
    return {"pUp": float(np.sum(cums > 0) / n_sims), "samples": samples}

# ──────────────────────────────────────────────────
# M2: パターン類似 kNN
# ──────────────────────────────────────────────────
def m2_knn(r, steps, window=10, k=25):
    n_hist = len(r) - window - steps
    if n_hist < 60:
        return None
    tgt = r[-window:]
    tm, ts = np.mean(tgt), np.std(tgt, ddof=1) + 1e-9
    tz = (tgt - tm) / ts
    dists = []
    for i in range(n_hist):
        w  = r[i:i + window]
        wm = np.mean(w); ws = np.std(w, ddof=1) + 1e-9
        wz = (w - wm) / ws
        dists.append((float(np.sum((wz - tz) ** 2)), i))
    dists.sort(key=lambda x: x[0])
    fut  = [float(r[i + window:i + window + steps].sum()) for _, i in dists[:k]]
    n_up = sum(1 for v in fut if v > 0)
    return {"pUp": n_up / len(fut), "nUp": n_up, "k": len(fut)}

# ──────────────────────────────────────────────────
# M3: AR(1) 自己相関
# ──────────────────────────────────────────────────
def m3_ar1(r, steps):
    if len(r) < 30:
        return None
    a, b   = r[:-1], r[1:]
    ma, mb = np.mean(a), np.mean(b)
    cov    = float(np.sum((a - ma) * (b - mb)))
    va     = float(np.sum((a - ma) ** 2))
    vb     = float(np.sum((b - mb) ** 2))
    rho    = cov / np.sqrt(max(va * vb, 1e-18))
    sig    = max(float(np.std(r, ddof=1)), 1e-9)
    e_cum  = sum(float(r[-1]) * (rho ** i) for i in range(1, steps + 1))
    p_up   = float(norm.cdf(e_cum / (sig * np.sqrt(steps))))
    return {"pUp": p_up, "rho": rho}

# ──────────────────────────────────────────────────
# M4: テクニカルスコア
# ──────────────────────────────────────────────────
def _rsi(c, period=14):
    delta  = np.diff(c)
    gain   = np.where(delta > 0, delta, 0.0)
    loss   = np.where(delta < 0, -delta, 0.0)
    avg_g  = np.mean(gain[:period])
    avg_l  = np.mean(loss[:period])
    for i in range(period, len(gain)):
        avg_g = (avg_g * (period - 1) + gain[i]) / period
        avg_l = (avg_l * (period - 1) + loss[i]) / period
    return 100 - 100 / (1 + avg_g / max(avg_l, 1e-9))

def _ema(arr, span):
    k   = 2 / (span + 1)
    out = np.zeros(len(arr))
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out

def _macd(c):
    e12  = _ema(c, 12); e26 = _ema(c, 26)
    line = e12 - e26
    sig  = _ema(line, 9)
    return line, sig, line - sig

def _bollinger(c, n=20):
    if len(c) < n:
        return None
    mid = np.array([np.mean(c[i - n:i]) for i in range(n, len(c) + 1)])
    sd  = np.array([np.std(c[i - n:i], ddof=1) for i in range(n, len(c) + 1)])
    return {"mid": mid, "sd": sd}

def m4_technical(c, bbn=20):
    score = 0
    rs    = _rsi(c)
    score += 2 if rs < 30 else (-2 if rs > 70 else 0)
    line, sig, hist = _macd(c)
    n = len(c) - 1
    if line[n] > sig[n] and hist[n] > hist[n - 1]:
        score += 2
    elif line[n] < sig[n] and hist[n] < hist[n - 1]:
        score -= 2
    if len(c) > 25:
        score += 1 if np.mean(c[-5:]) > np.mean(c[-25:]) else -1
    bb = _bollinger(c, bbn)
    if bb is not None:
        bb_dev = (c[-1] - bb["mid"][-1]) / max(bb["sd"][-1], 1e-9)
        if bb_dev <= -2:
            score += 2 if bb_dev <= -3 else 1
        elif bb_dev >= 2:
            score -= 2 if bb_dev >= 3 else 1
    score += 1 if c[-1] > c[-2] else -1
    return {"pUp": float(1 / (1 + np.exp(-0.35 * score))), "score": score}

# ──────────────────────────────────────────────────
# M5: ニュース（バックテスト時は中立固定）
# ──────────────────────────────────────────────────
def m5_news_dummy():
    return {"pUp": 0.5, "pos": 0, "neg": 0}

# ──────────────────────────────────────────────────
# M6: マクロ相関β（バックテスト時はFX変動=0→中立）
# ──────────────────────────────────────────────────
def m6_macro_backtest(symbol, stock_r, nikkei_r, fx_r):
    if not symbol.endswith(".T"):
        return None
    n = min(len(stock_r), len(nikkei_r), len(fx_r))
    if n < 120:
        return None
    rs, rn, rf = stock_r[-n:], nikkei_r[-n:], fx_r[-n:]
    def corr(x, y):
        mx, my = np.mean(x), np.mean(y)
        num    = np.sum((x - mx) * (y - my))
        den    = np.sqrt(max(np.sum((x - mx)**2) * np.sum((y - my)**2), 1e-18))
        return num / den
    return {"pUp": 0.5, "corrF": float(corr(rs, rf))}

# ──────────────────────────────────────────────────
# アンサンブル予測（3日分の中央値を返す）
# ──────────────────────────────────────────────────
def ensemble_predict(close_hist, steps, weights, m6_res=None):
    r  = log_ret(close_hist)
    s0 = float(close_hist[-1])

    m1 = m1_bootstrap(r, steps)
    m2 = m2_knn(r, steps)
    m3 = m3_ar1(r, steps)
    m4 = m4_technical(close_hist)
    m5 = m5_news_dummy()
    m6 = m6_res

    model_w = [
        (m1, weights[0]),
        (m2, weights[1]),
        (m3, weights[2]),
        (m4, weights[3]),
        (m5, weights[4]),
        (m6, weights[5]),
    ]
    votes, ws = [], []
    for m, w in model_w:
        if m is not None and w > 0:
            votes.append(m["pUp"])
            ws.append(w)

    if not votes:
        return None

    w_sum  = sum(ws)
    w_norm = [w / w_sum for w in ws]
    p_ens  = sum(v * w for v, w in zip(votes, w_norm))

    p_fan = 0.5 + DRIFT_SHRINK * (p_ens - 0.5)
    sigma = eff_vol(r)
    d     = float(norm.ppf(np.clip(p_fan, 1e-6, 1 - 1e-6))) * sigma / np.sqrt(steps)

    samples = m1["samples"] if m1 else [
        np.random.randn(steps) * sigma for _ in range(N_SIMS)
    ]
    gm = np.mean([v for p in samples for v in p])

    paths = []
    for p in samples:
        path, cum = [], 0.0
        for j in range(steps):
            cum += float(p[j]) - gm + d
            path.append(s0 * np.exp(cum))
        paths.append(path)

    medians = [float(np.median([paths[s][j] for s in range(len(paths))])) for j in range(steps)]
    return medians

# ──────────────────────────────────────────────────
# MAPE
# ──────────────────────────────────────────────────
def calc_mape(predicted, actual):
    errs = [abs(p - a) / abs(a) for p, a in zip(predicted, actual) if a != 0]
    return float(np.mean(errs) * 100) if errs else None

# ──────────────────────────────────────────────────
# グリッド生成（10%刻み、6モデル、合計1.0）
# ──────────────────────────────────────────────────
def generate_grid(step=0.1, n=6):
    vals  = [round(i * step, 2) for i in range(int(1 / step) + 1)]
    seen  = set()
    grid  = []
    for combo in itertools.combinations_with_replacement(vals, n):
        if abs(sum(combo) - 1.0) < 1e-6:
            for perm in set(itertools.permutations(combo)):
                if perm not in seen:
                    seen.add(perm)
                    grid.append(perm)
    return grid

# ──────────────────────────────────────────────────
# データキャッシュ付き取得
# ──────────────────────────────────────────────────
_cache = {}

def fetch(symbol, period="2y"):
    if symbol in _cache:
        return _cache[symbol]
    try:
        df = yf.download(symbol, period=period, progress=False, auto_adjust=True)
        if df.empty or len(df) < MIN_HISTORY:
            _cache[symbol] = None
        else:
            _cache[symbol] = df["Close"].values.astype(float)
    except Exception:
        _cache[symbol] = None
    return _cache[symbol]

# ──────────────────────────────────────────────────
# 1銘柄のバックテスト（全重みパターン）
# ──────────────────────────────────────────────────
def backtest_one_symbol(symbol, grid, nikkei_r, fx_r):
    """
    Returns
    -------
    log_rows  : list of dict  (銘柄ログ CSV 用)
    mape_by_w : list of float (grid と同順の MAPE 合計)
    n_valid   : int  (有効テスト数)
    """
    close = fetch(symbol)
    if close is None or len(close) < MIN_HISTORY + STEPS:
        return None, None, 0

    # 基準日候補：履歴がMIN_HISTORY以上あり、かつ3日後の実績が存在する範囲
    max_idx = len(close) - STEPS - 1
    min_idx = MIN_HISTORY
    if max_idx <= min_idx:
        return None, None, 0

    pool = list(range(min_idx, max_idx + 1))
    base_days = random.sample(pool, min(N_BASE_DAYS, len(pool)))
    base_days.sort()

    # M6データ準備（銘柄単位で一度だけ計算）
    stock_r = log_ret(close)
    m6_cache = {}
    def get_m6(idx):
        if idx not in m6_cache:
            m6_cache[idx] = m6_macro_backtest(
                symbol, stock_r[:idx-1], nikkei_r, fx_r)
        return m6_cache[idx]

    # 各基準日 × 各重みパターン の MAPE を収集
    mape_sum = [0.0] * len(grid)
    mape_cnt = [0]   * len(grid)
    log_rows  = []

    # 基準日ごとにまず実績値を確定
    base_data = []
    for idx in base_days:
        hist   = close[:idx]
        actual = close[idx:idx + STEPS].tolist()
        base_data.append((idx, hist, actual))

    # 重みループ（全パターン）
    for wi, weights in enumerate(grid):
        for idx, hist, actual in base_data:
            m6_res = get_m6(idx)
            try:
                pred = ensemble_predict(hist, STEPS, weights, m6_res)
                if pred is None:
                    continue
                mape = calc_mape(pred, actual)
                if mape is None:
                    continue
                mape_sum[wi] += mape
                mape_cnt[wi] += 1
            except Exception:
                continue

    # ログは現在の重み（元の設定値）での結果のみ記録
    current_w = (0.22, 0.27, 0.09, 0.18, 0.13, 0.11)
    for idx, hist, actual in base_data:
        m6_res = get_m6(idx)
        try:
            pred = ensemble_predict(hist, STEPS, current_w, m6_res)
            if pred is None:
                continue
            for day in range(STEPS):
                log_rows.append({
                    "symbol":    symbol,
                    "base_date": int(idx),
                    "day":       day + 1,
                    "actual":    round(float(actual[day]), 2),
                    "predicted": round(float(pred[day]),   2),
                    "error_pct": round(abs(pred[day] - actual[day]) / max(abs(actual[day]), 1e-9) * 100, 4),
                    "accuracy_pct": round(100 - abs(pred[day] - actual[day]) / max(abs(actual[day]), 1e-9) * 100, 4),
                })
        except Exception:
            continue

    n_valid = mape_cnt[0] if mape_cnt[0] > 0 else 0
    avg_mapes = [
        mape_sum[wi] / mape_cnt[wi] if mape_cnt[wi] > 0 else None
        for wi in range(len(grid))
    ]
    return log_rows, avg_mapes, n_valid

# ──────────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────────
def main():
    random.seed(42)
    np.random.seed(42)

    print("=" * 65)
    print("  Mirage Ensemble+  重み最適化グリッドサーチ")
    print("  藤井工藝 / Fujii Kogei")
    print("=" * 65)
    print(f"  開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  対象銘柄: {len(SYMBOLS)} 銘柄")
    print(f"  基準日数: {N_BASE_DAYS} 日/銘柄")
    print(f"  予測ステップ: {STEPS} 日")
    print()

    # ── STEP 1: グリッド生成 ──
    print("[STEP 1/4] 重みグリッド生成中...")
    grid = generate_grid(step=0.1, n=6)
    print(f"  → {len(grid):,} 通りの重み組み合わせ\n")

    # ── STEP 2: データ事前取得 ──
    print("[STEP 2/4] データ事前取得中...")
    nikkei_raw = fetch("^N225")
    fx_raw     = fetch("JPY=X")
    nikkei_r   = log_ret(nikkei_raw) if nikkei_raw is not None else np.array([])
    fx_r       = log_ret(fx_raw)     if fx_raw     is not None else np.array([])

    valid_syms = []
    with tqdm(SYMBOLS, desc="  銘柄データ取得", ncols=70) as bar:
        for sym in bar:
            d = fetch(sym)
            if d is not None and len(d) >= MIN_HISTORY + STEPS:
                valid_syms.append(sym)
            time.sleep(0.03)
    print(f"\n  → 有効銘柄: {len(valid_syms)} / {len(SYMBOLS)}\n")

    # ── STEP 3: バックテスト（全銘柄 × 全重み）──
    print("[STEP 3/4] バックテスト実行中...")
    print(f"  総評価数（上限）: {len(valid_syms)} 銘柄 × {N_BASE_DAYS} 基準日 × {len(grid):,} 重みパターン")
    print()

    # 重みパターンごとの MAPE 累積
    mape_total = [0.0] * len(grid)
    mape_count = [0]   * len(grid)
    all_log_rows = []

    with tqdm(valid_syms, desc="  銘柄処理", ncols=70, unit="銘柄") as bar:
        for sym in bar:
            bar.set_postfix_str(sym)
            log_rows, avg_mapes, n_valid = backtest_one_symbol(
                sym, grid, nikkei_r, fx_r)

            if avg_mapes is None:
                continue

            # 銘柄ログ保存
            if log_rows:
                df_log = pd.DataFrame(log_rows)
                log_path = os.path.join(LOG_DIR, f"{sym.replace('.T','')}.csv")
                df_log.to_csv(log_path, index=False, encoding="utf-8-sig")
                all_log_rows.extend(log_rows)

            # 重み集計
            for wi, mape in enumerate(avg_mapes):
                if mape is not None:
                    mape_total[wi] += mape
                    mape_count[wi] += 1

    # ── STEP 4: 集計・出力 ──
    print("\n[STEP 4/4] 集計・結果出力中...")

    results = []
    keys = ["m1","m2","m3","m4","m5","m6"]
    for wi, weights in enumerate(grid):
        if mape_count[wi] == 0:
            continue
        avg = mape_total[wi] / mape_count[wi]
        results.append({
            **dict(zip(keys, weights)),
            "mape":     round(avg, 4),
            "accuracy": round(100 - avg, 4),
            "n_eval":   mape_count[wi],
        })

    results.sort(key=lambda x: x["mape"])

    # all_weights.csv
    df_all = pd.DataFrame(results)
    df_all.to_csv(os.path.join(RESULT_DIR, "all_weights.csv"),
                  index=False, encoding="utf-8-sig")

    # summary.csv（全銘柄ログ結合）
    if all_log_rows:
        df_summary = pd.DataFrame(all_log_rows)
        df_summary.to_csv(os.path.join(RESULT_DIR, "summary_log.csv"),
                          index=False, encoding="utf-8-sig")

    # best_weights.json
    best = results[0]
    current_w_dict = {"m1":0.22,"m2":0.27,"m3":0.09,"m4":0.18,"m5":0.13,"m6":0.11}
    # 現在の重みのMAPEを検索
    current_mape = next(
        (r["mape"] for r in results
         if all(abs(r[k] - current_w_dict[k]) < 0.01 for k in keys)),
        None
    )

    output = {
        "generated_at":    datetime.now().isoformat(),
        "n_symbols":       len(valid_syms),
        "n_base_days":     N_BASE_DAYS,
        "steps":           STEPS,
        "best_weights":    {k: best[k] for k in keys},
        "best_mape":       best["mape"],
        "best_accuracy":   best["accuracy"],
        "top10": results[:10],
        "current_weights": current_w_dict,
        "current_mape":    current_mape,
        "improvement":     round((current_mape - best["mape"]), 4) if current_mape else None,
    }
    with open(os.path.join(RESULT_DIR, "best_weights.json"), "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # ── 結果表示 ──
    print()
    print("=" * 65)
    print("  【最適重み TOP 10】")
    print("=" * 65)
    print(f"  {'順位':>3}  {'M1':>5} {'M2':>5} {'M3':>5} {'M4':>5} {'M5':>5} {'M6':>5}"
          f"  {'MAPE%':>7}  {'精度%':>7}")
    print("  " + "-" * 60)
    for i, r in enumerate(results[:10]):
        print(f"  {i+1:>3}  "
              f"{r['m1']:>5.2f} {r['m2']:>5.2f} {r['m3']:>5.2f} "
              f"{r['m4']:>5.2f} {r['m5']:>5.2f} {r['m6']:>5.2f}  "
              f"{r['mape']:>7.3f}  {r['accuracy']:>7.3f}")

    print()
    print("  【最優秀重み】")
    for k in keys:
        print(f"    {k.upper()}: {best[k]:.2f}")
    print(f"    MAPE    : {best['mape']:.3f}%")
    print(f"    精度    : {best['accuracy']:.3f}%")

    if current_mape:
        imp = current_mape - best["mape"]
        print()
        print(f"  【現在の重み (元設定) の MAPE】: {current_mape:.3f}%")
        print(f"  【改善幅】: {imp:+.3f}% ({'改善' if imp > 0 else '変化なし/悪化'})")

    print()
    print(f"  出力先:")
    print(f"    ../log/<銘柄>.csv          — 銘柄別詳細ログ")
    print(f"    ../result/summary_log.csv  — 全銘柄統合ログ")
    print(f"    ../result/all_weights.csv  — 全重み結果")
    print(f"    ../result/best_weights.json— 最適重み")
    print()
    print(f"  終了: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

if __name__ == "__main__":
    main()
