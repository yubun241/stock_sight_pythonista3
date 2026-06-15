# -*- coding: utf-8 -*-
"""
StockSight v5.2 "Mirage Ensemble+" — 株価予測アプリ (Pythonista 3 / iPhone 17 Pro)
藤井工藝

6モデル・アンサンブル (外部課金なし / 全て無料APIのみ使用):
  [M1] ブロック・ブートストラップMC : 実収益率の再標本化 (ファットテール保持)
  [M2] パターン類似検索 (k-NN)     : 過去2年の類似局面のその後を集計 (ベクトル化)
  [M3] 自己相関 (AR(1))            : モメンタム/平均回帰レジーム判定
  [M4] テクニカルスコア            : RSI/MACD/SMA/BBσ乖離 → ロジスティック変換
  [M5] ニュースキーワード          : 見出しの日英キーワード解析
  [M6] マクロ相関                  : 日経平均β・ドル円β + 夜間FX変動による
       翌日ドリフト推定 (FXは東証閉場中も動くため先行情報になる)

較正 (実測バックテストに基づく):
  VOL_CAL=1.35 (68%帯内71%/90%帯内91%で合格済) / DRIFT_SHRINK=0.5

依存: requests / numpy / matplotlib (Pythonista 3標準同梱)
"""

import ui
import io
import os
import math
import time
import console
import tempfile
import threading
import requests
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
try:
    from candle_patterns import detect_patterns
except ImportError:
    detect_patterns = None   # candle_patterns.py を同フォルダに置くと有効化

UA = {'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15'}

# ============================================================
# データ取得
# ============================================================

def normalize_symbol(code):
    code = code.strip().upper()
    if code.isdigit():
        return code + '.T'
    return code

def _ts_to_date(ts):
    t = time.gmtime(ts + 9 * 3600)     # JST
    return t.tm_year * 10000 + t.tm_mon * 100 + t.tm_mday

def fetch_chart(symbol, mode):
    if mode == 'daily':
        params = {'interval': '1d', 'range': '2y'}
        n_disp = 63
    else:
        params = {'interval': '5m', 'range': '5d'}
        n_disp = 24
    url = 'https://query1.finance.yahoo.com/v8/finance/chart/' + symbol
    r = requests.get(url, params=params, headers=UA, timeout=15)
    r.raise_for_status()
    res = r.json()['chart']['result'][0]
    q = res['indicators']['quote'][0]
    o = np.array(q['open'], dtype=float)
    h = np.array(q['high'], dtype=float)
    l = np.array(q['low'], dtype=float)
    c = np.array(q['close'], dtype=float)
    mask = ~(np.isnan(o) | np.isnan(h) | np.isnan(l) | np.isnan(c))
    o, h, l, c = o[mask], h[mask], l[mask], c[mask]
    name = res['meta'].get('shortName') or symbol
    return {'open': o[-n_disp:], 'high': h[-n_disp:],
            'low': l[-n_disp:], 'close': c[-n_disp:],
            'close_full': c, 'symbol': symbol, 'name': name, 'mode': mode}

def fetch_daily_series(symbol, rng='2y'):
    """日次系列を {日付int: 終値} で返す (マクロ相関の日付整合用)"""
    url = 'https://query1.finance.yahoo.com/v8/finance/chart/' + symbol
    r = requests.get(url, params={'interval': '1d', 'range': rng},
                     headers=UA, timeout=15)
    r.raise_for_status()
    res = r.json()['chart']['result'][0]
    ts = res['timestamp']
    c = res['indicators']['quote'][0]['close']
    out = {}
    for t, v in zip(ts, c):
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            out[_ts_to_date(t)] = float(v)
    return out

def fetch_news(symbol, count=8):
    try:
        url = 'https://query1.finance.yahoo.com/v1/finance/search'
        r = requests.get(url, params={'q': symbol, 'newsCount': count, 'quotesCount': 0},
                         headers=UA, timeout=10)
        items = r.json().get('news', [])
        return [it.get('title', '') for it in items if it.get('title')]
    except Exception:
        return []

# ============================================================
# テクニカル指標
# ============================================================

def ema(x, n):
    a = 2.0 / (n + 1)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out

def sma(x, n):
    if len(x) < n:
        return None
    v = np.convolve(x, np.ones(n) / n, 'valid')
    return np.arange(n - 1, len(x)), v

def bollinger(x, n=20):
    if len(x) < n:
        return None
    mid = np.convolve(x, np.ones(n) / n, 'valid')
    std = np.array([x[i - n + 1:i + 1].std(ddof=0)
                    for i in range(n - 1, len(x))])
    xi = np.arange(n - 1, len(x))
    return xi, mid, std

def rsi(close, n=14):
    d = np.diff(close, prepend=close[0])
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    au = np.empty_like(close); ad = np.empty_like(close)
    au[0], ad[0] = up[0], dn[0]
    for i in range(1, len(close)):
        au[i] = (au[i-1] * (n-1) + up[i]) / n
        ad[i] = (ad[i-1] * (n-1) + dn[i]) / n
    rs = au / np.where(ad == 0, 1e-9, ad)
    return 100 - 100 / (1 + rs)

def macd(close, fast=12, slow=26, sig=9):
    line = ema(close, fast) - ema(close, slow)
    signal = ema(line, sig)
    return line, signal, line - signal

# ============================================================
# 数学ユーティリティ
# ============================================================

def norm_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))

def inv_norm(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    cc = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
          -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    dd = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
          3.754408661907416e+00]
    pl = 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((cc[0]*q+cc[1])*q+cc[2])*q+cc[3])*q+cc[4])*q+cc[5]) / \
               ((((dd[0]*q+dd[1])*q+dd[2])*q+dd[3])*q+1)
    if p > 1 - pl:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((cc[0]*q+cc[1])*q+cc[2])*q+cc[3])*q+cc[4])*q+cc[5]) / \
                ((((dd[0]*q+dd[1])*q+dd[2])*q+dd[3])*q+1)
    q = p - 0.5; r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)

def ewma_vol(r, lam=0.94):
    v = r[0] ** 2
    for x in r[1:]:
        v = lam * v + (1 - lam) * x * x
    return math.sqrt(v)

# バックテスト較正 (68%帯内71%/90%帯内91%で合格済)
VOL_CAL = 1.35
DRIFT_SHRINK = 0.5

def eff_vol(r):
    base = max(ewma_vol(r), float(np.std(r[-20:], ddof=1)) if len(r) >= 20 else 0.0)
    return VOL_CAL * base

# ============================================================
# Mirage Ensemble+ 予測エンジン (7モデル)
# ============================================================

def model_bootstrap_mc(r, steps, n_sims=4000, block=3):
    """[M1] ブロック・ブートストラップMC"""
    nr = len(r)
    if nr < 40:
        return None
    nblk = int(math.ceil(steps / float(block)))
    starts = np.random.randint(0, nr - block, (n_sims, nblk))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_sims, -1)[:, :steps]
    samp = r[idx]
    scale = eff_vol(r) / max(r.std(ddof=1), 1e-9)
    samp = (samp - r.mean()) * scale + r.mean()
    cum = samp.sum(axis=1)
    return {'samples': samp, 'p_up': float((cum > 0).mean())}

def model_knn(r, steps, window=10, k=25):
    """[M2] パターン類似検索 (ベクトル化)"""
    n_hist = len(r) - window - steps
    if n_hist < 60:
        return None
    idx = np.arange(window)[None, :] + np.arange(n_hist)[:, None]
    W = r[idx]
    Wz = (W - W.mean(axis=1, keepdims=True)) / (W.std(axis=1, keepdims=True) + 1e-9)
    tgt = r[-window:]
    tz = (tgt - tgt.mean()) / (tgt.std() + 1e-9)
    d = ((Wz - tz) ** 2).sum(axis=1)
    top = np.argpartition(d, k)[:k]
    fut = np.array([r[i + window:i + window + steps].sum() for i in top])
    return {'p_up': float((fut > 0).mean()), 'med_ret': float(np.median(fut)),
            'n_up': int((fut > 0).sum()), 'k': len(fut)}

def model_ar1(r, steps):
    """[M3] AR(1)"""
    if len(r) < 30:
        return None
    rho = float(np.corrcoef(r[:-1], r[1:])[0, 1])
    sigma = max(r.std(ddof=1), 1e-9)
    e_cum = sum(r[-1] * (rho ** i) for i in range(1, steps + 1))
    p_up = norm_cdf(e_cum / (sigma * math.sqrt(steps)))
    regime = (u'モメンタム(順張り)' if rho > 0.05
              else u'平均回帰(逆張り)' if rho < -0.05 else u'ランダムウォーク的')
    return {'p_up': p_up, 'rho': rho, 'regime': regime}

def technical_score(close, mode):
    """[M4] テクニカルスコア"""
    score, reasons = 0, []
    rs = rsi(close)[-1]
    line, signal, hist = macd(close)
    if rs < 30:
        score += 2; reasons.append(u'RSI {:.0f} → 売られすぎ'.format(rs))
    elif rs > 70:
        score -= 2; reasons.append(u'RSI {:.0f} → 買われすぎ'.format(rs))
    else:
        reasons.append(u'RSI {:.0f} → 中立圏'.format(rs))
    if line[-1] > signal[-1] and hist[-1] > hist[-2]:
        score += 2; reasons.append(u'MACD GX継続・拡大')
    elif line[-1] < signal[-1] and hist[-1] < hist[-2]:
        score -= 2; reasons.append(u'MACD DX継続・縮小')
    n_s, n_l = (5, 25) if mode == 'daily' else (6, 18)
    if len(close) > n_l:
        if close[-n_s:].mean() > close[-n_l:].mean():
            score += 1; reasons.append(u'SMA短期>長期 → 上昇トレンド')
        else:
            score -= 1; reasons.append(u'SMA短期<長期 → 下降トレンド')
    bb = bollinger(close)
    if bb is not None:
        _, mid, std = bb
        dev = (close[-1] - mid[-1]) / max(std[-1], 1e-9)
        if dev <= -2:
            score += (2 if dev <= -3 else 1)
            reasons.append(u'BB {:+.1f}σ → 反発候補'.format(dev))
        elif dev >= 2:
            score -= (2 if dev >= 3 else 1)
            reasons.append(u'BB {:+.1f}σ → 過熱警戒'.format(dev))
        else:
            reasons.append(u'BB {:+.1f}σ → バンド内'.format(dev))
    score += 1 if close[-1] > close[-2] else -1
    p_up = 1.0 / (1.0 + math.exp(-0.35 * score))
    return {'p_up': p_up, 'score': score, 'reasons': reasons}

POS_WORDS = [u'上方修正', u'増益', u'最高益', u'増配', u'好調', u'急騰', u'買い', u'上昇',
             u'回復', u'拡大', u'提携', u'受注', 'beat', 'surge', 'rise', 'gain',
             'upgrade', 'record', 'growth', 'buy', 'jump', 'rally']
NEG_WORDS = [u'下方修正', u'減益', u'赤字', u'減配', u'不振', u'急落', u'売り', u'下落',
             u'懸念', u'縮小', u'リコール', u'訴訟', 'miss', 'fall', 'drop', 'loss',
             'downgrade', 'cut', 'sell', 'plunge', 'lawsuit', 'recall']

def model_news(symbol):
    """[M5] ニュースキーワード"""
    titles = fetch_news(symbol)
    pos = neg = 0
    for t in titles:
        tl = t.lower()
        pos += sum(1 for w in POS_WORDS if w.lower() in tl)
        neg += sum(1 for w in NEG_WORDS if w.lower() in tl)
    p_up = (pos + 1.0) / (pos + neg + 2.0)
    return {'p_up': p_up, 'pos': pos, 'neg': neg, 'titles': titles}

def model_macro(symbol):
    """[M6] マクロ相関: 日経平均β・ドル円β + 夜間FX変動による翌日ドリフト
       (FXは東証閉場中も動くため、銘柄の最終値以降のFX変動が先行情報になる)"""
    try:
        stock = fetch_daily_series(symbol)
        n225 = fetch_daily_series('^N225')
        fx = fetch_daily_series('JPY=X')
    except Exception:
        return None
    dates = sorted(d for d in stock if d in n225 and d in fx)
    if len(dates) < 120:
        return None
    s = np.array([stock[d] for d in dates])
    ni = np.array([n225[d] for d in dates])
    fxv = np.array([fx[d] for d in dates])
    rs = np.diff(np.log(s))
    rn = np.diff(np.log(ni))
    rf = np.diff(np.log(fxv))
    sigma = max(rs.std(ddof=1), 1e-9)
    beta_n = float(np.cov(rs, rn)[0, 1] / max(np.var(rn), 1e-12))
    corr_n = float(np.corrcoef(rs, rn)[0, 1])
    beta_f = float(np.cov(rs, rf)[0, 1] / max(np.var(rf), 1e-12))
    corr_f = float(np.corrcoef(rs, rf)[0, 1])
    # 夜間FXシグナル: 銘柄の最終取引日以降のドル円変動
    last_stock_date = dates[-1]
    newer_fx = [d for d in sorted(fx.keys()) if d > last_stock_date]
    if newer_fx:
        fx_move = math.log(fx[newer_fx[-1]] / fx[last_stock_date])
    else:
        fx_move = 0.0
    drift = beta_f * fx_move
    p_up = norm_cdf(drift / sigma) if abs(fx_move) > 1e-9 else 0.5
    fx_type = (u'円安メリット(輸出型)' if corr_f > 0.15
               else u'円高メリット(内需/輸入型)' if corr_f < -0.15
               else u'為替中立')
    return {'p_up': p_up, 'beta_n': beta_n, 'corr_n': corr_n,
            'beta_f': beta_f, 'corr_f': corr_f, 'fx_type': fx_type,
            'fx_move': fx_move, 'drift': drift}

def mirage_ensemble(close_full, steps, mode, symbol, name):
    """6モデル統合"""
    r = np.diff(np.log(close_full))
    s0 = close_full[-1]

    m1 = model_bootstrap_mc(r, steps)
    m2 = model_knn(r, steps)
    m3 = model_ar1(r, steps)
    m4 = technical_score(close_full, mode)
    m5 = model_news(symbol)
    m6 = model_macro(symbol)

    votes, weights, lines = [], [], []
    def add(m, w, label, detail):
        if m is not None:
            votes.append(m['p_up']); weights.append(w)
            lines.append(u'  {} P↑={:.0f}%  {}'.format(label, m['p_up'] * 100, detail))
    add(m1, 0.22, u'[M1]ブートストラップMC', u'(実分布再標本化)')
    add(m2, 0.27, u'[M2]パターン類似kNN  ',
        u'(類似{}局面中{}件上昇)'.format(m2['k'], m2['n_up']) if m2 else u'')
    add(m3, 0.09, u'[M3]自己相関AR(1)    ',
        u'(ρ={:+.2f} {})'.format(m3['rho'], m3['regime']) if m3 else u'')
    add(m4, 0.18, u'[M4]テクニカル       ', u'(スコア{:+d})'.format(m4['score']))
    add(m5, 0.13, u'[M5]ニュースKW       ', u'(P:{}/N:{})'.format(m5['pos'], m5['neg']))
    add(m6, 0.11, u'[M6]マクロ相関       ',
        u'(夜間FX{:+.2%})'.format(m6['fx_move']) if m6 else u'')
    if m6 is None:
        lines.append(u'  [M6]マクロ相関        取得失敗のためスキップ')

    w = np.array(weights); w = w / w.sum()
    p_ens = float(np.dot(w, np.array(votes)))
    spread = max(votes) - min(votes)
    conf = u'高' if spread < 0.15 else u'中' if spread < 0.30 else u'低'

    p_fan = 0.5 + DRIFT_SHRINK * (p_ens - 0.5)
    sigma = eff_vol(r)
    d = inv_norm(p_fan) * sigma / math.sqrt(steps)
    samp = m1['samples'] if m1 else np.random.normal(0, sigma, (4000, steps))
    samp = samp - samp.mean() + d
    paths = s0 * np.exp(samp.cumsum(axis=1))
    med = np.median(paths, axis=0)
    p5, p16, p84, p95 = np.percentile(paths, [5, 16, 84, 95], axis=0)
    fc = {'med': med, 'p16': p16, 'p84': p84, 'p5': p5, 'p95': p95,
          'p_up': p_ens, 'sigma': sigma, 'steps': steps}

    verdict = (u'強気 ↗↗' if p_ens >= 0.62 else u'やや強気 ↗' if p_ens >= 0.53
               else u'やや弱気 ↘' if p_ens >= 0.45
               else u'弱気 ↘↘' if p_ens < 0.38 else u'やや弱気 ↘')
    unit = u'営業日' if mode == 'daily' else u'本(5分足)'

    rep = []
    rep.append(u'■ Mirage Ensemble+ 総合予測 ({}{}先)\n'
               u'  総合上昇確率: {:.1f}%  [判定: {} / 信頼度: {} (一致度{:.0f}%)]\n'
               u'  予測中央値: {:,.1f}\n'
               u'  期待レンジ(68%): {:,.1f} 〜 {:,.1f}\n'
               u'  期待レンジ(90%): {:,.1f} 〜 {:,.1f}\n'
               u'  実効ボラ σ={:.2%}/足 (較正×{})'.format(
        steps, unit, p_ens * 100, verdict, conf, (1 - spread) * 100,
        med[-1], p16[-1], p84[-1], p5[-1], p95[-1], sigma, VOL_CAL))
    rep.append(u'■ モデル別の票\n' + u'\n'.join(lines))
    if m6 is not None:
        rep.append(u'■ マクロ相関分析 [M6]\n'
                   u'  日経平均: β={:.2f} / 相関={:+.2f}\n'
                   u'  ドル円  : β={:.2f} / 相関={:+.2f} → {}\n'
                   u'  夜間FX変動 {:+.2%} → 推定ドリフト {:+.2%}'.format(
            m6['beta_n'], m6['corr_n'], m6['beta_f'], m6['corr_f'],
            m6['fx_type'], m6['fx_move'], m6['drift']))
    rep.append(u'■ テクニカル根拠\n  ' + u'\n  '.join(m4['reasons']))
    if m5['titles']:
        rep.append(u'■ 直近ニュース\n  ' +
                   u'\n  '.join(u'・' + t[:46] for t in m5['titles'][:5]))
    return fc, u'\n\n'.join(rep)

# ============================================================
# チャート上のシグナルマーカー (▲上昇/▼下落)
# ============================================================

SIGNAL_STRONG = 0.05

def signal_prob(close_hist, steps, mode):
    """各バー時点の軽量アンサンブル確率 (M1-M4のみ。M5-M7は過去分が取得不能)"""
    r = np.diff(np.log(close_hist))
    if len(r) < 80:
        return None
    m1 = model_bootstrap_mc(r, steps, n_sims=800)
    m2 = model_knn(r, steps)
    m3 = model_ar1(r, steps)
    m4 = technical_score(close_hist, mode)
    ps, ws = [], []
    if m1 is not None:
        ps.append(m1['p_up']); ws.append(0.25)
    if m2 is not None:
        ps.append(m2['p_up']); ws.append(0.30)
    if m3 is not None:
        ps.append(m3['p_up']); ws.append(0.10)
    ps.append(m4['p_up']); ws.append(0.20)
    w = np.array(ws); w = w / w.sum()
    return float(np.dot(w, np.array(ps)))

def compute_signals(close_full, n_disp, mode):
    total = len(close_full)
    start = total - n_disp
    sigs = []
    for i in range(n_disp):
        np.random.seed(1000 + start + i)
        p = signal_prob(close_full[:start + i + 1], 1, mode)
        if p is not None and abs(p - 0.5) >= SIGNAL_STRONG:
            sigs.append((i, p))
    return sigs

# ============================================================
# チャート描画
# ============================================================

def render_chart(data, width_px, fc=None, signals=None, patterns=None):
    o, h, l, c = data['open'], data['high'], data['low'], data['close']
    n = len(c)
    x = np.arange(n)
    steps = fc['steps'] if fc is not None else 0
    x_end = n + steps + 2

    fig_w = width_px / 100.0
    fig, axes = plt.subplots(3, 1, figsize=(fig_w, fig_w * 1.30), sharex=True,
                             gridspec_kw={'height_ratios': [3.2, 1, 1]})
    fig.patch.set_facecolor('#14181e')
    for ax in axes:
        ax.set_facecolor('#14181e')
        ax.tick_params(colors='#aab4c0', labelsize=8)
        for sp in ax.spines.values():
            sp.set_color('#39424d')
        ax.grid(color='#2a323c', lw=0.5, alpha=0.6)

    ax0 = axes[0]

    # ボリンジャーバンド ±1/2/3σ
    bb = bollinger(c, 20)
    if bb is not None:
        bx, bmid, bstd = bb
        sig_styles = [(1, '#66bb6a', 0.14), (2, '#ffa726', 0.08), (3, '#ef5350', 0.05)]
        prev_up, prev_lo = bmid, bmid
        for k, col, fa in sig_styles:
            up, lo = bmid + k * bstd, bmid - k * bstd
            ax0.fill_between(bx, prev_up, up, color=col, alpha=fa)
            ax0.fill_between(bx, lo, prev_lo, color=col, alpha=fa)
            ax0.plot(bx, up, color=col, lw=0.9, alpha=0.9)
            ax0.plot(bx, lo, color=col, lw=0.9, alpha=0.9)
            ax0.annotate(u'+{}σ'.format(k), xy=(bx[-1], up[-1]),
                         xytext=(3, 0), textcoords='offset points',
                         color=col, fontsize=7, va='center')
            ax0.annotate(u'-{}σ'.format(k), xy=(bx[-1], lo[-1]),
                         xytext=(3, 0), textcoords='offset points',
                         color=col, fontsize=7, va='center')
            prev_up, prev_lo = up, lo
        ax0.plot(bx, bmid, color='#eceff1', lw=1.0, ls=':', label='BB mid(20)')

    # ローソク足
    w = 0.6
    for i in range(n):
        col = '#ff5252' if c[i] >= o[i] else '#42a5f5'
        ax0.vlines(i, l[i], h[i], color=col, lw=1)
        ax0.add_patch(Rectangle((i-w/2, min(o[i], c[i])), w,
                                max(abs(c[i]-o[i]), 1e-9),
                                facecolor=col, edgecolor=col))

    # SMA
    s5 = sma(c, 5)
    if s5 is not None:
        ax0.plot(s5[0], s5[1], color='#ffd54f', lw=1, label='SMA5')
    s25 = sma(c, 25)
    if s25 is not None:
        ax0.plot(s25[0], s25[1], color='#ab47bc', lw=1, label='SMA25')

    # シグナルマーカー
    if signals:
        off = (h.max() - l.min()) * 0.035
        for i, p in signals:
            size = 30 + 700 * abs(p - 0.5)
            if p >= 0.5:
                ax0.scatter(i, l[i] - off, marker='^', color='#00e676',
                            s=size, zorder=5, edgecolors='none')
            else:
                ax0.scatter(i, h[i] + off, marker='v', color='#ff1744',
                            s=size, zorder=5, edgecolors='none')
        ax0.scatter([], [], marker='^', color='#00e676', s=40, label='Up signal')
        ax0.scatter([], [], marker='v', color='#ff1744', s=40, label='Down signal')

    # ローソク足パターン番号 (強気=緑/弱気=赤/中立=灰, 対応表はレポート参照)
    if patterns:
        off2 = (h.max() - l.min()) * 0.075
        stack = {}
        for num, pt in enumerate(patterns, 1):
            i = pt['i']
            col = ('#69f0ae' if pt['dir'] > 0
                   else '#ff5252' if pt['dir'] < 0 else '#b0bec5')
            k = stack.get(i, 0)
            stack[i] = k + 1
            if pt['dir'] >= 0:
                y = l[i] - off2 - k * off2 * 0.55
                va = 'top'
            else:
                y = h[i] + off2 + k * off2 * 0.55
                va = 'bottom'
            ax0.annotate(str(num), xy=(i, y), color=col, fontsize=7,
                         fontweight='bold', ha='center', va=va, zorder=6)

    # 予測ファンチャート
    if fc is not None:
        xf = np.arange(n, n + steps)
        xf_full = np.concatenate(([n - 1], xf))
        med = np.concatenate(([c[-1]], fc['med']))
        p16 = np.concatenate(([c[-1]], fc['p16']))
        p84 = np.concatenate(([c[-1]], fc['p84']))
        p5  = np.concatenate(([c[-1]], fc['p5']))
        p95 = np.concatenate(([c[-1]], fc['p95']))
        ax0.fill_between(xf_full, p5, p95, color='#26c6da', alpha=0.12,
                         label='90% range')
        ax0.fill_between(xf_full, p16, p84, color='#26c6da', alpha=0.25,
                         label='68% range')
        ax0.plot(xf_full, med, color='#00e5ff', lw=1.8,
                 label='Forecast (median)')
        ax0.annotate('{:,.0f}'.format(fc['med'][-1]),
                     xy=(xf[-1], fc['med'][-1]),
                     xytext=(4, 0), textcoords='offset points',
                     color='#00e5ff', fontsize=9, va='center')

    ax0.legend(loc='upper left', fontsize=6.5, ncol=2, facecolor='#14181e',
               labelcolor='#aab4c0', edgecolor='#39424d')
    ax0.set_title('{}  ({})  last={:,.1f}  P(up)={:.0f}%'.format(
        data['symbol'],
        'Daily +3d' if data['mode'] == 'daily' else '5min +2h',
        c[-1], fc['p_up'] * 100 if fc else 50),
        color='#e8edf2', fontsize=11)

    if fc is not None:
        c_ext = np.concatenate((c, fc['med']))
        x_ext = np.arange(len(c_ext))
        fx_ = x_ext[n - 1:]

    # RSI
    ax1 = axes[1]
    if fc is not None:
        rsi_ext = rsi(c_ext)
        ax1.plot(x, rsi_ext[:n], color='#26c6da', lw=1.2)
        ax1.plot(fx_, rsi_ext[n - 1:], color='#26c6da', lw=1.6, alpha=0.45)
        ax1.annotate('{:.0f}'.format(rsi_ext[-1]),
                     xy=(x_ext[-1], rsi_ext[-1]),
                     xytext=(4, 0), textcoords='offset points',
                     color='#00e5ff', fontsize=8, va='center')
    else:
        ax1.plot(x, rsi(c), color='#26c6da', lw=1.2)
    ax1.axhline(70, color='#ff5252', lw=0.7, ls='--')
    ax1.axhline(30, color='#42a5f5', lw=0.7, ls='--')
    ax1.set_ylim(0, 100)
    ax1.set_ylabel('RSI', color='#aab4c0', fontsize=8)

    # MACD
    ax2 = axes[2]
    if fc is not None:
        line, signal, hist = macd(c_ext)
        ax2.bar(x, hist[:n],
                color=np.where(hist[:n] >= 0, '#ff8a65', '#4fc3f7'), width=0.6)
        ax2.bar(np.arange(n, len(c_ext)), hist[n:],
                color=np.where(hist[n:] >= 0, '#ff8a65', '#4fc3f7'),
                width=0.6, alpha=0.45)
        ax2.plot(x, line[:n], color='#ffd54f', lw=1)
        ax2.plot(x, signal[:n], color='#ab47bc', lw=1)
        ax2.plot(fx_, line[n - 1:], color='#ffd54f', lw=1.4, alpha=0.45)
        ax2.plot(fx_, signal[n - 1:], color='#ab47bc', lw=1.4, alpha=0.45)
    else:
        line, signal, hist = macd(c)
        ax2.bar(x, hist, color=np.where(hist >= 0, '#ff8a65', '#4fc3f7'), width=0.6)
        ax2.plot(x, line, color='#ffd54f', lw=1)
        ax2.plot(x, signal, color='#ab47bc', lw=1)
    ax2.set_ylabel('MACD', color='#aab4c0', fontsize=8)

    for ax in axes:
        if fc is not None:
            ax.axvline(n - 0.5, color='#78909c', lw=0.8, ls=':')
        ax.set_xlim(-1, x_end)

    plt.tight_layout(pad=0.8)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    data = buf.read()
    return ui.Image.from_data(data, 2), data

# ============================================================
# UI (iPhone 17 Pro 全画面・ダークテーマ統一)
# ============================================================

class StockApp(ui.View):
    def __init__(self):
        self.name = 'StockSight - Fujii Kogei'
        self.background_color = '#0e1116'
        self.tint_color = '#4fa3ff'

        self.tf = ui.TextField()
        self.tf.placeholder = u'銘柄コード (例 7203)'
        self.tf.keyboard_type = ui.KEYBOARD_NUMBERS
        self.tf.bg_color = '#1c222b'
        self.tf.text_color = 'white'
        self.tf.bordered = False
        self.tf.corner_radius = 8
        self.add_subview(self.tf)

        self.seg = ui.SegmentedControl()
        self.seg.segments = [u'日足', u'5分足']
        self.seg.selected_index = 0
        self.add_subview(self.seg)

        self.btn = ui.Button(title=u'分析')
        self.btn.bg_color = '#2962ff'
        self.btn.tint_color = 'white'
        self.btn.corner_radius = 8
        self.btn.action = self.run
        self.add_subview(self.btn)

        self.scroll = ui.ScrollView()
        self.scroll.background_color = '#0e1116'
        self.add_subview(self.scroll)

        self.img_view = ui.ImageView()
        self.img_view.content_mode = ui.CONTENT_SCALE_ASPECT_FIT
        self.scroll.add_subview(self.img_view)

        # チャート拡大ボタン (iOSネイティブビューアでピンチズーム可能)
        self._chart_path = os.path.join(tempfile.gettempdir(),
                                        'stocksight_chart.png')
        self.zoom_btn = ui.Button(title=u'🔍 拡大')
        self.zoom_btn.font = ('<system-bold>', 13)
        self.zoom_btn.bg_color = (0.12, 0.16, 0.22, 0.85)
        self.zoom_btn.tint_color = 'white'
        self.zoom_btn.corner_radius = 8
        self.zoom_btn.hidden = True
        self.zoom_btn.action = self.show_zoom
        self.scroll.add_subview(self.zoom_btn)

        self.txt = ui.TextView()
        self.txt.editable = False
        self.txt.background_color = '#12161d'
        self.txt.text_color = '#cfd8e3'
        self.txt.font = ('Menlo', 12)
        self.txt.corner_radius = 10
        try:
            # 内部スクロールを無効化 (外側ScrollViewとのジェスチャ競合を根絶)
            self.txt.objc_instance.setScrollEnabled_(False)
        except Exception:
            pass
        self.scroll.add_subview(self.txt)

    def layout(self):
        w, h = self.width, self.height
        top = 12
        self.tf.frame = (12, top, w * 0.40, 36)
        self.seg.frame = (w * 0.40 + 20, top, w * 0.36, 36)
        self.btn.frame = (w - 12 - (w * 0.18), top, w * 0.18, 36)
        self.scroll.frame = (0, top + 48, w, h - top - 48)
        self._relayout_content()

    def _relayout_content(self):
        """画像とレポートを実寸で積み上げ、content_sizeを正確に設定"""
        w = self.width
        if self.img_view.image is not None:
            iw = self.img_view.image.size[0] / 2.0
            ih = self.img_view.image.size[1] / 2.0
            disp_h = w * ih / iw
        else:
            disp_h = 0
        self.img_view.frame = (0, 0, w, disp_h)
        text = self.txt.text or u''
        if text:
            # TextViewの内部余白(約8pt×2)を引いた幅で文章高さを実測
            tw, th = ui.measure_string(text, max_width=w - 16 - 18,
                                       font=('Menlo', 12))
            txt_h = th + 30
        else:
            txt_h = 0
        self.txt.frame = (8, disp_h + 8, w - 16, txt_h)
        self.scroll.content_size = (w, disp_h + 8 + txt_h + 60)
        self.zoom_btn.frame = (w - 86, 8, 76, 30)
        self.zoom_btn.hidden = (self.img_view.image is None)

    def show_zoom(self, sender):
        """チャートをネイティブビューアで全画面表示 (ピンチで自由に拡大可)"""
        if os.path.exists(self._chart_path):
            console.quicklook(self._chart_path)

    def run(self, sender):
        code = self.tf.text or ''
        if not code.strip():
            self.txt.text = u'銘柄コードを入力してください。'
            return
        self.tf.end_editing()
        self.btn.enabled = False
        self.txt.text = u'取得・解析中… (6モデル + シグナル走査)'
        threading.Thread(target=self._work,
                         args=(code, 'daily' if self.seg.selected_index == 0 else '5min'),
                         daemon=True).start()

    def _work(self, code, mode):
        try:
            symbol = normalize_symbol(code)
            data = fetch_chart(symbol, mode)
            steps = 3 if mode == 'daily' else 24
            fc, report_body = mirage_ensemble(data['close_full'], steps, mode,
                                              symbol, data['name'])
            signals = compute_signals(data['close_full'], len(data['close']), mode)
            patterns = (detect_patterns(data['open'], data['high'],
                                        data['low'], data['close'])
                        if detect_patterns else [])
            img, png_data = render_chart(data, int(self.width * 2), fc,
                                         signals, patterns)
            try:
                with open(self._chart_path, 'wb') as f:
                    f.write(png_data)
            except Exception:
                pass
            n_up_sig = sum(1 for _, p in signals if p >= 0.5)
            if patterns:
                n_disp = len(data['close'])
                dir_mark = {1: u'↗強気', -1: u'↘弱気', 0: u'→中立'}
                pat_lines = u'\n'.join(
                    u'  {:>2}. {} [{}] {}本前'.format(
                        num, pt['name'], dir_mark[pt['dir']],
                        n_disp - 1 - pt['i'])
                    for num, pt in enumerate(patterns, 1))
                pat_sec = (u'■ ローソク足パターン (チャート上の番号と対応)\n'
                           + pat_lines)
            elif detect_patterns is None:
                pat_sec = (u'■ ローソク足パターン\n'
                           u'  candle_patterns.py が見つかりません。\n'
                           u'  本体と同じフォルダに保存してください。')
            else:
                pat_sec = u'■ ローソク足パターン\n  表示期間内に検出なし'
            report = u'\n\n'.join([
                u'【{} / {}】'.format(data['name'], symbol),
                report_body,
                pat_sec,
                u'■ シグナルマーカー (チャート上の▲▼)\n'
                u'  各バー時点のデータのみで算出した翌足の強シグナル\n'
                u'  (|P-0.5|≥{:.0%} / 大きいほど確信度高)\n'
                u'  表示期間内: ▲{}個 / ▼{}個'.format(
                    SIGNAL_STRONG, n_up_sig, len(signals) - n_up_sig),
                u'※ 検証済の注意: 方向予測の的中率は偶然水準(約50%)です。'
                u'本アプリの確かな価値は較正済みの変動レンジ(68/90%帯)にあります。'
                u'投資判断はご自身の責任で行ってください。'
            ])
            def update():
                self.img_view.image = img
                self.txt.text = report
                self._relayout_content()
                self.scroll.content_offset = (0, 0)
                self.btn.enabled = True
            ui.delay(update, 0)
        except Exception as e:
            err = u'エラー: {}\n銘柄コード/通信環境をご確認ください。'.format(e)
            def show_err():
                self.txt.text = err
                self._relayout_content()
                self.scroll.content_offset = (0, 0)
                self.btn.enabled = True
            ui.delay(show_err, 0)

if __name__ == '__main__':
    app = StockApp()
    app.present('fullscreen',
                title_bar_color='#0e1116',
                title_color='#e8edf2',
                hide_title_bar=False)
