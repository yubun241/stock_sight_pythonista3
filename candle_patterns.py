# -*- coding: utf-8 -*-
"""
candle_patterns.py — ローソク足パターン認識エンジン (Pythonista 3)
藤井工藝 / StockSight用モジュール

detect_patterns(o, h, l, c) -> [{'i':終了バーindex, 'name':名称,
                                 'dir':+1強気/-1弱気/0中立, 'grp':分類}, ...]

実装パターン:
  単体  : ハンマー/逆ハンマー/首吊り線/流れ星/大陽線/大陰線/坊主陽線/坊主陰線/
          下影陽線/上影陰線/十字線/トンボ/墓石/コマ
  2本組 : 陽の包み足/陰の包み足/陽のはらみ線/陰のはらみ線/はらみ寄せ線/
          切り込み線/黒雲/たくり線/出会い線/行き違い線/差し込み線/被せ線
  3本組 : 明けの明星/宵の明星/三川明けの明星/三川宵の明星/赤三兵/三羽烏/
          三空叩き込み/三空踏み上げ/三兵押し目/三羽戻り/三手大陽線/三手大陰線
  継続  : 上げ三法/下げ三法/上放れ二羽烏/下放れ二本黒/はらみ上放れ/
          はらみ下放れ/持ち合い放れ/窓埋め/窓開け継続
  窓系  : ギャップアップ/ギャップダウン/ブレイクアウェイG/ランナウェイG/
          エグゾースションG/アイランドリバーサル
  酒田  : 三山/三尊天井/逆三尊
  (赤三兵・三羽烏・三空・三法・三兵は酒田五法と重複のため3本組/継続側で検出)
"""

import numpy as np

MAX_MARKS = 28          # チャートの可読性のための表示上限
_GRP_PRIO = {u'酒田': 0, u'3本': 1, u'継続': 2, u'2本': 3, u'窓': 4, u'単体': 5}


def _trend(c, i, k):
    """バーiに入る直前のトレンド (+1上昇/-1下降/0持合)"""
    j = max(0, i - k)
    if i <= 0:
        return 0
    if c[i] > c[j] * 1.01:
        return 1
    if c[i] < c[j] * 0.99:
        return -1
    return 0


def detect_patterns(o, h, l, c):
    n = len(c)
    if n < 12:
        return []
    o = np.asarray(o, float); h = np.asarray(h, float)
    l = np.asarray(l, float); c = np.asarray(c, float)

    body = np.abs(c - o)
    rng = np.maximum(h - l, 1e-9)
    upsh = h - np.maximum(o, c)
    losh = np.minimum(o, c) - l
    bull = c >= o
    bear = ~bull
    ab = np.array([body[max(0, i - 14):i + 1].mean() for i in range(n)])   # 平均実体
    ar = np.array([rng[max(0, i - 14):i + 1].mean() for i in range(n)])    # 平均レンジ
    sma25 = np.array([c[max(0, i - 24):i + 1].mean() for i in range(n)])
    std20 = np.array([c[max(0, i - 19):i + 1].std(ddof=0) for i in range(n)])

    res = []
    seen = set()

    def add(i, name, dr, grp):
        key = (i, name)
        if key not in seen and 0 <= i < n:
            seen.add(key)
            res.append({'i': int(i), 'name': name, 'dir': dr, 'grp': grp})

    big = lambda i: body[i] >= 1.6 * max(ab[max(0, i - 1)], 1e-9)
    small = lambda i: body[i] <= 0.45 * max(ab[max(0, i - 1)], 1e-9)
    doji = lambda i: body[i] <= 0.10 * rng[i]

    # ================= 単体パターン =================
    for i in range(6, n):
        t = _trend(c, i - 1, 5)
        r = rng[i]; b = body[i]; u = upsh[i]; lo = losh[i]
        hammer_shape = (lo >= 0.55 * r and u <= 0.15 * r and b <= 0.35 * r and not doji(i))
        inv_shape = (u >= 0.55 * r and lo <= 0.15 * r and b <= 0.35 * r and not doji(i))

        if doji(i):
            if lo >= 0.6 * r and u <= 0.12 * r:
                add(i, u'トンボ', +1 if t < 0 else 0, u'単体')
            elif u >= 0.6 * r and lo <= 0.12 * r:
                add(i, u'墓石', -1 if t > 0 else 0, u'単体')
            else:
                add(i, u'十字線(寄引同時線)', 0, u'単体')
        elif u <= 0.05 * r and lo <= 0.05 * r and b >= 0.9 * r and big(i):
            add(i, u'坊主陽線' if bull[i] else u'坊主陰線',
                +1 if bull[i] else -1, u'単体')
        elif big(i):
            add(i, u'大陽線' if bull[i] else u'大陰線',
                +1 if bull[i] else -1, u'単体')
        elif hammer_shape:
            if t < 0:
                add(i, u'ハンマー', +1, u'単体')
            elif t > 0:
                add(i, u'首吊り線', -1, u'単体')
            elif bull[i] and lo >= 2 * max(b, 1e-9):
                add(i, u'下影陽線', +1, u'単体')
        elif inv_shape:
            if t < 0:
                add(i, u'逆ハンマー', +1, u'単体')
            elif t > 0:
                add(i, u'流れ星', -1, u'単体')
            elif bear[i] and u >= 2 * max(b, 1e-9):
                add(i, u'上影陰線', -1, u'単体')
        elif bull[i] and lo >= 2 * b and lo >= 0.4 * r:
            add(i, u'下影陽線', +1, u'単体')
        elif bear[i] and u >= 2 * b and u >= 0.4 * r:
            add(i, u'上影陰線', -1, u'単体')
        elif b <= 0.35 * r and u >= 0.2 * r and lo >= 0.2 * r:
            add(i, u'コマ', 0, u'単体')

    # ================= 2本組パターン =================
    for i in range(7, n):
        t = _trend(c, i - 2, 5)
        p, q = i - 1, i
        bp_hi, bp_lo = max(o[p], c[p]), min(o[p], c[p])
        bq_hi, bq_lo = max(o[q], c[q]), min(o[q], c[q])
        mid_p = (o[p] + c[p]) / 2.0

        # 包み足
        if bull[q] and bear[p] and bq_hi >= bp_hi and bq_lo <= bp_lo and body[q] > body[p]:
            add(q, u'陽の包み足', +1, u'2本')
        elif bear[q] and bull[p] and bq_hi >= bp_hi and bq_lo <= bp_lo and body[q] > body[p]:
            add(q, u'陰の包み足', -1, u'2本')
        # はらみ
        elif body[p] >= 1.2 * ab[p] and bq_hi <= bp_hi and bq_lo >= bp_lo:
            if doji(q):
                add(q, u'はらみ寄せ線', +1 if bear[p] else -1, u'2本')
            elif bear[p] and bull[q] and body[q] <= 0.6 * body[p]:
                add(q, u'陽のはらみ線', +1, u'2本')
            elif bull[p] and bear[q] and body[q] <= 0.6 * body[p]:
                add(q, u'陰のはらみ線', -1, u'2本')
        # 切り込み線 / 差し込み線
        elif bear[p] and bull[q] and o[q] < l[p]:
            if c[q] > mid_p and c[q] < o[p]:
                add(q, u'切り込み線', +1, u'2本')
            elif c[q] > c[p] and c[q] <= mid_p:
                add(q, u'差し込み線', -1, u'2本')
        # 黒雲 / 被せ線
        elif bull[p] and bear[q] and o[q] > h[p] and c[q] < mid_p and c[q] > o[p]:
            add(q, u'黒雲', -1, u'2本')
        elif bull[p] and bear[q] and o[q] > c[p] and mid_p <= c[q] < c[p]:
            add(q, u'被せ線', -1, u'2本')
        # たくり線 (下降中の窓開けハンマー)
        if t < 0 and h[q] < c[p] and losh[q] >= 0.55 * rng[q] and \
                upsh[q] <= 0.15 * rng[q] and body[q] <= 0.35 * rng[q]:
            add(q, u'たくり線', +1, u'2本')
        # 出会い線 (逆色で終値が一致)
        if bull[p] != bull[q] and abs(c[q] - c[p]) <= 0.08 * ar[q]:
            add(q, u'出会い線', +1 if bull[q] else -1, u'2本')
        # 行き違い線 (逆色で窓を開け逆方向へ)
        if bull[p] and bear[q] and o[q] > h[p] and c[q] > c[p]:
            pass  # 上放れ系で扱う
        elif bear[p] and bull[q] and o[q] > c[p] and c[q] > o[p] and o[q] > o[p]:
            add(q, u'行き違い線', +1, u'2本')
        elif bull[p] and bear[q] and o[q] < c[p] and c[q] < o[p] and o[q] < o[p]:
            add(q, u'行き違い線', -1, u'2本')

    # ================= 3本組パターン =================
    for i in range(8, n):
        a, b3, q = i - 2, i - 1, i
        t_long = _trend(c, a - 1, 15)
        t_short = _trend(c, a - 1, 4)
        ba_hi, ba_lo = max(o[a], c[a]), min(o[a], c[a])
        bb_hi, bb_lo = max(o[b3], c[b3]), min(o[b3], c[b3])
        mid_a = (o[a] + c[a]) / 2.0

        # 明けの明星 / 三川明けの明星
        if bear[a] and big(a) and bb_hi < c[a] and small(b3) and \
                bull[q] and c[q] > mid_a:
            if doji(b3) and bb_hi < min(c[a], o[q]):
                add(q, u'三川明けの明星', +1, u'3本')
            else:
                add(q, u'明けの明星', +1, u'3本')
        # 宵の明星 / 三川宵の明星
        if bull[a] and big(a) and bb_lo > c[a] and small(b3) and \
                bear[q] and c[q] < mid_a:
            if doji(b3) and bb_lo > max(c[a], o[q]):
                add(q, u'三川宵の明星', -1, u'3本')
            else:
                add(q, u'宵の明星', -1, u'3本')
        # 赤三兵 / 三兵押し目
        if all(bull[k] for k in (a, b3, q)) and c[a] < c[b3] < c[q] and \
                ba_lo <= o[b3] <= ba_hi and bb_lo <= o[q] <= bb_hi and \
                all(body[k] >= 0.6 * ab[k] for k in (a, b3, q)):
            if t_long > 0 and t_short < 0:
                add(q, u'三兵押し目', +1, u'3本')
            else:
                add(q, u'赤三兵', +1, u'3本')
        # 三羽烏 / 三羽戻り
        if all(bear[k] for k in (a, b3, q)) and c[a] > c[b3] > c[q] and \
                ba_lo <= o[b3] <= ba_hi and bb_lo <= o[q] <= bb_hi and \
                all(body[k] >= 0.6 * ab[k] for k in (a, b3, q)):
            if t_long < 0 and t_short > 0:
                add(q, u'三羽戻り', -1, u'3本')
            else:
                add(q, u'三羽烏', -1, u'3本')
        # 三空踏み上げ / 三空叩き込み
        if all(l[k] > h[k - 1] for k in (a, b3, q)) and all(bull[k] for k in (a, b3, q)):
            add(q, u'三空踏み上げ', -1, u'3本')
        if all(h[k] < l[k - 1] for k in (a, b3, q)) and all(bear[k] for k in (a, b3, q)):
            add(q, u'三空叩き込み', +1, u'3本')
        # 三手大陽線 / 三手大陰線
        if all(bull[k] and body[k] >= 1.4 * ab[k] for k in (a, b3, q)):
            add(q, u'三手大陽線', -1, u'3本')
        if all(bear[k] and body[k] >= 1.4 * ab[k] for k in (a, b3, q)):
            add(q, u'三手大陰線', +1, u'3本')

    # ================= 継続パターン =================
    gaps = []   # (index, 'up'/'dn', gap上端, gap下端)
    for i in range(1, n):
        if l[i] > h[i - 1] and (l[i] - h[i - 1]) >= 0.45 * ar[i]:
            gaps.append((i, 'up', l[i], h[i - 1]))
        elif h[i] < l[i - 1] and (l[i - 1] - h[i]) >= 0.45 * ar[i]:
            gaps.append((i, 'dn', l[i - 1], h[i]))

    for i in range(10, n):
        t = _trend(c, i - 1, 8)
        # 上げ三法 / 下げ三法 (5本)
        f = i - 4
        mids = [f + 1, f + 2, f + 3]
        if bull[f] and big(f) and bull[i] and big(i) and c[i] > c[f] and \
                all(small(k) or body[k] < body[f] * 0.5 for k in mids) and \
                all(l[k] >= l[f] - 0.1 * ar[i] and h[k] <= h[f] + 0.1 * ar[i] for k in mids):
            add(i, u'上げ三法', +1, u'継続')
        if bear[f] and big(f) and bear[i] and big(i) and c[i] < c[f] and \
                all(small(k) or body[k] < body[f] * 0.5 for k in mids) and \
                all(l[k] >= l[f] - 0.1 * ar[i] and h[k] <= h[f] + 0.1 * ar[i] for k in mids):
            add(i, u'下げ三法', -1, u'継続')
        # 上放れ二羽烏
        if _trend(c, i - 2, 8) > 0 and o[i - 1] > c[i - 2] and bear[i - 1] and \
                bear[i] and o[i] >= o[i - 1] and c[i] < c[i - 1] and c[i] > c[i - 2]:
            add(i, u'上放れ二羽烏', -1, u'継続')
        # 下放れ二本黒
        if _trend(c, i - 2, 8) < 0 and h[i - 1] < l[i - 2] and bear[i - 1] and bear[i]:
            add(i, u'下放れ二本黒', -1, u'継続')
        # はらみ上放れ / 下放れ
        p2, p1 = i - 2, i - 1
        if body[p2] >= 1.2 * ab[p2] and \
                max(o[p1], c[p1]) <= max(o[p2], c[p2]) and \
                min(o[p1], c[p1]) >= min(o[p2], c[p2]):
            if c[i] > max(h[p1], h[p2]):
                add(i, u'はらみ上放れ', +1, u'継続')
            elif c[i] < min(l[p1], l[p2]):
                add(i, u'はらみ下放れ', -1, u'継続')
        # 持ち合い放れ (直近7本のレンジをブレイク)
        win_h = h[i - 7:i].max(); win_l = l[i - 7:i].min()
        if (win_h - win_l) <= 2.2 * ar[i]:
            if c[i] > win_h + 0.1 * ar[i]:
                add(i, u'持ち合い放れ(上)', +1, u'継続')
            elif c[i] < win_l - 0.1 * ar[i]:
                add(i, u'持ち合い放れ(下)', -1, u'継続')

    # ================= 窓(ギャップ)系 =================
    for gi, (i, gdir, g_hi, g_lo) in enumerate(gaps):
        t = _trend(c, i - 1, 8)
        win_h = h[max(0, i - 10):i].max()
        win_l = l[max(0, i - 10):i].min()
        consolidating = (win_h - win_l) <= 2.5 * ar[i]
        extended = abs(c[i - 1] - sma25[i - 1]) > 2.2 * max(std20[i - 1], 1e-9)
        if gdir == 'up':
            if consolidating and l[i] > win_h:
                add(i, u'ブレイクアウェイG(上)', +1, u'窓')
            elif t > 0 and extended:
                add(i, u'エグゾースションG(上)', -1, u'窓')
            elif t > 0:
                add(i, u'ランナウェイG(上)', +1, u'窓')
            else:
                add(i, u'ギャップアップ', +1, u'窓')
            if bull[i] and t > 0 and c[i] > h[i - 1] + 0.3 * ar[i]:
                add(i, u'窓開け継続(上)', +1, u'継続')
        else:
            if consolidating and h[i] < win_l:
                add(i, u'ブレイクアウェイG(下)', -1, u'窓')
            elif t < 0 and extended:
                add(i, u'エグゾースションG(下)', +1, u'窓')
            elif t < 0:
                add(i, u'ランナウェイG(下)', -1, u'窓')
            else:
                add(i, u'ギャップダウン', -1, u'窓')
            if bear[i] and t < 0 and c[i] < l[i - 1] - 0.3 * ar[i]:
                add(i, u'窓開け継続(下)', -1, u'継続')
        # 窓埋め (10本以内に窓ゾーンへ到達)
        for j in range(i + 1, min(i + 11, n)):
            if l[j] <= g_lo if gdir == 'up' else h[j] >= g_hi:
                add(j, u'窓埋め', 0, u'継続')
                break
        # アイランドリバーサル (逆方向の窓が8本以内)
        for (i2, gdir2, _, _) in gaps[gi + 1:]:
            if i2 - i <= 8 and gdir2 != gdir:
                add(i2, u'アイランドリバーサル',
                    -1 if gdir == 'up' else +1, u'窓')
                break

    # ================= 酒田五法 (三山/三尊/逆三尊) =================
    def _peaks(arr, mode):
        pk = []
        for i in range(2, n - 2):
            seg = arr[i - 2:i + 3]
            if mode == 'hi' and arr[i] == seg.max() and arr[i] > arr[i - 2] and arr[i] > arr[i + 2]:
                pk.append(i)
            if mode == 'lo' and arr[i] == seg.min() and arr[i] < arr[i - 2] and arr[i] < arr[i + 2]:
                pk.append(i)
        return pk

    hp = [i for i in _peaks(h, 'hi')]
    lp = [i for i in _peaks(l, 'lo')]
    if len(hp) >= 3:
        p1, p2, p3 = hp[-3], hp[-2], hp[-1]
        if p2 - p1 >= 3 and p3 - p2 >= 3 and (n - 1 - p3) <= 12:
            v1, v2, v3 = h[p1], h[p2], h[p3]
            avg = (v1 + v2 + v3) / 3.0
            if max(v1, v2, v3) / min(v1, v2, v3) <= 1.03:
                add(p3, u'三山', -1, u'酒田')
            elif v2 > v1 * 1.01 and v2 > v3 * 1.01 and abs(v1 - v3) / avg <= 0.03:
                add(p3, u'三尊天井', -1, u'酒田')
    if len(lp) >= 3:
        p1, p2, p3 = lp[-3], lp[-2], lp[-1]
        if p2 - p1 >= 3 and p3 - p2 >= 3 and (n - 1 - p3) <= 12:
            v1, v2, v3 = l[p1], l[p2], l[p3]
            avg = (v1 + v2 + v3) / 3.0
            if v2 < v1 * 0.99 and v2 < v3 * 0.99 and abs(v1 - v3) / avg <= 0.03:
                add(p3, u'逆三尊', +1, u'酒田')

    # ================= 整理: 同名連続の間引き → 優先度順に上限適用 =================
    res.sort(key=lambda d: (d['name'], d['i']))
    dedup, last = [], {}
    for d in res:
        if d['name'] in last and d['i'] - last[d['name']] <= 2:
            last[d['name']] = d['i']
            continue
        last[d['name']] = d['i']
        dedup.append(d)
    res = dedup

    def _prio(d):
        p = _GRP_PRIO.get(d['grp'], 9)
        if d['name'] == u'窓埋め':
            p = 8                       # 情報量が低いため最下位
        return p
    res.sort(key=lambda d: (_prio(d), -d['i']))
    res = res[:MAX_MARKS]
    res.sort(key=lambda d: d['i'])
    return res
