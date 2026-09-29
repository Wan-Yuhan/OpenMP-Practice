#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_figures.py  --  高性能计算 编程作业3 (OpenMP Practice)

读取 results/raw/*.txt 的实测日志, 生成
  results/*.csv       汇总数据
  figures/*.png       报告插图 5 张(图内文字一律用英文)

配色沿用第二次作业已通过 dataviz 校验的三色
  #6a3d9a / #2a7fbf / #d9730d
(全对组合 CVD ΔE 11.5, 普通视觉 ΔE 17.8, 对比度全部 >= 3:1)

用法: python plot_figures.py
"""

import io
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, 'results', 'raw')
FIG = os.path.join(HERE, 'figures')
RES = os.path.join(HERE, 'results')

# ---- 配色(已校验) ----
C1 = '#6a3d9a'          # Case 1 / 串行
C2 = '#2a7fbf'          # Case 2 / 外层并行
C3 = '#d9730d'          # Case 3 / 内层并行
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#2e2d2a'        # 坐标轴标题与柱上数值, 压深到接近正文黑
MUTED = '#898781'       # 仅留给参考线/理想线性
GRID = '#e1e0d9'
AXIS = '#94928a'        # 坐标轴线, 比原来的 #c3c2b7 明显加深

plt.rcParams.update({
    # 报告要求图内英文用 Times New Roman; 数学刻度(10^n)用 STIX(与 Times 同族)
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10,
    'axes.facecolor': SURFACE,
    'figure.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
    'axes.edgecolor': AXIS,
    'axes.labelcolor': INK,
    'xtick.color': INK2,
    'ytick.color': INK2,
    'text.color': INK,
    'axes.titlecolor': INK,
    'axes.grid': True,
    'grid.color': GRID,
    'grid.linewidth': 0.8,
    'axes.axisbelow': True,
    'legend.frameon': False,
    'figure.dpi': 200,
})


def read_raw(name):
    path = os.path.join(RAW, name)
    if not os.path.exists(path):
        sys.stderr.write('  [skip] %s not found\n' % name)
        return None
    with io.open(path, encoding='utf-8', errors='replace') as fh:
        return fh.read()


def parse_kv(line):
    """把 'KEY a=1 b=2' 解析成 dict"""
    out = {}
    for m in re.finditer(r'([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)', line):
        out[m.group(1)] = m.group(2)
    return out


def style(ax, title=None, xlabel=None, ylabel=None):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS)
    if title:
        ax.set_title(title, fontsize=11, color=INK, pad=10, loc='left')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9.5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.5)
    ax.tick_params(length=0, labelsize=9)


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print('  -> figures/%s' % name)


def write_csv(name, header, rows):
    path = os.path.join(RES, name)
    with io.open(path, 'w', encoding='utf-8-sig') as fh:
        fh.write(','.join(header) + '\n')
        for r in rows:
            fh.write(','.join(str(x) for x in r) + '\n')
    print('  -> results/%s' % name)


# =====================================================================
# 图1/图2: 三种 Case 的执行时间与吞吐量
# =====================================================================
def fig_cases():
    txt = read_raw('run_cases.txt')
    if txt is None:
        return None

    res = {}      # name -> {case: (time, perf, checksum)}
    order = []
    for line in txt.splitlines():
        if line.startswith('RESULT '):
            kv = parse_kv(line)
            nm = kv['config']
            if nm not in res:
                res[nm] = {}
                order.append(nm)
            res[nm][int(kv['case'])] = (float(kv['time']), float(kv['perf']))

    if not order:
        print('  [warn] no RESULT lines in run_cases.txt')
        return None

    labels = {
        '1k_x10M': 'VEC_SIZE 1k\nNUM_LOOP 10M',
        '10k_x100k': 'VEC_SIZE 10k\nNUM_LOOP 100k',
        '100k_x100k': 'VEC_SIZE 100k\nNUM_LOOP 100k',
    }
    cases = [1, 2, 3]
    names = ['Case 1 (serial)', 'Case 2 (outer parallel)', 'Case 3 (inner parallel)']
    cols = [C1, C2, C3]

    rows = []
    for nm in order:
        for c in cases:
            if c in res[nm]:
                rows.append([nm, c, '%.6f' % res[nm][c][0], '%.2f' % res[nm][c][1]])
    write_csv('cases.csv', ['config', 'case', 'time_s', 'perf_mflops'], rows)

    # ---- 图1: 执行时间(对数轴) ----
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    w = 0.26
    xs = range(len(order))
    for i, (c, col, nm) in enumerate(zip(cases, cols, names)):
        vals = [res[o].get(c, (0, 0))[0] for o in order]
        pos = [x + (i - 1) * (w + 0.02) for x in xs]
        ax.bar(pos, vals, width=w, color=col, label=nm, zorder=3)
        for x, v in zip(pos, vals):
            if v <= 0:
                continue
            # 统一 3 位小数, 与表 5 逐位一致(原先是按量级换精度的 %.3g/%.1f/%.0f,
            # 会把 3.745 印成 3.7、740.362 印成 740, 与表格对不上)
            lab = '%.3f s' % v
            # 横排: 对数轴把同组三根柱子拉开, 数值差别最小的 3.745/4.480 s 也差
            # 0.078 个数量级(约 2 pt), 而柱心间距 41 pt、最宽的 "740.362 s" 约 36 pt,
            # 不会互相压住
            ax.annotate(lab, (x, v), textcoords='offset points', xytext=(0, 3),
                        ha='center', va='bottom', fontsize=8, color=INK2)

    ax.set_yscale('log')
    ax.set_ylim(1e-2, 1e4)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([labels.get(o, o) for o in order], fontsize=9)
    # 无标题(图题由报告正文的图注承担); 图例移到坐标区上方横排一行,
    # 避免压在 1k×10M 那组柱子上
    style(ax, None, ylabel='Execution time (s, log scale)')
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.005), ncol=3, fontsize=9,
              handlelength=1.5, handleheight=0.9, handletextpad=0.6,
              columnspacing=2.2, borderaxespad=0.0)
    save(fig, 'fig_cases_time.png')

    return res


# =====================================================================
# 图3/图4: 线程数扫描的加速比
# =====================================================================
def read_scale(fname):
    txt = read_raw(fname)
    if txt is None:
        return None
    sweep = {}
    data = {}
    for line in txt.splitlines():
        if line.startswith('SWEEP '):
            kv = parse_kv(line)
            sweep[kv.get('prog', kv.get('name', '?'))] = kv
        elif line.startswith('SCALE '):
            kv = parse_kv(line)
            prog = kv['prog']
            data.setdefault(prog, []).append(kv)
    for prog in data:
        data[prog].sort(key=lambda k: int(k['threads']))
    return data, sweep


def fig_speedup(fname, tag, series, rowsname):
    got = read_scale(fname)
    if got is None:
        return
    data, _ = got
    if not data:
        print('  [warn] no SCALE lines in %s' % fname)
        return

    rows = []
    for prog in data:
        for kv in data[prog]:
            rows.append([prog, kv['threads'], '%.6f' % float(kv['mean']),
                         '%.6f' % float(kv['min']), '%.3f' % float(kv['speedup']),
                         '%.3f' % float(kv['efficiency']),
                         '%.2f' % float(kv['perf'])])
    write_csv(rowsname, ['prog', 'threads', 'time_mean_s', 'time_min_s',
                         'speedup', 'efficiency', 'perf_mflops'], rows)

    keys = [k for k, _, _ in series if k in data]
    if not keys:
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.3))
    ax = axes[0]
    xs = [int(kv['threads']) for kv in data[keys[0]]]
    ax.plot(xs, xs, ls='--', lw=1.4, color=MUTED, zorder=2,
            label='Ideal linear speedup')
    for k, col, lab in series:
        if k not in data:
            continue
        ys, es = [], []
        ref = data[k][0]
        for kv in data[k]:
            sp = float(kv['speedup'])
            # 误差按相对标准差传播: sd(T1/Tp)/T1/Tp = sqrt(cv1^2 + cvp^2)
            c1 = float(ref['sd']) / float(ref['mean']) if float(ref['mean']) else 0.0
            cp = float(kv['sd']) / float(kv['mean']) if float(kv['mean']) else 0.0
            ys.append(sp)
            es.append(sp * ((c1 ** 2 + cp ** 2) ** 0.5))
        ax.errorbar(xs, ys, yerr=es, marker='o', ms=5.5, lw=2.0, color=col,
                    label=lab, zorder=3, capsize=3, elinewidth=1.2,
                    markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax.set_xticks(xs)
    ax.set_xlabel('Threads')
    style(ax, None, ylabel='Speedup  T(1)/T(p)')
    # 左上角本来就是空的: 理想线性虚线在 x<=6 处 y<=6, 三条实测曲线也不超过 2.4,
    # 都落在图例盒子(约占上方 1/3、横向 45%)下方, 只留一点顶部余量别让虚线顶到框
    ax.set_ylim(0, max(xs) * 1.06)
    ax.legend(loc='upper left', fontsize=8.5)

    ax = axes[1]
    perf = []
    for k, col, lab in series:
        if k not in data:
            continue
        ys = [float(kv['perf']) for kv in data[k]]
        perf += ys
        ax.plot(xs, ys, marker='s', ms=5.5, lw=2.0, color=col, label=lab, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax.set_xticks(xs)
    ax.set_xlabel('Threads')
    style(ax, None, ylabel='Throughput (MFLOP/s)')
    # 右图的左上角**不是**空的: daxpy 的 100k 那条在 x=6~8 升到近 1.2 万,
    # 默认上限下正好斜穿图例。图例是左上角锚定的, 压不压线只取决于上限留多高,
    # 所以量出图例盒实际占坐标区高度的比例, 再把上限抬到让曲线最高点刚好落在
    # 图例下沿之下 —— matmul 只有 2 行图例, 会比 3 行的 daxpy 少留约 8% 空白
    leg = ax.legend(loc='upper left', fontsize=8.5)

    fig.tight_layout()
    fig.canvas.draw()       # 量图例尺寸需要先有 renderer
    h_leg = leg.get_window_extent().transformed(ax.transAxes.inverted()).height
    y_bot = min(perf) * 0.85
    ax.set_ylim(y_bot, y_bot + (max(perf) - y_bot) / (1.0 - h_leg - 0.02))
    save(fig, 'fig_threads_%s.png' % tag)


def fig_daxpy_threads():
    fig_speedup(
        'run_threads_daxpy.txt', 'daxpy',
        [('daxpy_small', C1, 'VEC_SIZE 1k (L1 resident)'),
         ('daxpy_mid', C2, 'VEC_SIZE 10k (L2 resident)'),
         ('daxpy_big', C3, 'VEC_SIZE 100k (out of L2)')],
        'threads_daxpy.csv')


def fig_matmul_threads():
    fig_speedup(
        'run_threads_matmul.txt', 'matmul',
        [('matmul_big', C1, 'n = 1024'),
         ('matmul_small', C2, 'n = 256')],
        'threads_matmul.csv')


# =====================================================================
# 图5: parallel region 开销
# =====================================================================
def fig_region():
    txt = read_raw('run_region.txt')
    if txt is None:
        return
    pts = []
    for line in txt.splitlines():
        if line.startswith('REGION_OVERHEAD threads='):
            kv = parse_kv(line)
            pts.append((int(kv['threads']), float(kv['us'])))
    if not pts:
        return
    pts.sort()
    write_csv('region_overhead.csv', ['threads', 'us_per_region'],
              [[p[0], '%.3f' % p[1]] for p in pts])

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    ax.plot(xs, ys, marker='o', ms=6, lw=2.0, color=C3, zorder=3,
            markeredgecolor=SURFACE, markeredgewidth=1.2)
    for x, y in zip(xs, ys):
        ax.annotate('%.1f' % y, (x, y), textcoords='offset points', xytext=(0, 7),
                    ha='center', fontsize=8, color=INK2)
    ax.set_xticks(xs)
    ax.set_ylim(0, max(ys) * 1.28)
    ax.set_xlabel('Threads')
    # 单条曲线, 不需要图例; 标题也去掉(图题由报告的图注承担)
    style(ax, None, ylabel='Cost per parallel region (µs)')
    save(fig, 'fig_region_overhead.png')


# =====================================================================
# 图6: 对照实验 —— 共享写 z vs 每线程私有 z
# =====================================================================
def fig_coherence():
    txt = read_raw('run_coherence.txt')
    if txt is None:
        return
    # 日志形如:
    #   COHERENCE threads=4 shared_z=0.70 speedup=0.88 | private_z=0.22 speedup=2.80
    # 两侧都有 speedup= 键, 必须先按 '|' 切成两半再各自解析
    pts = []
    for line in txt.splitlines():
        if line.startswith('COHERENCE threads='):
            left, sep, right = line.partition('|')
            if not sep:
                continue
            kl = parse_kv(left)
            kr = parse_kv(right)
            pts.append((int(kl['threads']),
                        float(kl['speedup']),      # 共享写
                        float(kr['speedup']),      # 私有写
                        float(kl['shared_z']),
                        float(kr['private_z'])))
    if not pts:
        print('  [warn] no COHERENCE lines in run_coherence.txt')
        return
    pts.sort()
    # 基准必须用「同一个内核、1 线程」的时间, 不能拿串行版的 kernel_case1 当基准:
    # -O2 下只有串行版的循环会被自动向量化, 两个内核对不上, 会出现整段 <1 的假象。
    base_shared = pts[0][3]
    base_priv   = pts[0][4]
    shared = [base_shared / p[3] for p in pts]
    priv   = [base_priv / p[4] for p in pts]

    write_csv('coherence.csv',
              ['threads', 'shared_z_speedup', 'private_z_speedup',
               'shared_z_s', 'private_z_s'],
              [[p[0], '%.3f' % s, '%.3f' % v, '%.6f' % p[3], '%.6f' % p[4]]
               for p, s, v in zip(pts, shared, priv)])

    xs = [p[0] for p in pts]

    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    w = 0.36
    pos = [x - (w / 2 + 0.01) for x in xs]
    ax.bar(pos, shared, width=w, color=C1, label='Shared z (literal PPT code)', zorder=3)
    pos = [x + (w / 2 + 0.01) for x in xs]
    ax.bar(pos, priv, width=w, color=C2, label='Private z slice per thread', zorder=3)
    # 直接标注只给"有加速"的那一路, 共享写那一路靠 1x 参考线读
    for x, v in zip(pos, priv):
        ax.annotate('%.2f' % v, (x, v), textcoords='offset points', xytext=(0, 4),
                    ha='center', fontsize=8, color=INK2)
    ax.axhline(1.0, color=MUTED, ls='--', lw=1.3, zorder=2)
    # 这条注记原来放在 max(xs)+0.95, 文字右半截正好压在 12 线程那根蓝柱上;
    # 把右边界再放 1 个单位、注记中心移到 max(xs)+1.75, 才整段落在柱子的右边
    ax.annotate('1x = no speedup', (max(xs) + 1.75, 1.02), fontsize=8, color=INK2,
                va='bottom', ha='center')

    ax.set_xticks(xs)
    ax.set_xlim(min(xs) - 0.7, max(xs) + 3.0)
    ax.set_ylim(0, max(priv) * 1.18)
    ax.set_xlabel('Threads')
    style(ax, None, ylabel='Speedup vs serial')
    # 左上角是空的: 该区域最高的 4 线程柱只有 2.55, 图例两行只占上方约 1/5
    ax.legend(loc='upper left', fontsize=8.5)
    save(fig, 'fig_coherence.png')


# =====================================================================
def main():
    if not os.path.isdir(FIG):
        os.makedirs(FIG)
    if not os.path.isdir(RES):
        os.makedirs(RES)

    print('parsing raw logs ...')
    fig_cases()
    fig_daxpy_threads()
    fig_matmul_threads()
    fig_region()
    fig_coherence()
    print('done.')


if __name__ == '__main__':
    main()
