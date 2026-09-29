#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_tables.py  --  把 results/*.csv 汇总成报告用的 Markdown 表格

输出到 results/tables.md, 直接粘进报告即可, 避免手工誊抄出错。
用法: python make_tables.py
"""

import io
import os
import csv

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')

CFG_LABEL = {
    '1k_x10M': 'VEC_SIZE = 1k, NUM_LOOP = 10M',
    '10k_x100k': 'VEC_SIZE = 10k, NUM_LOOP = 100k',
    '100k_x100k': 'VEC_SIZE = 100k, NUM_LOOP = 100k',
}
CFG_ELEM = {'1k_x10M': '1.024e10', '10k_x100k': '1.000e9', '100k_x100k': '1.000e10'}


def rd(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        return None
    with io.open(path, encoding='utf-8-sig') as fh:
        return list(csv.DictReader(fh))


def t_case(out):
    rows = rd('cases.csv')
    if not rows:
        return
    out.write('### 表 5 三种 Case 的执行时间（12 线程，scalar 内核）\n\n')
    out.write('| 配置 | 元素总数 | Case 1 (s) | Case 2 (s) | Case 3 (s) | '
              'Case 2 / Case 1 | Case 3 / Case 1 |\n')
    out.write('| --- | --- | --- | --- | --- | --- | --- |\n')
    d = {}
    for r in rows:
        d.setdefault(r['config'], {})[int(r['case'])] = float(r['time_s'])
    for cfg in ('1k_x10M', '10k_x100k', '100k_x100k'):
        if cfg not in d:
            continue
        v = d[cfg]
        out.write('| %s | %s | %.3f | %.3f | %.3f | %.2f× | %.1f× |\n' % (
            CFG_LABEL[cfg], CFG_ELEM[cfg], v[1], v[2], v[3], v[2] / v[1], v[3] / v[1]))
    out.write('\n')


def t_scale(out, csvname, title, keys, names, index):
    rows = rd(csvname)
    if not rows:
        return
    d = {}
    for r in rows:
        d.setdefault(r['prog'], {})[int(r['threads'])] = r
    ps = sorted({int(r['threads']) for r in rows})
    out.write('### 表 %s %s\n\n' % (index, title))
    out.write('| 线程数 |')
    for nm in names:
        out.write(' %s 时间 (s) | %s 加速比 |' % (nm, nm))
    out.write('\n| --- |')
    for _ in names:
        out.write(' --- | --- |')
    out.write('\n')
    for p in ps:
        out.write('| %d |' % p)
        for k in keys:
            r = d.get(k, {}).get(p)
            if r is None:
                out.write(' – | – |')
            else:
                out.write(' %.3f | %.2f× |' % (float(r['time_mean_s']),
                                               float(r['speedup'])))
        out.write('\n')
    out.write('\n')


def t_overhead(out, index):
    reg = rd('region_overhead.csv')
    coh = rd('coherence.csv')
    if not reg and not coh:
        return
    regd = {int(r['threads']): float(r['us_per_region']) for r in (reg or [])}
    cohd = {int(r['threads']): r for r in (coh or [])}
    ps = sorted(set(list(regd.keys()) + list(cohd.keys())))
    out.write('### 表 %s 并行区开销与共享写／私有写对照（VEC_SIZE = 1k, NUM_LOOP = 10M）\n\n' % index)
    out.write('| 线程数 | parallel region 开销 (µs) | 共享写 z 加速比 | 私有写 z 加速比 |\n')
    out.write('| --- | --- | --- | --- |\n')
    for p in ps:
        u = ('%.1f' % regd[p]) if p in regd else ' – '
        cs = ('%.2f×' % float(cohd[p]['shared_z_speedup'])) if p in cohd else ' – '
        pv = ('%.2f×' % float(cohd[p]['private_z_speedup'])) if p in cohd else ' – '
        out.write('| %d | %s | %s | %s |\n' % (p, u, cs, pv))
    out.write('\n')


def main():
    buf = io.StringIO()
    t_case(buf)
    t_scale(buf, 'threads_daxpy.csv', 'DAXPY 线程数扫描（外层循环并行）',
            ['daxpy_small', 'daxpy_mid', 'daxpy_big'],
            ['1k(L1)', '10k(L2)', '100k(超出L2)'], 6)
    t_scale(buf, 'threads_matmul.csv', '矩阵乘法线程数扫描（并行最外层 i）',
            ['matmul_small', 'matmul_big'], ['n=256', 'n=1024'], 7)
    t_overhead(buf, 8)

    text = buf.getvalue()
    path = os.path.join(RES, 'tables.md')
    with io.open(path, 'w', encoding='utf-8') as fh:
        fh.write(text)
    # 控制台是 GBK, 中文/µ 直接 print 会抛 UnicodeEncodeError, 写成文件再看
    print('wrote %s (%d chars)' % (path, len(text)))


if __name__ == '__main__':
    main()
