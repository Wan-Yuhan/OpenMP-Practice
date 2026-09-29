#!/usr/bin/env bash
# ==========================================================================
#  run_cases.sh  --  高性能计算 编程作业3  任务1
#  三种 Case(串行 / 外层并行 / 内层并行) × 三种 (数组长度, 循环次数)
#
#  注意: Case 3 在 1k×10M 这一组需要建立 10^7 个 parallel region,
#        本机会跑十几分钟, 属正常现象(这正是该 Case 的结论所在)。
#
#  CFLAGS 说明:
#    -O2 下 GCC 只会把「串行写的」内层循环自动向量化(Case 1 用 SSE2 每次 2 个
#    double 再展开 2 次), 而经过 OpenMP 拐弯的 Case 2/3 内层循环保持标量 ——
#    因为 parallel 区拐弯后 GCC 丢掉了数组指针的不相交信息。为让三种 Case 的
#    机器码可比, 本脚本默认加 -fno-tree-vectorize 把三者都压成标量。
#    想复现「-O2 默认」那一批, 用 CFLAGS="-O2 -fopenmp" 再跑一次即可。
#  用法: bash run_cases.sh
# ==========================================================================
set -u
cd "$(dirname "$0")"

GCC=${GCC:-/c/mingw64/bin/gcc.exe}
if [ ! -x "$GCC" ]; then GCC=gcc; fi
export PATH="/c/mingw64/bin:$PATH"

OUT=results/raw
mkdir -p "$OUT"
LOG="${LOG:-$OUT/run_cases.txt}"
NTHREADS=${NTHREADS:-12}
CFLAGS=${CFLAGS:--O2 -fopenmp -fno-tree-vectorize}

battery() { powershell -NoProfile -Command "(Get-CimInstance Win32_Battery).BatteryStatus" 2>/dev/null | tr -d '\r\n '; }
clockmhz() { powershell -NoProfile -Command "(Get-CimInstance Win32_Processor).CurrentClockSpeed" 2>/dev/null | tr -d '\r\n '; }
cpuname() { powershell -NoProfile -Command "(Get-CimInstance Win32_Processor).Name" 2>/dev/null | tr -d '\r\n'; }

# 以「高」优先级运行: 本机后台常驻微信/浏览器, 不提高优先级则满载测量离散度极大
run_prio() {
    "$@" &
    local bg=$!
    sleep 1
    powershell -NoProfile -Command \
        "Get-Process -Name $1 -ErrorAction SilentlyContinue | ForEach-Object { \$_.PriorityClass = 'High' }" \
        >/dev/null 2>&1
    wait $bg
}

echo "== build =="
# shellcheck disable=SC2086
"$GCC" $CFLAGS -Wall -o omp_daxpy.exe omp_daxpy.c || exit 1

{
    echo "# ============================================================"
    echo "# run_cases.sh    $(date '+%Y-%m-%d %H:%M:%S')"
    echo "# cpu            = $(cpuname)"
    echo "# compiler       = $("$GCC" --version | head -1)"
    echo "# build flags    = $CFLAGS"
    echo "# battery_status = $(battery)   (1=电池供电, 2=接电源)"
    echo "# clock_mhz      = $(clockmhz)"
    echo "# process priority = High"
    echo "# nthreads       = $NTHREADS"
    echo "# ============================================================"
    echo
    echo "########## verify ##########"
    run_prio ./omp_daxpy.exe verify
    echo
    echo "########## cases ##########"
    run_prio ./omp_daxpy.exe cases "$NTHREADS"
    echo
    echo "# battery_status_after = $(battery)"
    echo "# clock_mhz_after      = $(clockmhz)"
    echo "# finished $(date '+%Y-%m-%d %H:%M:%S')"
} 2>&1 | tee "$LOG"

echo "LOG written to $LOG"
