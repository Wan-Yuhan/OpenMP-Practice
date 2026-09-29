#!/usr/bin/env bash
# ==========================================================================
#  run_threads.sh  --  高性能计算 编程作业3  任务2
#  用 OpenMP 运行时库设置线程数, 考察线程数对执行时间的影响。
#    (1) DAXPY 线程数扫描(小/中/大三档工作集)
#    (2) 矩阵乘法 线程数扫描(n=1024 与 n=256)
#    (3) 一个 parallel region 的建立+销毁开销
#    (4) 对照实验: 共享写 z vs 每线程私有 z 分片
#  用法: bash run_threads.sh
# ==========================================================================
set -u
cd "$(dirname "$0")"

GCC=${GCC:-/c/mingw64/bin/gcc.exe}
if [ ! -x "$GCC" ]; then GCC=gcc; fi
export PATH="/c/mingw64/bin:$PATH"

OUT=results/raw
mkdir -p "$OUT"
PMAX=${PMAX:-12}

battery() { powershell -NoProfile -Command "(Get-CimInstance Win32_Battery).BatteryStatus" 2>/dev/null | tr -d '\r\n '; }
clockmhz() { powershell -NoProfile -Command "(Get-CimInstance Win32_Processor).CurrentClockSpeed" 2>/dev/null | tr -d '\r\n '; }
cpuname() { powershell -NoProfile -Command "(Get-CimInstance Win32_Processor).Name" 2>/dev/null | tr -d '\r\n'; }

# 以「高」优先级运行, 并让 Windows 把定时器精度提到 0.5 ms;
# 本机后台常驻微信/浏览器等, 不提高优先级则满载测量的离散度极大。
run_prio() {
    "$@" &
    local bg=$!
    sleep 1
    powershell -NoProfile -Command \
        "Get-Process -Name $1 -ErrorAction SilentlyContinue | ForEach-Object { \$_.PriorityClass = 'High' }" \
        >/dev/null 2>&1
    wait $bg
}

header() {
    echo "# ============================================================"
    echo "# $1    $(date '+%Y-%m-%d %H:%M:%S')"
    echo "# cpu            = $(cpuname)"
    echo "# compiler       = $("$GCC" --version | head -1)"
    echo "# build flags    = -O2 -fopenmp  (内核本身为标量, 见 run_cases.sh 说明)"
    echo "# battery_status = $(battery)   (1=电池供电, 2=接电源)"
    echo "# clock_mhz      = $(clockmhz)"
    echo "# process priority = High"
    echo "# pmax           = $PMAX"
    echo "# omp_wait_policy= ${OMP_WAIT_POLICY:-<default>}"
    echo "# ============================================================"
}

echo "== build =="
CF=${CFLAGS:--O2 -fopenmp}
"$GCC" $CF -Wall -o omp_daxpy.exe  omp_daxpy.c  || exit 1
"$GCC" $CF -Wall -o omp_matmul.exe omp_matmul.c || exit 1

echo "== (1) daxpy thread sweep =="
{ header "run_threads_daxpy.sh"; run_prio ./omp_daxpy.exe threads "$PMAX"; } 2>&1 | tee "$OUT/run_threads_daxpy.txt"

echo "== (2) matmul thread sweep =="
{ header "run_threads_matmul.sh"; run_prio ./omp_matmul.exe check; run_prio ./omp_matmul.exe threads "$PMAX" 1024 256; } 2>&1 | tee "$OUT/run_threads_matmul.txt"

echo "== (3) parallel region overhead =="
{ header "run_region.sh"; run_prio ./omp_daxpy.exe region "$PMAX"; } 2>&1 | tee "$OUT/run_region.txt"

echo "== (4) coherence control experiment =="
{ header "run_coherence.sh"; run_prio ./omp_daxpy.exe coherence "$PMAX"; } 2>&1 | tee "$OUT/run_coherence.txt"

echo "== (5) clock frequency under 1-thread and 12-thread load =="
{
    header "run_freq.sh"
    for T in 1 12; do
        ./omp_daxpy.exe threads "$T" >/dev/null 2>&1 &
        bg=$!
        sleep 1
        powershell -NoProfile -Command \
            "Get-Process -Name omp_daxpy -ErrorAction SilentlyContinue | ForEach-Object { \$_.PriorityClass = 'High' }" \
            >/dev/null 2>&1
        sleep 2
        for k in 1 2 3 4 5; do
            echo "FREQ threads=$T sample=$k clock_mhz=$(clockmhz)"
            sleep 1
        done
        kill $bg 2>/dev/null
        powershell -NoProfile -Command \
            "Get-Process omp_daxpy -ErrorAction SilentlyContinue | Stop-Process -Force" >/dev/null 2>&1
        wait $bg 2>/dev/null
    done
} 2>&1 | tee "$OUT/run_freq.txt"

echo
echo "battery_status_after = $(battery)   clock_mhz_after = $(clockmhz)"
echo "logs written to $OUT/"
