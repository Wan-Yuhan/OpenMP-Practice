#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define SEED         20260924ULL    /* 固定种子, 保证每次运行数据完全一致 */
#define A_SCALAR     2.0            /* z = a*x + y 中的标量 a */
#define REGION_PROBE 20000L         /* 测并行区开销量用的区域个数 */
#define MAXP         16

static unsigned long long lcg_state;/* 固定种子的线性同余发生器: 生成 [1, 100] 的随机浮点数 */

static double lcg_next(void)
{
    lcg_state = lcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return 1.0 + 99.0 * ((double)(lcg_state >> 11) / 9007199254740992.0);
}

static void fill_vectors(double *x, double *y, long vlen)
{
    long j;

    lcg_state = SEED;
    for (j = 0; j < vlen; j++) {
        x[j] = lcg_next();
        y[j] = lcg_next();
    }
}

/* ------------------------------------------------------------------ */
/* 四个内核                                                            */
/* ------------------------------------------------------------------ */

/* Case 1: 不开多线程 */
static void kernel_case1(double *z, const double *x, const double *y,
                         long vlen, long nloop)
{
    long i, j;

    for (i = 0; i < nloop; i++)
        for (j = 0; j < vlen; j++)
            z[j] = A_SCALAR * x[j] + y[j];
}

/* Case 2: 外层循环(i)多线程 —— 整段只有一个 parallel region */
static void kernel_case2(double *z, const double *x, const double *y,
                         long vlen, long nloop)
{
    long i, j;

#pragma omp parallel for
    for (i = 0; i < nloop; i++)
        for (j = 0; j < vlen; j++)
            z[j] = A_SCALAR * x[j] + y[j];
}

/* Case 2': 外层循环多线程, 但每个线程写自己私有的 z 分片
 *          (仅用于对照实验, 隔离「所有线程写同一个 z」这一因素) */
static void kernel_case2p(double *zbuf, const double *x, const double *y,
                          long vlen, long nloop)
{
#pragma omp parallel
    {
        int nt  = omp_get_num_threads();
        int tid = omp_get_thread_num();
        double *mz = zbuf + (size_t)tid * (size_t)vlen;
        long lo = nloop * tid / nt;
        long hi = nloop * (tid + 1) / nt;
        long i, j;

        for (i = lo; i < hi; i++)
            for (j = 0; j < vlen; j++)
                mz[j] = A_SCALAR * x[j] + y[j];
    }
}

/* Case 3: 内层循环(j)多线程 —— 外层每转一圈都要建/拆一次 parallel region */
static void kernel_case3(double *z, const double *x, const double *y,
                         long vlen, long nloop)
{
    long i, j;

    for (i = 0; i < nloop; i++) {
#pragma omp parallel for
        for (j = 0; j < vlen; j++)
            z[j] = A_SCALAR * x[j] + y[j];
    }
}

/* ------------------------------------------------------------------ */
/* 工具                                                                */
/* ------------------------------------------------------------------ */

static void *xmalloc(size_t n)
{
    void *p = malloc(n);

    if (p == NULL) {
        fprintf(stderr, "out of memory (%zu bytes)\n", n);
        exit(1);
    }
    return p;
}

static double checksum(const double *z, long vlen)
{
    long j;
    double s = 0.0;

    for (j = 0; j < vlen; j++)
        s += z[j];
    return s;
}

static double max_abs_diff(const double *p, const double *q, long vlen)
{
    long j;
    double m = 0.0, d;

    for (j = 0; j < vlen; j++) {
        d = fabs(p[j] - q[j]);
        if (d > m)
            m = d;
    }
    return m;
}

/* 计时器分辨率: 连续两次 omp_get_wtime() 的最小非零差值 */
static double timer_resolution(void)
{
    double t0, t1, best = 1e30;
    int k;

    for (k = 0; k < 1000; k++) {
        t0 = omp_get_wtime();
        do {
            t1 = omp_get_wtime();
        } while (t1 == t0);
        if (t1 - t0 < best)
            best = t1 - t0;
    }
    return best;
}

struct stats {
    double mean, min, max, sd, cv;
};

static struct stats stat_of(const double *v, int n)
{
    struct stats s;
    double sum = 0.0;
    int i;

    s.min = v[0];
    s.max = v[0];
    for (i = 0; i < n; i++) {
        sum += v[i];
        if (v[i] < s.min) s.min = v[i];
        if (v[i] > s.max) s.max = v[i];
    }
    s.mean = sum / n;
    s.sd = 0.0;
    if (n > 1) {
        for (i = 0; i < n; i++)
            s.sd += (v[i] - s.mean) * (v[i] - s.mean);
        s.sd = sqrt(s.sd / (n - 1));
    }
    s.cv = (s.mean > 0.0) ? s.sd / s.mean * 100.0 : 0.0;
    return s;
}

/* 线程数列表 */
static int make_plist(int pmax, int *pl)
{
    static const int all[] = { 1, 2, 3, 4, 6, 8, 10, 12 };
    int i, n = 0;

    for (i = 0; i < (int)(sizeof(all) / sizeof(all[0])); i++)
        if (all[i] <= pmax)
            pl[n++] = all[i];
    return n;
}

/* ------------------------------------------------------------------ */
/* 正确性验证                                                          */
/* ------------------------------------------------------------------ */
static void run_verify(void)
{
    const int p = omp_get_max_threads();

    printf("# ---- omp_daxpy verify ----\n");
    printf("# omp_get_num_procs=%d  omp_get_max_threads=%d\n",
           omp_get_num_procs(), omp_get_max_threads());

    /* 小规模: 逐元素打印, 供手工核对 (x 用固定公式便于手算) */
    {
        const long vlen = 4, nloop = 3;
        double x[4], y[4], z[4], z2[4], z3[4];
        long j;

        for (j = 0; j < vlen; j++) {
            x[j] = 10.0 + (double)j;
            y[j] = 1.0 + 0.5 * (double)j;
        }
        memset(z,  0, sizeof(z));
        memset(z2, 0, sizeof(z2));
        memset(z3, 0, sizeof(z3));

        omp_set_num_threads(1);
        kernel_case1(z, x, y, vlen, nloop);
        omp_set_num_threads(p);
        kernel_case2(z2, x, y, vlen, nloop);
        kernel_case3(z3, x, y, vlen, nloop);

        printf("VERIFY_SMALL vec_size=%ld nloop=%ld a=%.1f\n", vlen, nloop, A_SCALAR);
        for (j = 0; j < vlen; j++)
            printf("  j=%ld  x=%.1f  y=%.1f  a*x+y=%.1f  case1=%.1f  case2=%.1f  case3=%.1f  %s\n",
                   j, x[j], y[j], A_SCALAR * x[j] + y[j], z[j], z2[j], z3[j],
                   (z[j] == z2[j] && z[j] == z3[j]) ? "OK" : "MISMATCH");
        printf("VERIFY_SMALL maxabs case2=%d case3=%d\n",
               (int)max_abs_diff(z2, z, vlen), (int)max_abs_diff(z3, z, vlen));
    }

    /* 大规模: 三种 Case 的结果必须逐位一致 */
    {
        long sizes[] = { 1024L, 10000L, 100000L };
        int s;

        for (s = 0; s < 3; s++) {
            long vlen = sizes[s];
            long nloop = 1000L;
            double *x, *y, *z, *z2, *z3;

            x  = (double *)xmalloc(sizeof(double) * (size_t)vlen);
            y  = (double *)xmalloc(sizeof(double) * (size_t)vlen);
            z  = (double *)xmalloc(sizeof(double) * (size_t)vlen);
            z2 = (double *)xmalloc(sizeof(double) * (size_t)vlen);
            z3 = (double *)xmalloc(sizeof(double) * (size_t)vlen);
            fill_vectors(x, y, vlen);

            omp_set_num_threads(1);
            kernel_case1(z, x, y, vlen, nloop);
            omp_set_num_threads(p);
            kernel_case2(z2, x, y, vlen, nloop);
            kernel_case3(z3, x, y, vlen, nloop);

            printf("VERIFY vec_size=%ld nloop=%ld threads=%d "
                   "maxabs_case2=%.3e maxabs_case3=%.3e checksum=%.10e\n",
                   vlen, nloop, p, max_abs_diff(z2, z, vlen),
                   max_abs_diff(z3, z, vlen), checksum(z, vlen));
            fflush(stdout);

            free(x); free(y); free(z); free(z2); free(z3);
        }
    }
}

/* ------------------------------------------------------------------ */
/* 任务1: 三种 Case × 三种 (VEC_SIZE, NUM_LOOP) 组合                     */
/* ------------------------------------------------------------------ */
struct config {
    long        vlen;
    long        nloop;
    const char *name;
};

static const struct config CASES[] = {
    {   1024L,   10000000L, "1k_x10M"    },
    {  10000L,     100000L, "10k_x100k"  },
    { 100000L,     100000L, "100k_x100k" },
};

static void run_cases(int nthreads)
{
    const int ncfg = (int)(sizeof(CASES) / sizeof(CASES[0]));
    const int ROUND = 3;            /* Case 1/2 重复 3 次取最小; Case 3 只跑一次 */
    int c;

    printf("# ---- omp_daxpy cases ----\n");
    printf("# omp_get_num_procs=%d  omp_get_max_threads=%d  nthreads=%d\n",
           omp_get_num_procs(), omp_get_max_threads(), nthreads);
    printf("# timer_resolution=%.3e s  A_SCALAR=%.1f  SEED=%llu\n",
           timer_resolution(), A_SCALAR, (unsigned long long)SEED);
    fflush(stdout);

    omp_set_num_threads(nthreads);

    for (c = 0; c < ncfg; c++) {
        long vlen  = CASES[c].vlen;
        long nloop = CASES[c].nloop;
        double nflop = 2.0 * (double)vlen * (double)nloop;
        double *x, *y, *z, *zref, *z1;
        double tv[8], t0, t1, t2, t3, sum;
        char tag[80];
        int k;

        x    = (double *)xmalloc(sizeof(double) * (size_t)vlen);
        y    = (double *)xmalloc(sizeof(double) * (size_t)vlen);
        z    = (double *)xmalloc(sizeof(double) * (size_t)vlen);
        zref = (double *)xmalloc(sizeof(double) * (size_t)vlen);
        z1   = (double *)xmalloc(sizeof(double) * (size_t)vlen);

        fill_vectors(x, y, vlen);

        /* 注意: Windows/LLP64 下 long 是 32 位, vlen*nloop 会溢出, 必须用 double 打印 */
        printf("CONFIG %s vec_size=%ld nloop=%ld elements=%.4e flops=%.4e nthreads=%d\n",
               CASES[c].name, vlen, nloop, (double)vlen * (double)nloop, nflop, nthreads);
        fflush(stdout);

        /* ---- Case 1: 串行; 结果作为基准 ---- */
        omp_set_num_threads(1);
        kernel_case1(zref, x, y, vlen, nloop);          /* 热身, 同时得到基准结果 */
        for (k = 0; k < ROUND; k++) {
            t0 = omp_get_wtime();
            kernel_case1(z1, x, y, vlen, nloop);
            tv[k] = omp_get_wtime() - t0;
        }
        t1 = tv[0];
        for (k = 1; k < ROUND; k++)
            if (tv[k] < t1) t1 = tv[k];
        sum = checksum(z1, vlen);
        sprintf(tag, "RESULT config=%s case=1 threads=1", CASES[c].name);
        printf("%s time=%.6f flops=%.4e perf=%.2f MFLOP/s checksum=%.10e\n",
               tag, t1, nflop, nflop / t1 / 1.0e6, sum);
        sprintf(tag, "RESULT_DETAIL config=%s case=1", CASES[c].name);
        printf("%s rounds=%d", tag, ROUND);
        for (k = 0; k < ROUND; k++)
            printf(" t%d=%.6f", k + 1, tv[k]);
        printf("\n");
        printf("CHECK config=%s case=1 vs serial_reference maxabs=%.3e\n",
               CASES[c].name, max_abs_diff(z1, zref, vlen));
        fflush(stdout);

        /* ---- Case 2: 外层循环多线程 ---- */
        omp_set_num_threads(nthreads);
        for (k = 0; k < ROUND; k++) {
            t0 = omp_get_wtime();
            kernel_case2(z1, x, y, vlen, nloop);
            tv[k] = omp_get_wtime() - t0;
        }
        t2 = tv[0];
        for (k = 1; k < ROUND; k++)
            if (tv[k] < t2) t2 = tv[k];
        sum = checksum(z1, vlen);
        sprintf(tag, "RESULT config=%s case=2 threads=%d", CASES[c].name, nthreads);
        printf("%s time=%.6f flops=%.4e perf=%.2f MFLOP/s checksum=%.10e\n",
               tag, t2, nflop, nflop / t2 / 1.0e6, sum);
        sprintf(tag, "RESULT_DETAIL config=%s case=2", CASES[c].name);
        printf("%s rounds=%d", tag, ROUND);
        for (k = 0; k < ROUND; k++)
            printf(" t%d=%.6f", k + 1, tv[k]);
        printf("\n");
        printf("CHECK config=%s case=2 vs case=1 maxabs=%.3e\n",
               CASES[c].name, max_abs_diff(z1, zref, vlen));
        fflush(stdout);

        /* ---- Case 3: 内层循环多线程 ---- */
        t0 = omp_get_wtime();
        kernel_case3(z1, x, y, vlen, nloop);
        t3 = omp_get_wtime() - t0;
        sum = checksum(z1, vlen);
        sprintf(tag, "RESULT config=%s case=3 threads=%d", CASES[c].name, nthreads);
        printf("%s time=%.6f flops=%.4e perf=%.2f MFLOP/s checksum=%.10e\n",
               tag, t3, nflop, nflop / t3 / 1.0e6, sum);
        printf("CHECK config=%s case=3 vs case=1 maxabs=%.3e\n",
               CASES[c].name, max_abs_diff(z1, zref, vlen));
        printf("SPEEDUP config=%s case2_over_case1=%.3f case1_over_case3=%.3f\n",
               CASES[c].name, t1 / t2, t3 / t1);
        fflush(stdout);

        free(x); free(y); free(z); free(zref); free(z1);
    }
}

/* ------------------------------------------------------------------ */
/* 任务2: 线程数扫描                                                    */
/* ------------------------------------------------------------------ */
static void sweep_daxpy(const char *name, long vlen, long nloop,
                        const int *pl, int np, int rounds)
{
    double *x, *y, *z;
    double nflop = 2.0 * (double)vlen * (double)nloop;
    double tv[MAXP][16], t1_mean, t1_min;
    struct stats st, st1;
    int i, r;

    x = (double *)xmalloc(sizeof(double) * (size_t)vlen);
    y = (double *)xmalloc(sizeof(double) * (size_t)vlen);
    z = (double *)xmalloc(sizeof(double) * (size_t)vlen);
    fill_vectors(x, y, vlen);

    printf("SWEEP %s vec_size=%ld nloop=%ld flops=%.4e rounds=%d\n",
           name, vlen, nloop, nflop, rounds);
    fflush(stdout);

    /* 先各热身一遍 */
    for (i = 0; i < np; i++) {
        omp_set_num_threads(pl[i]);
        kernel_case2(z, x, y, vlen, nloop);
    }

    /* 轮流测量: 每一轮都按同样的顺序访问所有线程数, 使频率漂移对各方等价 */
    for (r = 0; r < rounds; r++) {
        for (i = 0; i < np; i++) {
            double t0;

            omp_set_num_threads(pl[i]);
            t0 = omp_get_wtime();
            kernel_case2(z, x, y, vlen, nloop);
            tv[i][r] = omp_get_wtime() - t0;
        }
    }

    st1 = stat_of(tv[0], rounds);       /* 1 线程作为基准 */
    t1_mean = st1.mean;
    t1_min  = st1.min;

    for (i = 0; i < np; i++) {
        double sp, sp_min, eff;

        st = stat_of(tv[i], rounds);
        sp     = t1_mean / st.mean;
        sp_min = t1_min / st.min;
        eff    = sp / (double)pl[i];
        printf("SCALE prog=daxpy_%s threads=%d mean=%.6f min=%.6f max=%.6f sd=%.6f "
               "cv=%.1f speedup=%.3f speedup_min=%.3f efficiency=%.3f "
               "perf=%.2f MFLOP/s\n",
               name, pl[i], st.mean, st.min, st.max, st.sd, st.cv,
               sp, sp_min, eff, nflop / st.mean / 1.0e6);
        fflush(stdout);
    }

    printf("CHECK prog=daxpy_%s checksum=%.10e\n", name, checksum(z, vlen));
    free(x); free(y); free(z);
}

/* 一个空的 parallel region 的建立+销毁开销, 单位微秒 */
static double region_overhead(int p)
{
    double t0, t1;
    long i;
    int hit = 0;

    omp_set_num_threads(p);
#pragma omp parallel
    { if (omp_get_thread_num() < 0) hit = 1; }

    t0 = omp_get_wtime();
    for (i = 0; i < REGION_PROBE; i++) {
#pragma omp parallel num_threads(p)
        {
            if (omp_get_thread_num() < 0)
                hit = 1;
        }
    }
    t1 = omp_get_wtime();

    if (hit)
        printf("# unexpected negative thread id\n");
    return (t1 - t0) / (double)REGION_PROBE * 1.0e6;
}

static void run_region(int pmax)
{
    int pl[MAXP], np, i;

    np = make_plist(pmax, pl);
    printf("# ---- omp_daxpy region ----\n");
    printf("# probe=%ld\n", REGION_PROBE);
    for (i = 0; i < np; i++)
        printf("REGION_OVERHEAD threads=%d us=%.3f\n", pl[i], region_overhead(pl[i]));
    fflush(stdout);
}

static void run_threads(int pmax)
{
    int pl[MAXP], np;

    np = make_plist(pmax, pl);
    printf("# ---- omp_daxpy threads ----\n");
    printf("# omp_get_num_procs=%d  omp_get_max_threads=%d  pmax=%d\n",
           omp_get_num_procs(), omp_get_max_threads(), pmax);

    /* (a) 小规模: 工作集 24 KB(x+y+z), 常驻 L1 */
    sweep_daxpy("small", 1024L, 10000000L, pl, np, 5);
    /* (b) 中等规模: 工作集 240 KB, 常驻 L2 */
    sweep_daxpy("mid", 10000L, 1000000L, pl, np, 5);
    /* (c) 大规模: 工作集 2.4 MB, 超出 L2, 受共享 L3/内存带宽限制 */
    sweep_daxpy("big", 100000L, 100000L, pl, np, 5);
}

/* ------------------------------------------------------------------ */
/* 对照实验: 共享写 z vs 每线程写私有分片                                  */
/* ------------------------------------------------------------------ */
static void run_coherence(int pmax)
{
    const long vlen = 1024L, nloop = 10000000L;
    const int ROUND = 3;
    int pl[MAXP], np, i, r;
    double *x, *y, *z, *zbuf, tv[MAXP][8], ts_serial, ts_shared, ts_priv;
    struct stats st;

    np = make_plist(pmax, pl);
    x    = (double *)xmalloc(sizeof(double) * (size_t)vlen);
    y    = (double *)xmalloc(sizeof(double) * (size_t)vlen);
    z    = (double *)xmalloc(sizeof(double) * (size_t)vlen);
    zbuf = (double *)xmalloc(sizeof(double) * (size_t)vlen * (size_t)MAXP);
    fill_vectors(x, y, vlen);

    printf("# ---- omp_daxpy coherence ----\n");
    printf("# vec_size=%ld nloop=%ld rounds=%d  (唯一变量: z 是共享还是每线程私有)\n",
           vlen, nloop, ROUND);

    omp_set_num_threads(1);
    kernel_case1(z, x, y, vlen, nloop);
    for (r = 0; r < ROUND; r++) {
        double t0 = omp_get_wtime();
        kernel_case1(z, x, y, vlen, nloop);
        tv[0][r] = omp_get_wtime() - t0;
    }
    st = stat_of(tv[0], ROUND);
    ts_serial = st.mean;
    printf("SERIAL threads=1 mean=%.6f min=%.6f\n", st.mean, st.min);

    for (i = 0; i < np; i++) {
        int p = pl[i];

        omp_set_num_threads(p);
        kernel_case2(z, x, y, vlen, nloop);
        kernel_case2p(zbuf, x, y, vlen, nloop);
        for (r = 0; r < ROUND; r++) {
            double t0 = omp_get_wtime();
            kernel_case2(z, x, y, vlen, nloop);
            tv[i][r] = omp_get_wtime() - t0;
        }
        st = stat_of(tv[i], ROUND);
        ts_shared = st.mean;
        for (r = 0; r < ROUND; r++) {
            double t0 = omp_get_wtime();
            kernel_case2p(zbuf, x, y, vlen, nloop);
            tv[i][r] = omp_get_wtime() - t0;
        }
        st = stat_of(tv[i], ROUND);
        ts_priv = st.mean;

        printf("COHERENCE threads=%d shared_z=%.6f speedup=%.3f | private_z=%.6f "
               "speedup=%.3f\n", p, ts_shared, ts_serial / ts_shared,
               ts_priv, ts_serial / ts_priv);
        fflush(stdout);
    }

    free(x); free(y); free(z); free(zbuf);
}

/* ------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    const char *mode = (argc >= 2) ? argv[1] : "";
    int p = omp_get_num_procs();

    if (argc >= 3)
        p = atoi(argv[2]);
    if (p < 1)
        p = 1;
    if (p > MAXP)
        p = MAXP;

    if (strcmp(mode, "cases") == 0) {
        run_cases(p);
        printf("DONE cases\n");
        return 0;
    }
    if (strcmp(mode, "threads") == 0) {
        run_threads(p);
        printf("DONE threads\n");
        return 0;
    }
    if (strcmp(mode, "region") == 0) {
        run_region(p);
        printf("DONE region\n");
        return 0;
    }
    if (strcmp(mode, "coherence") == 0) {
        run_coherence(p);
        printf("DONE coherence\n");
        return 0;
    }
    if (strcmp(mode, "verify") == 0) {
        run_verify();
        printf("DONE verify\n");
        return 0;
    }

    fprintf(stderr,
            "usage: %s {cases|threads|region|coherence|verify} [pmax]\n", argv[0]);
    return 1;
}
