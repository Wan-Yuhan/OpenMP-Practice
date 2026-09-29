/* ==========================================================================
 *  omp_matmul.c  --  高性能计算 编程作业3  (任务2 的矩阵乘法部分)
 *
 *  矩阵乘法 C = A*B, A/B/C 均为 n×n 的 double 方阵。
 *  内核采用 ikj 顺序(行主序下的最优形式, 即把 k 提到中间、j 放最内层):
 *
 *      for (i = 0; i < n; i++)
 *          for (k = 0; k < n; k++) {
 *              aik = A[i][k];
 *              for (j = 0; j < n; j++)
 *                  C[i][j] += aik * B[k][j];
 *          }
 *
 *  该内层循环逐行连续访问 C 与 B, 每个 k 迭代就是一次 DAXPY,
 *  与任务1 的内核同源; 并行化点选在最外层 i 上 —— 不同的 i 对应 C 的不同行,
 *  各线程写的是互不相交的内存, 因此不像 DAXPY 那样存在「共享写」问题。
 *
 *  运行模式
 *    check                  正确性验证(串行 vs 并行, 含 n<=4 的完整结果)
 *    threads [pmax] [nb] [ns]   线程数扫描(默认 nb=1024, ns=256)
 *
 *  计时区自动重复若干次(以 1 线程约 0.4 s 为准自适应确定重复次数),
 *  与作业2 中「把计时区拉长」的做法一致; 各线程数轮流测量, 以削弱频率漂移。
 *
 *  编译:  gcc -O2 -fopenmp omp_matmul.c -o omp_matmul
 * ========================================================================== */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define SEED        20260924ULL
#define TARGET_SEC  0.4             /* 计时区目标时长(秒) */
#define MAX_REPEAT  500
#define MAXP        16
#define MAT_SCALE   1.0e-3          /* 让结果数值不至于过大 */

static unsigned long long lcg_state;

static double lcg_uni(void)
{
    lcg_state = lcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(lcg_state >> 11) / 9007199254740992.0;
}

static void *xmalloc(size_t n)
{
    void *p = malloc(n);

    if (p == NULL) {
        fprintf(stderr, "out of memory (%zu bytes)\n", n);
        exit(1);
    }
    return p;
}

static double *mat_alloc(long n)
{
    return (double *)xmalloc(sizeof(double) * (size_t)n * (size_t)n);
}

static void mat_fill(double *m, long n)
{
    long i, j;

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            m[i * n + j] = MAT_SCALE * lcg_uni();
}

static void mat_zero(double *m, long n)
{
    memset(m, 0, sizeof(double) * (size_t)n * (size_t)n);
}

/* ---------------- 串行内核 (ikj 顺序) ---------------- */
static void matmul_serial(double *c, const double *a, const double *b, long n)
{
    long i, k, j;

    for (i = 0; i < n; i++) {
        double *ci = c + i * n;

        for (k = 0; k < n; k++) {
            double aik = a[i * n + k];
            const double *bk = b + k * n;

            for (j = 0; j < n; j++)
                ci[j] += aik * bk[j];
        }
    }
}

/* ---------------- OpenMP 内核: 并行化最外层 i ---------------- */
static void matmul_omp(double *c, const double *a, const double *b, long n)
{
    long i, k, j;

#pragma omp parallel for
    for (i = 0; i < n; i++) {
        double *ci = c + i * n;

        for (k = 0; k < n; k++) {
            double aik = a[i * n + k];
            const double *bk = b + k * n;

            for (j = 0; j < n; j++)
                ci[j] += aik * bk[j];
        }
    }
}

/* 相对误差: max|p-q| / max(1, max|q|) */
static double rel_error(const double *p, const double *q, long n)
{
    long i, m = n * n;
    double dmax = 0.0, qmax = 0.0, d;

    for (i = 0; i < m; i++) {
        d = fabs(p[i] - q[i]);
        if (d > dmax)
            dmax = d;
        if (fabs(q[i]) > qmax)
            qmax = fabs(q[i]);
    }
    return dmax / (qmax > 1.0 ? qmax : 1.0);
}

static double mat_checksum(const double *m, long n)
{
    long i, m2 = n * n;
    double s = 0.0;

    for (i = 0; i < m2; i++)
        s += m[i];
    return s;
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

static int make_plist(int pmax, int *pl)
{
    static const int all[] = { 1, 2, 3, 4, 6, 8, 10, 12 };
    int i, n = 0;

    for (i = 0; i < (int)(sizeof(all) / sizeof(all[0])); i++)
        if (all[i] <= pmax)
            pl[n++] = all[i];
    return n;
}

/* ---------------- 正确性验证 ---------------- */
static void run_check(void)
{
    static const long sizes[] = { 1, 4, 64, 256 };
    int s;
    const int p = omp_get_max_threads();

    printf("# ---- omp_matmul check ----\n");
    printf("# omp_get_num_procs=%d  omp_get_max_threads=%d\n",
           omp_get_num_procs(), omp_get_max_threads());
    omp_set_num_threads(p);

    for (s = 0; s < (int)(sizeof(sizes) / sizeof(sizes[0])); s++) {
        long n = sizes[s];
        double *a, *b, *cs, *cp;
        int k;

        lcg_state = SEED + (unsigned long long)n;
        a  = mat_alloc(n);
        b  = mat_alloc(n);
        cs = mat_alloc(n);
        cp = mat_alloc(n);
        mat_fill(a, n);
        mat_fill(b, n);

        mat_zero(cs, n);
        matmul_serial(cs, a, b, n);

        for (k = 0; k < 3; k++) {       /* 并行版每次都应与串行版逐位一致 */
            mat_zero(cp, n);
            matmul_omp(cp, a, b, n);
        }

        printf("CHECK n=%ld threads=%d rel_error=%.3e checksum_serial=%.10e "
               "checksum_omp=%.10e %s\n",
               n, p, rel_error(cp, cs, n), mat_checksum(cs, n), mat_checksum(cp, n),
               (rel_error(cp, cs, n) == 0.0) ? "EXACT" : "DIFF");

        if (n <= 4) {                   /* 小规模完整打印, 便于人工核对 */
            long i, j;

            for (i = 0; i < n; i++) {
                printf("  C[%ld][*] =", i);
                for (j = 0; j < n; j++)
                    printf(" %.10f", cs[i * n + j]);
                printf("\n");
            }
        }
        fflush(stdout);

        free(a); free(b); free(cs); free(cp);
    }
}

/* ---------------- 线程数扫描 ---------------- */
static void sweep(const char *name, long n, const int *pl, int np, int rounds,
                  double *a, double *b, double *c, double *cref)
{
    double flop_one = 2.0 * (double)n * (double)n * (double)n;
    double tv[MAXP][16], t1_mean, t1_min, calib;
    struct stats st, st1;
    int i, r, reps = 1;
    double t0, t1, rel;

    /* 用 1 线程标定重复次数, 使计时区约 TARGET_SEC */
    omp_set_num_threads(1);
    mat_zero(c, n);
    matmul_serial(c, a, b, n);
    t0 = omp_get_wtime();
    mat_zero(c, n);
    matmul_serial(c, a, b, n);
    t1 = omp_get_wtime();
    calib = t1 - t0;
    reps = (int)(TARGET_SEC / (calib > 0.0 ? calib : 1e-9) + 0.999);
    if (reps < 1)
        reps = 1;
    if (reps > MAX_REPEAT)
        reps = MAX_REPEAT;

    printf("SWEEP %s n=%ld flops_per_mm=%.4e calib_1thread=%.6f s reps=%d "
           "timed_flops=%.4e rounds=%d\n",
           name, n, flop_one, calib, reps, flop_one * (double)reps, rounds);
    fflush(stdout);

    /* 热身 */
    for (i = 0; i < np; i++) {
        omp_set_num_threads(pl[i]);
        mat_zero(c, n);
        matmul_omp(c, a, b, n);
    }
    rel = rel_error(c, cref, n);
    printf("CHECK prog=matmul_%s rel_error=%.3e\n", name, rel);

    /* 轮流测量 */
    for (r = 0; r < rounds; r++) {
        for (i = 0; i < np; i++) {
            int k;

            omp_set_num_threads(pl[i]);
            t0 = omp_get_wtime();
            for (k = 0; k < reps; k++) {
                mat_zero(c, n);
                matmul_omp(c, a, b, n);
            }
            t1 = omp_get_wtime();
            tv[i][r] = t1 - t0;
        }
    }

    st1 = stat_of(tv[0], rounds);
    t1_mean = st1.mean;
    t1_min  = st1.min;

    for (i = 0; i < np; i++) {
        double sp, sp_min, eff;

        st = stat_of(tv[i], rounds);
        sp     = t1_mean / st.mean;
        sp_min = t1_min / st.min;
        eff    = sp / (double)pl[i];
        printf("SCALE prog=matmul_%s n=%ld threads=%d mean=%.6f min=%.6f max=%.6f "
               "sd=%.6f cv=%.1f per_mm=%.6f speedup=%.3f speedup_min=%.3f "
               "efficiency=%.3f perf=%.2f MFLOP/s\n",
               name, n, pl[i], st.mean, st.min, st.max, st.sd, st.cv,
               st.mean / reps, sp, sp_min, eff,
               flop_one * (double)reps / st.mean / 1.0e6);
        fflush(stdout);
    }
}

static void run_threads(int pmax, long nbig, long nsmall)
{
    int pl[MAXP], np, s;
    long sizes[2];
    const int rounds = 7;      /* 本机满载时波动大, 多测几轮取均值/标准差 */

    np = make_plist(pmax, pl);
    printf("# ---- omp_matmul threads ----\n");
    printf("# omp_get_num_procs=%d  omp_get_max_threads=%d  pmax=%d rounds=%d\n",
           omp_get_num_procs(), omp_get_max_threads(), pmax, rounds);

    sizes[0] = nbig;
    sizes[1] = nsmall;

    for (s = 0; s < 2; s++) {
        long n2 = sizes[s];
        double *a, *b, *c, *cref;

        lcg_state = SEED + (unsigned long long)n2;
        a    = mat_alloc(n2);
        b    = mat_alloc(n2);
        c    = mat_alloc(n2);
        cref = mat_alloc(n2);
        mat_fill(a, n2);
        mat_fill(b, n2);

        omp_set_num_threads(1);
        mat_zero(cref, n2);
        matmul_serial(cref, a, b, n2);

        sweep(s == 0 ? "big" : "small", n2, pl, np, rounds, a, b, c, cref);

        free(a); free(b); free(c); free(cref);
    }
}

/* ------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    const char *mode = (argc >= 2) ? argv[1] : "";

    if (strcmp(mode, "check") == 0) {
        run_check();
        printf("DONE check\n");
        return 0;
    }

    if (strcmp(mode, "threads") == 0) {
        int p = omp_get_num_procs();
        long nbig = 1024L, nsmall = 256L;

        if (argc >= 3)
            p = atoi(argv[2]);
        if (argc >= 4)
            nbig = atol(argv[3]);
        if (argc >= 5)
            nsmall = atol(argv[4]);
        if (p < 1)
            p = 1;
        if (p > MAXP)
            p = MAXP;
        run_threads(p, nbig, nsmall);
        printf("DONE threads\n");
        return 0;
    }

    fprintf(stderr, "usage: %s check | threads [pmax] [nbig] [nsmall]\n", argv[0]);
    return 1;
}
