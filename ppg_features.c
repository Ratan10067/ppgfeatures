#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#define HP_ORDER 2
#define HP_NCOEF (HP_ORDER + 1)
#define M_PI 3.14159265358979323846
#define BP_ORDER 4
#define BP_NCOEF 5
double *read_csv(const char *filename, size_t *out_len)
{
    FILE *f = fopen(filename, "r");
    if (!f)
    {
        fprintf(stderr, "Error: cannot open '%s'\n", filename);
        exit(1);
    }
    size_t capacity = 1024;
    size_t len = 0;
    double *buffer = (double *)malloc(sizeof(double) * capacity);
    if (!buffer)
    {
        fprintf(stderr, "Error: malloc failed\n");
        exit(1);
    }
    char line[256];
    while (fgets(line, sizeof(line), f))
    {
        // Skip empty lines
        char *p = line;
        while (*p == ' ' || *p == '\t')
            p++;
        if (*p == '\n' || *p == '\0')
            continue;
        double v = strtod(line, NULL);
        if (len >= capacity)
        {
            capacity *= 2;
            buffer = (double *)realloc(buffer, sizeof(double) * capacity);
            if (!buffer)
            {
                fprintf(stderr, "Error: realloc failed\n");
                exit(1);
            }
        }
        buffer[len++] = v;
    }
    fclose(f);
    *out_len = len;
    for (int i = 0; i < 10; i++)
    {
        printf("%f\n", buffer[i]);
    }
    return buffer;
}

/*
   Chebyshev‐Type I filter coefficients (order=2, ripple=0.5 dB):
   – High‐pass: cutoff=0.5 Hz @ fs=100 Hz
     b_hp = [ 0.93017014,  -1.86034028,  0.93017014 ]
     a_hp = [ 1.0,        -1.97025163,  0.97089310 ]

   – Band‐pass: passband=0.5–5 Hz @ fs=100 Hz
     b_bp = [ 0.02350275,  0.0,  -0.04700550,  0.0,  0.02350275 ]
     a_bp = [ 1.0,  -3.55374217,  4.78188186,  -2.89911171,  0.67105192 ]
*/

static const double b_hp[HP_NCOEF] = {0.93017014, -1.86034028, 0.93017014};
static const double a_hp[HP_NCOEF] = {1.0000000, -1.97025163, 0.9708931};

static const double b_bp[BP_NCOEF] = {0.02350275, 0.0000, -0.0470055, 0.0000, 0.02350275};
static const double a_bp[BP_NCOEF] = {1.000, -3.55374217, 4.78188186, -2.89911171, 0.67105192};

/*
   Apply a single‐stage IIR filter (direct‐form I) to input "x"
   with coefficients (b[0..N], a[0..N]) → output "y".
   We assume a[0] = 1.0.  Length = Ndata.
*/

void iir_filter_df2t(const double *b, const double *a, int order,
                     const double *x, double *y, size_t Ndata)
{
    // Allocate and initialize state variables
    double *state = (double *)calloc(order, sizeof(double));
    if (!state)
    {
        fprintf(stderr, "Error: calloc failed\n");
        exit(1);
    }

    for (size_t n = 0; n < Ndata; n++)
    {
        double xn = x[n];
        double yn = b[0] * xn + state[0];
        y[n] = yn;

        // Update state variables
        for (int i = 0; i < order - 1; i++)
        {
            state[i] = b[i + 1] * xn + state[i + 1] - a[i + 1] * yn;
        }
        state[order - 1] = b[order] * xn - a[order] * yn;
    }
    free(state);
}

/* ----------------------------------------------------------
 *  Improved filtfilt implementation with proper state handling
 * ---------------------------------------------------------- */
void filtfilt_cheby(const double *b, const double *a, int Ncoef,
                    const double *x, double *y, size_t Ndata)
{
    int order = Ncoef - 1;
    int padlen = 3 * order;
    size_t Npad = Ndata + 2 * padlen;

    // Allocate padded arrays
    double *xp = (double *)malloc(Npad * sizeof(double));
    double *yp = (double *)malloc(Npad * sizeof(double));
    double *yr = (double *)malloc(Npad * sizeof(double));

    if (!xp || !yp || !yr)
    {
        fprintf(stderr, "Error: malloc failed in filtfilt\n");
        exit(1);
    }

    // Create padded signal with reflected boundaries
    for (int i = 0; i < padlen; i++)
    {
        xp[i] = 2.0 * x[0] - x[padlen - i];
        xp[Npad - 1 - i] = 2.0 * x[Ndata - 1] - x[Ndata - 2 - i];
    }
    memcpy(xp + padlen, x, Ndata * sizeof(double));

    // Forward filter
    iir_filter_df2t(b, a, order, xp, yp, Npad);

    // Reverse filtered signal
    for (size_t i = 0; i < Npad; i++)
    {
        yr[i] = yp[Npad - 1 - i];
    }

    // Backward filter
    iir_filter_df2t(b, a, order, yr, yp, Npad);

    // Reverse result and extract central part
    for (size_t i = 0; i < Ndata; i++)
    {
        y[i] = yp[Npad - 1 - (i + padlen)];
    }

    // Cleanup
    free(xp);
    free(yp);
    free(yr);
}

/* ----------------------------------------------------------
 *  Normalize an array of length N to [0,1]: (x - min)/(max - min).
 * ---------------------------------------------------------- */
void normalize01(double *x, size_t N)
{
    double mn = x[0], mx = x[0];
    for (size_t i = 1; i < N; i++)
    {
        if (x[i] < mn)
            mn = x[i];
        if (x[i] > mx)
            mx = x[i];
    }
    double range = mx - mn;
    if (range < 1e-15)
        range = 1e-15;
    for (size_t i = 0; i < N; i++)
    {
        x[i] = (x[i] - mn) / range;
    }
}

/* ----------------------------------------------------------
 *  Compute "kte" array: kte[i] = ppg[i]^2 - ppg[i-1]*ppg[i+1]
 *  for 1 ≤ i < N-1; kte[0]=kte[N-1]=0.
 * ---------------------------------------------------------- */
void compute_kte(const double *ppg, double *kte, size_t N)
{
    if (N == 0)
        return;
    kte[0] = 0.0;
    for (size_t i = 1; i + 1 < N; i++)
    {
        kte[i] = ppg[i] * ppg[i] - ppg[i - 1] * ppg[i + 1];
    }
    kte[N - 1] = 0.0;
}

/* ----------------------------------------------------------
 *  Basic statistics: mean, variance, skewness, kurtosis:
 *   – mean = (1/N) ∑ x[i]
 *   – var  = (1/N) ∑ (x[i] - mean)^2
 *   – skew = (1/N) ∑ (x[i] - mean)^3  / (var^(3/2))
 *   – kurt = (1/N) ∑ (x[i] - mean)^4  / (var^2)   [“Fisher=0” version]
 * ---------------------------------------------------------- */
double array_mean(const double *x, size_t N)
{
    if (N == 0)
        return 0.0;
    double s = 0.0;
    for (size_t i = 0; i < N; i++)
        s += x[i];
    // printf("the value of size is %d\n", N);
    // printf("the value of s is %f\n", s);
    return s / (double)N;
}
double array_variance(const double *x, size_t N, double mean)
{
    if (N == 0)
        return 0.0;
    double s = 0.0;
    for (size_t i = 0; i < N; i++)
    {
        double d = x[i] - mean;
        s += d * d;
    }
    return s / (double)N;
}
double array_skewness(const double *x, size_t N, double mean, double var)
{
    if (N < 3 || var < 1e-15)
        return 0.0;
    double m3 = 0.0, sd = sqrt(var);
    for (size_t i = 0; i < N; i++)
    {
        double d = x[i] - mean;
        m3 += d * d * d;
    }
    m3 /= (double)N;
    double g1 = m3 / (sd * sd * sd);
    // bias‐correct
    return g1 * sqrt((double)N * (N - 1)) / (N - 2);
}

double array_kurtosis(const double *x, size_t N, double mean, double var)
{
    if (N == 0 || var < 1e-15)
        return 0.0;
    double s = 0.0;
    for (size_t i = 0; i < N; i++)
    {
        double d = x[i] - mean;
        s += d * d * d * d;
    }
    double pearson = (s / (double)N) / (var * var);
    return pearson - 3.0; // convert to Fisher kurtosis
}

/* ----------------------------------------------------------
 *  Simpson's rule: integrate y[0..N-1] with spacing h=1:
 *    ∫ y dx ≈ (h/3) [ y[0] + 4y[1] + 2y[2] + 4y[3] + ... + y[N-1] ]
 *  If N is even, N-1 is odd index, so final term weight=1.
 * ---------------------------------------------------------- */
double simpson_auc(const double *y, size_t N)
{
    if (N < 2)
        return 0.0;
    double s = y[0] + y[N - 1];
    for (size_t i = 1; i + 1 < N; i++)
    {
        if (i % 2 == 1)
            s += 4.0 * y[i];
        else
            s += 2.0 * y[i];
    }
    return s / 3.0; // h=1 ⇒ h/3=1/3
}

/* ----------------------------------------------------------
 *  Next power of two ≥ n:
 * ---------------------------------------------------------- */
size_t next_pow2(size_t n)
{
    size_t p = 1;
    while (p < n)
        p <<= 1;
    return p;
}

//  This is O(N log N) recursion.
void fft_rec(double *real, double *imag, size_t N)
{
    if (N <= 1)
        return;
    // Divide: allocate arrays for evens & odds
    size_t half = N / 2;
    double *r_even = (double *)malloc(sizeof(double) * half);
    double *i_even = (double *)malloc(sizeof(double) * half);
    double *r_odd = (double *)malloc(sizeof(double) * half);
    double *i_odd = (double *)malloc(sizeof(double) * half);
    if (!r_even || !i_even || !r_odd || !i_odd)
    {
        fprintf(stderr, "Error: malloc failed in fft_rec\n");
        exit(1);
    }
    for (size_t i = 0; i < half; i++)
    {
        r_even[i] = real[2 * i];
        i_even[i] = imag[2 * i];
        r_odd[i] = real[2 * i + 1];
        i_odd[i] = imag[2 * i + 1];
    }
    // Recursively FFT subarrays
    fft_rec(r_even, i_even, half);
    fft_rec(r_odd, i_odd, half);
    // Combine
    for (size_t k = 0; k < half; k++)
    {
        double t_re = cos(-2.0 * M_PI * k / N) * r_odd[k] - sin(-2.0 * M_PI * k / N) * i_odd[k];
        double t_im = sin(-2.0 * M_PI * k / N) * r_odd[k] + cos(-2.0 * M_PI * k / N) * i_odd[k];
        real[k] = r_even[k] + t_re;
        imag[k] = i_even[k] + t_im;
        real[k + half] = r_even[k] - t_re;
        imag[k + half] = i_even[k] - t_im;
    }
    free(r_even);
    free(i_even);
    free(r_odd);
    free(i_odd);
}
double *compute_fft_magnitude(const double *x, size_t N, size_t *out_len)
{
    // Find next power of 2
    size_t M = next_pow2(N);
    // Allocate real/imag arrays of length M, initialize with zeros
    double *r = (double *)calloc(M, sizeof(double));
    double *im = (double *)calloc(M, sizeof(double));
    if (!r || !im)
    {
        fprintf(stderr, "Error: calloc failed in FFT prep\n");
        exit(1);
    }
    // Copy x into r[0..N-1]
    for (size_t i = 0; i < N; i++)
    {
        r[i] = x[i];
    }
    // FFT in‐place on (r,im)
    fft_rec(r, im, M);
    // We want one‐sided magnitude: indices 0..M/2-1
    size_t Nout = M / 2;
    double *mag = (double *)malloc(sizeof(double) * Nout);
    if (!mag)
    {
        fprintf(stderr, "Error: malloc failed\n");
        exit(1);
    }
    for (size_t k = 0; k < Nout; k++)
    {
        mag[k] = sqrt(r[k] * r[k] + im[k] * im[k]);
    }
    free(r);
    free(im);
    *out_len = Nout;
    return mag;
}

/* ----------------------------------------------------------
 *  Hanning window of length L: w[n] = 0.5*(1 - cos(2πn/(L-1)))
 * ---------------------------------------------------------- */
double *hann_window(size_t L)
{
    double *w = (double *)malloc(sizeof(double) * L);
    if (!w)
    {
        fprintf(stderr, "Error: malloc failed\n");
        exit(1);
    }
    for (size_t n = 0; n < L; n++)
    {
        w[n] = 0.5 * (1.0 - cos(2.0 * M_PI * (double)n / (double)(L - 1)));
    }
    return w;
}

double *welch_psd(const double *data, size_t N, size_t *psd_len_out)
{
    size_t nperseg = 256;
    double fs = 100.0;
    size_t noverlap = nperseg / 2;
    if (N < nperseg)
    {
        nperseg = N;
        noverlap = 0;
    }
    size_t step = nperseg - noverlap;
    size_t n_segments = 1 + (N - nperseg) / step;

    // Use next power of two for FFT
    size_t nfft = next_pow2(nperseg);
    size_t half_bin = nfft / 2 + 1;

    // Allocate accumulation array
    double *psd_sum = (double *)calloc(half_bin, sizeof(double));
    if (!psd_sum)
    {
        perror("calloc");
        exit(1);
    }

    // Create Hann window and compute U = sum(window^2)
    double *window = hann_window(nperseg);
    double U = 0.0;
    for (size_t i = 0; i < nperseg; i++)
    {
        U += window[i] * window[i];
    }
    // No multiplication by fs here

    // Buffers for each segment
    double *seg = (double *)malloc(sizeof(double) * nperseg);
    double *buf_re = (double *)calloc(nfft, sizeof(double));
    double *buf_im = (double *)calloc(nfft, sizeof(double));
    if (!seg || !buf_re || !buf_im)
    {
        perror("malloc");
        exit(1);
    }

    // Process each segment
    for (size_t s = 0; s < n_segments; s++)
    {
        size_t start = s * step;
        // Windowed segment
        double mean_seg = 0.0;
        for (size_t i = 0; i < nperseg; i++)
        {
            mean_seg += data[start + i];
        }
        mean_seg /= (double)nperseg;
        for (size_t i = 0; i < nperseg; i++)
        {
            seg[i] = (data[start + i] - mean_seg) * window[i];
            buf_re[i] = seg[i];
            buf_im[i] = 0.0;
        }
        // Zero pad remainder
        for (size_t i = nperseg; i < nfft; i++)
        {
            buf_re[i] = 0.0;
            buf_im[i] = 0.0;
        }
        // FFT in-place
        fft_rec(buf_re, buf_im, nfft);

        // Compute two-sided periodogram and accumulate
        for (size_t k = 0; k < half_bin; k++)
        {
            double mag2 = buf_re[k] * buf_re[k] + buf_im[k] * buf_im[k];
            // Divide by fs*U to normalize power
            psd_sum[k] += mag2 / (fs * U);
        }
    }

    // Average across segments
    for (size_t k = 0; k < half_bin; k++)
    {
        psd_sum[k] /= (double)n_segments;
    }

    // Convert to one-sided: double bins 1..(half_bin-2)
    // (leave k=0 and k=half_bin-1 unmodified)
    for (size_t k = 1; k + 1 < half_bin; k++)
    {
        psd_sum[k] *= 2.0;
    }

    free(window);
    free(seg);
    free(buf_re);
    free(buf_im);

    *psd_len_out = half_bin;
    return psd_sum;
}

/* ----------------------------------------------------------
 *  Detect peaks in a signal x[0..N-1] with:
 *    - minimal distance "min_dist" between peaks
 *    - we simply require x[i] > x[i-1] && x[i] > x[i+1].
 *  Returns an array of indices (dynamically allocated), and sets *npeaks.
 * ---------------------------------------------------------- */
int *detect_peaks(const double *x, size_t N, size_t min_dist, size_t *npeaks_out)
{
    // First pass: find all local maxima
    int *temp = (int *)malloc(sizeof(int) * N);
    if (!temp)
    {
        fprintf(stderr, "Error: malloc failed\n");
        exit(1);
    }
    size_t count = 0;
    for (size_t i = 1; i + 1 < N; i++)
    {
        if (x[i] > x[i - 1] && x[i] > x[i + 1])
        {
            temp[count++] = (int)i;
        }
    }
    // Now enforce minimal distance: we’ll do the standard greedy approach:
    // sort peaks by amplitude descending, pick one, remove all within min_dist, etc.
    // But for simplicity, since no amplitude threshold was specified, we'll do a simpler
    // left‐to‐right approach: keep a rolling “last_peak” and only accept if i - last_peak ≥ min_dist.
    int *peaks = (int *)malloc(sizeof(int) * count);
    if (!peaks)
    {
        fprintf(stderr, "Error: malloc failed\n");
        exit(1);
    }
    size_t keep = 0;
    int last_peak = -(int)min_dist - 1;
    for (size_t j = 0; j < count; j++)
    {
        int idx = temp[j];
        if (idx - last_peak >= (int)min_dist)
        {
            peaks[keep++] = idx;
            last_peak = idx;
        }
    }
    free(temp);
    *npeaks_out = keep;
    return peaks;
}

int *detect_troughs(const double *x, size_t N, size_t min_dist, size_t *ntroughs_out)
{
    // Create a temporary array neg_x = -x
    double *negx = (double *)malloc(sizeof(double) * N);
    if (!negx)
    {
        fprintf(stderr, "Error: malloc failed\n");
        exit(1);
    }
    for (size_t i = 0; i < N; i++)
        negx[i] = -x[i];
    int *troughs = detect_peaks(negx, N, min_dist, ntroughs_out);
    free(negx);
    return troughs;
}

double slope_int(int x1, double y1, int x2, double y2)
{
    return (y2 - y1) / (double)(x2 - x1);
}

double *compute_more_ppg_features(const double *ppg_bp, const double *ppg_bp_inv, size_t N)
{
    // 1) Detect peaks on ppg_bp:
    size_t npeaks = 0;
    int *peaks = detect_peaks(ppg_bp, N, 50, &npeaks);
    // 2) Detect troughs on ppg_bp_inv:
    size_t nonsets = 0;
    int *onsets = detect_troughs(ppg_bp_inv, N, 50, &nonsets);

    // Initialize accumulators:
    double H1 = 0.0, T1 = 0.0, T2 = 0.0;
    double P1_sum = 0.0, P2_sum = 0.0;
    double S1 = 0.0, S2 = 0.0;

    size_t ind = 0;
    // For each pair of onsets[ i ], onsets[ i+1 ], find first peak >= onsets[i]
    for (size_t i = 0; i + 1 < nonsets; i++)
    {
        int first_onset = onsets[i];
        int second_onset = onsets[i + 1];
        // Move ind so that peaks[ind] >= first_onset
        while (ind < npeaks && peaks[ind] < first_onset)
            ind++;
        if (ind < npeaks)
        {
            int sys_peak = peaks[ind];
            if (sys_peak >= second_onset)
            {
                // If the first peak is beyond second_onset, there's no peak in [first,second)
                continue;
            }
            // Compute:
            double t1 = (double)(sys_peak - first_onset);
            double t2 = (double)(second_onset - sys_peak);
            double h1 = ppg_bp[sys_peak];
            double p1 = slope_int(first_onset, ppg_bp[first_onset], sys_peak, ppg_bp[sys_peak]);
            double p2 = slope_int(sys_peak, ppg_bp[sys_peak], second_onset, ppg_bp[second_onset]);
            double s1 = 0.5 * t1 * h1;
            double s2 = 0.5 * t2 * h1;
            H1 += h1;
            T1 += t1;
            T2 += t2;
            P1_sum += p1;
            P2_sum += p2;
            S1 += s1;
            S2 += s2;
        }
    }

    double Tsum = T1 + T2;
    double Tdiff = T1 - T2;
    double Tratio = (T2 != 0.0) ? (T1 / T2) : 0.0;
    double Ssum = S1 + S2;
    double s1_div_s = (Ssum != 0.0) ? (S1 / Ssum) : 0.0;
    double s2_div_s = (Ssum != 0.0) ? (S2 / Ssum) : 0.0;
    double s_ratio = (S2 != 0.0) ? (S1 / S2) : 0.0;

    double *out = (double *)malloc(sizeof(double) * 15);
    if (!out)
    {
        fprintf(stderr, "Error: malloc failed\n");
        exit(1);
    }
    out[0] = H1;
    out[1] = T1;
    out[2] = T2;
    out[3] = Tsum;
    out[4] = Tdiff;
    out[5] = Tratio;
    out[6] = S1;
    out[7] = S2;
    out[8] = Ssum;
    out[9] = s1_div_s;
    out[10] = s2_div_s;
    out[11] = s_ratio;
    out[12] = P1_sum;
    out[13] = P2_sum;
    free(onsets);
    free(peaks);
    return out;
}

double *compute_ppg_features(const double *ppg_bp, size_t N)
{
    double *features = (double *)malloc(sizeof(double) * 14);
    if (!features)
    {
        fprintf(stderr, "Error: malloc failed\n");
        exit(1);
    }

    // 1) Welch PSD
    size_t psd_len;
    double *psd = welch_psd(ppg_bp, N, &psd_len);
    double mean_psd = array_mean(psd, psd_len);
    double var_psd = array_variance(psd, psd_len, mean_psd);
    double kurt_psd = array_kurtosis(psd, psd_len, mean_psd, var_psd);

    // 2) FFT magnitude
    size_t fftmag_len;
    double *fftmag = compute_fft_magnitude(ppg_bp, N, &fftmag_len);
    double mean_fft = array_mean(fftmag, fftmag_len);
    double var_fft = array_variance(fftmag, fftmag_len, mean_fft);
    double skew_fft = array_skewness(fftmag, fftmag_len, mean_fft, var_fft);
    double kurt_fft = array_kurtosis(fftmag, fftmag_len, mean_fft, var_fft);

    // 3) DC & AC
    double DC = array_mean(ppg_bp, N);
    double ac_acc = 0.0;
    for (size_t i = 0; i < N; i++)
    {
        double d = ppg_bp[i] - DC;
        ac_acc += d * d;
    }
    double AC = sqrt(ac_acc / (double)N);

    // 4) kte
    double *kte = (double *)malloc(sizeof(double) * N);
    if (!kte)
    {
        fprintf(stderr, "Error: malloc failed\n");
        exit(1);
    }
    compute_kte(ppg_bp, kte, N);
    double mean_kte = array_mean(kte, N);
    double var_kte = array_variance(kte, N, mean_kte);
    double skew_kte = array_skewness(kte, N, mean_kte, var_kte);
    double kurt_kte = array_kurtosis(kte, N, mean_kte, var_kte);
    double std_kte = sqrt(var_kte);
    double auc_kte = simpson_auc(kte, N);

    // Fill features:
    features[0] = mean_psd;
    features[1] = kurt_psd;
    features[2] = var_psd;
    features[3] = kurt_kte;
    features[4] = var_kte;
    features[5] = mean_kte;
    features[6] = skew_kte;
    features[7] = auc_kte;
    features[8] = std_kte;
    features[9] = kurt_fft;
    features[10] = skew_fft;
    features[11] = var_fft;
    features[12] = DC;
    features[13] = AC;

    free(psd);
    free(fftmag);
    free(kte);
    return features;
}

int main(int argc, char *argv[])
{
    clock_t start, end;
    double cpu_time_used;
    start = clock();
    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s raw.csv inverted.csv\n", argv[0]);
        return 1;
    }

    // 1) Read CSVs
    size_t Nraw, Ninv;
    double *raw = read_csv(argv[1], &Nraw);
    double *inv = read_csv(argv[2], &Ninv);

    // Print out what we actually loaded, for debugging
    fprintf(stderr, "Debug: raw.csv     length = %zu\n", Nraw);
    fprintf(stderr, "Debug: inverted.csv length = %zu\n", Ninv);
    // Take the minimum length of the two files

    size_t Nfull = (Nraw < Ninv) ? Nraw : Ninv;
    if (Nfull <= 500)
    {
        fprintf(stderr, "Error: after trimming to min length, data length is %zu (≤ 500)\n", Nfull);
        return 1;
    }

    // 2) Discard first 500 samples from both (using the common min length)
    size_t N = Nfull - 500;
    // printf("the value of N is : %d\n", N);
    double *ppg = (double *)malloc(sizeof(double) * (N));
    double *ppg_inv = (double *)malloc(sizeof(double) * (N));
    if (!ppg || !ppg_inv)
    {
        fprintf(stderr, "Error: malloc failed\n");
        return 1;
    }
    for (size_t i = 0; i < N; i++)
    {
        ppg[i] = raw[i + 500];
        ppg_inv[i] = inv[i + 500];
    }
    // printf("ppg_raw Values Printed after this\n");
    // for (int i = 0; i < N; i++)
    // {
    //     printf("value at index %d is %f\n",i,ppg[i]);
    // }
    // printf("ppg_raw inverted Values Printed after this\n");
    // for (int i = 0; i < N; i++)
    // {
    //     printf("%f\n",ppg_inv[i]);
    // }
    free(raw);
    free(inv);

    // 3) Apply high‐pass filter via filtfilt
    double *hp_ppg = (double *)malloc(sizeof(double) * N);
    double *hp_ppg_inv = (double *)malloc(sizeof(double) * N);
    if (!hp_ppg || !hp_ppg_inv)
    {
        fprintf(stderr, "Error: malloc failed\n");
        return 1;
    }
    filtfilt_cheby(b_hp, a_hp, HP_NCOEF, ppg, hp_ppg, N);
    filtfilt_cheby(b_hp, a_hp, HP_NCOEF, ppg_inv, hp_ppg_inv, N);
    // for (int i = 0; i < N; i++)
    // {
    //     printf("The value os vjdflijv : %f\n", hp_ppg[i]);
    // }
    for (int i = 0; i < 10; i++)
    {
        printf("vfvfgv%.8f\n", hp_ppg[i]);
    }
    FILE *file = fopen("hp_ppg_data.txt", "w"); // Open the file in write mode

    // Check if the file was opened successfully
    if (file == NULL)
    {
        perror("Error opening file for writing");
        // Handle the error, perhaps exit the program or return an error code
        // return 1;
        exit(EXIT_FAILURE); // Exit the program if file opening fails
    }

    // Write the values from the hp_ppg array to the file
    for (int i = 0; i < N; ++i)
    {
        fprintf(file, "%0.14f\n", hp_ppg[i]); // Write each double value followed by a newline
    }

    // Close the file
    fclose(file);
    // 4) Apply band‐pass filter via filtfilt
    double *bp_ppg = (double *)malloc(sizeof(double) * N);
    double *bp_ppg_inv = (double *)malloc(sizeof(double) * N);
    if (!bp_ppg || !bp_ppg_inv)
    {
        fprintf(stderr, "Error: malloc failed\n");
        return 1;
    }
    filtfilt_cheby(b_bp, a_bp, BP_NCOEF, hp_ppg, bp_ppg, N);
    filtfilt_cheby(b_bp, a_bp, BP_NCOEF, hp_ppg_inv, bp_ppg_inv, N);

    FILE *file1 = fopen("bp_ppg_data.txt", "w"); // Open the file in write mode

    // Check if the file was opened successfully
    if (file1 == NULL)
    {
        perror("Error opening file for writing");
        // Handle the error, perhaps exit the program or return an error code
        // return 1;
        exit(EXIT_FAILURE); // Exit the program if file opening fails
    }

    // Write the values from the hp_ppg array to the file
    for (int i = 0; i < N; ++i)
    {
        fprintf(file1, "%0.14f\n", bp_ppg[i]); // Write each double value followed by a newline
    }

    // Close the file
    fclose(file1);
    free(hp_ppg);
    free(hp_ppg_inv);

    // 5) Normalize each to [0,1]
    normalize01(bp_ppg, N);
    normalize01(bp_ppg_inv, N);
    // for (int i = 0; i < N; i++)
    // {
    //     printf("the value is : %f\n", bp_ppg[i]);
    // }

    // 6) Compute ppg_features
    double *f_ppg = compute_ppg_features(bp_ppg, N);

    // 7) Compute more_ppg_features
    double *f_more = compute_more_ppg_features(bp_ppg, bp_ppg_inv, N);

    // 8) Print results with descriptive labels

    // --- Line 1: 14 ppg_features ---
    printf("### ppg_features:\n");
    printf("  1) mean_psd    = %.14f\n", f_ppg[0]);
    printf("  2) kurt_psd    = %.14f\n", f_ppg[1]);
    printf("  3) var_psd     = %.14f\n", f_ppg[2]);

    printf("  4) kurt_kte    = %.14f\n", f_ppg[3]);
    printf("  5) var_kte     = %.14f\n", f_ppg[4]);
    printf("  6) mean_kte    = %.14f\n", f_ppg[5]);
    printf("  7) skew_kte    = %.14f\n", f_ppg[6]);
    printf("  8) auc_kte     = %.14f\n", f_ppg[7]);
    printf("  9) std_kte     = %.14f\n", f_ppg[8]);

    printf(" 10) kurt_fft    = %.14f\n", f_ppg[9]);
    printf(" 11) skew_fft    = %.14f\n", f_ppg[10]);
    printf(" 12) var_fft     = %.14f\n", f_ppg[11]);

    printf(" 13) DC          = %.14f\n", f_ppg[12]);
    printf(" 14) AC          = %.14f\n", f_ppg[13]);

    printf("\n");

    // --- Line 2: 14 more_ppg_features ---
    printf("### more_ppg_features:\n");
    printf("  1) H1          = %.14f\n", f_more[0]);
    printf("  2) T1          = %.14f\n", f_more[1]);
    printf("  3) T2          = %.14f\n", f_more[2]);
    printf("  4) Tsum (T1+T2)= %.14f\n", f_more[3]);
    printf("  5) Tdiff (T1–T2)= %.14f\n", f_more[4]);
    printf("  6) Tratio      = %.14f\n", f_more[5]);

    printf("  7) S1          = %.14f\n", f_more[6]);
    printf("  8) S2          = %.14f\n", f_more[7]);
    printf("  9) Ssum (S1+S2)= %.14f\n", f_more[8]);
    printf(" 10) S1/Ssum     = %.14f\n", f_more[9]);
    printf(" 11) S2/Ssum     = %.14f\n", f_more[10]);
    printf(" 12) Sratio (S1/S2)= %.14f\n", f_more[11]);

    printf(" 13) P1_sum      = %.14f\n", f_more[12]);
    printf(" 14) P2_sum      = %.14f\n", f_more[13]);

    // Cleanup
    free(ppg);
    free(ppg_inv);
    free(bp_ppg);
    free(bp_ppg_inv);
    free(f_ppg);
    free(f_more);
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time taken: %f seconds\n", cpu_time_used);
    return 0;
}