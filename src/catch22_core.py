from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.interpolate import UnivariateSpline

FEATURE_NAMES = [
    "DN_HistogramMode_5",
    "DN_HistogramMode_10",
    "CO_f1ecac",
    "CO_FirstMin_ac",
    "CO_HistogramAMI_even_2_5",
    "CO_trev_1_num",
    "MD_hrv_classic_pnn40",
    "SB_BinaryStats_mean_longstretch1",
    "SB_TransitionMatrix_3ac_sumdiagcov",
    "PD_PeriodicityWang_th0_01",
    "CO_Embed2_Dist_tau_d_expfit_meandiff",
    "IN_AutoMutualInfoStats_40_gaussian_fmmi",
    "FC_LocalSimple_mean1_tauresrat",
    "DN_OutlierInclude_p_001_mdrmd",
    "DN_OutlierInclude_n_001_mdrmd",
    "SP_Summaries_welch_rect_area_5_1",
    "SB_BinaryStats_diff_longstretch0",
    "SB_MotifThree_quantile_hh",
    "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1",
    "SP_Summaries_welch_rect_centroid",
    "FC_LocalSimple_mean3_stderr",
]

SHORT_NAMES = [
    "mode_5",
    "mode_10",
    "acf_timescale",
    "acf_first_min",
    "ami2",
    "trev",
    "high_fluctuation",
    "stretch_high",
    "transition_matrix",
    "periodicity",
    "embedding_dist",
    "ami_timescale",
    "whiten_timescale",
    "outlier_timing_pos",
    "outlier_timing_neg",
    "low_freq_power",
    "stretch_decreasing",
    "entropy_pairs",
    "rs_range",
    "dfa",
    "centroid_freq",
    "forecast_error",
]


def _as_float_array(values: Iterable[float]) -> np.ndarray:
    x = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    x = x.ravel()
    if x.size < 10:
        raise ValueError("catch22 necesita una serie con al menos 10 observaciones")
    if not np.all(np.isfinite(x)):
        finite = np.isfinite(x)
        if not finite.any():
            return np.zeros_like(x)
        x = np.interp(np.arange(x.size), np.flatnonzero(finite), x[finite])
    return x


def _z_normalise(x: np.ndarray) -> np.ndarray:
    std = np.std(x)
    if std == 0 or not np.isfinite(std):
        return np.zeros_like(x)
    return (x - np.mean(x)) / std


def _histogram_mode(x: np.ndarray, bins: int) -> float:
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if xmax == xmin:
        return 0.0
    counts, edges = np.histogram(x, bins=bins, range=(xmin, xmax))
    centres = (edges[:-1] + edges[1:]) / 2
    return float(np.mean(centres[counts == counts.max()]))


def _autocorrelation(x: np.ndarray) -> np.ndarray:
    y = x - np.mean(x)
    n = len(y)
    nfft = 1 << int(np.ceil(np.log2(max(2, 2 * n - 1))))
    spectrum = np.fft.rfft(y, n=nfft)
    ac = np.fft.irfft(spectrum * np.conj(spectrum), n=nfft)[:n]
    if ac[0] == 0:
        return np.zeros(n)
    return np.real(ac / ac[0])


def _first_zero(ac: np.ndarray) -> int:
    hits = np.where(ac[1:] <= 0)[0]
    return int(hits[0] + 1) if hits.size else len(ac)


def _first_1e_crossing(ac: np.ndarray) -> float:
    threshold = math.exp(-1)
    for i in range(len(ac) - 1):
        if ac[i + 1] < threshold:
            slope = ac[i + 1] - ac[i]
            return float(i if slope == 0 else i + (threshold - ac[i]) / slope)
    return float(len(ac))


def _first_minimum(ac: np.ndarray) -> float:
    for i in range(1, len(ac) - 1):
        if ac[i] < ac[i - 1] and ac[i] < ac[i + 1]:
            return float(i)
    return float(len(ac))


def _histogram_ami(x: np.ndarray, lag: int = 2, bins: int = 5) -> float:
    if len(x) <= lag:
        return 0.0
    xmin, xmax = np.min(x) - 0.1, np.max(x) + 0.1
    width = (xmax - xmin) / bins
    if width == 0:
        return 0.0
    a = np.clip(((x[:-lag] - xmin) / width).astype(int), 0, bins - 1)
    b = np.clip(((x[lag:] - xmin) / width).astype(int), 0, bins - 1)
    joint = np.zeros((bins, bins), dtype=float)
    for i, j in zip(a, b):
        joint[i, j] += 1
    joint /= max(1, joint.sum())
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    denom = px @ py
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log(joint[mask] / denom[mask])))


def _time_reversibility(x: np.ndarray) -> float:
    return float(np.mean(np.diff(x) ** 3))


def _pnn40(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(x)) > 0.04))


def _longest_run(binary: np.ndarray, value: int) -> int:
    best = current = 0
    for item in binary:
        if int(item) == value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def _longstretch_above_mean(x: np.ndarray) -> float:
    return float(_longest_run((x[:-1] > np.mean(x)).astype(int), 1))


def _quantile_labels(x: np.ndarray, groups: int = 3) -> np.ndarray:
    edges = np.quantile(x, np.linspace(0, 1, groups + 1), method="linear")
    edges[0] -= 1e-12
    labels = np.zeros(len(x), dtype=int)
    for i in range(groups):
        mask = (x > edges[i]) & (x <= edges[i + 1])
        labels[mask] = i
    return labels


def _transition_matrix_covariance(x: np.ndarray, lag: int) -> float:
    lag = max(1, min(int(lag), len(x) - 1))
    downsampled = x[::lag]
    if len(downsampled) < 3:
        return 0.0
    labels = _quantile_labels(downsampled, 3)
    matrix = np.zeros((3, 3), dtype=float)
    for a, b in zip(labels[:-1], labels[1:]):
        matrix[a, b] += 1
    matrix /= max(1, len(labels) - 1)
    cov = np.cov(matrix.T, bias=False)
    return float(np.trace(cov)) if cov.ndim == 2 else 0.0


def _periodicity_wang(x: np.ndarray) -> float:
    n = len(x)
    grid = np.arange(n, dtype=float)
    try:
        spline = UnivariateSpline(grid, x, k=min(3, n - 1), s=n * np.var(x) * 0.1)
        detrended = x - spline(grid)
    except Exception:
        detrended = x - np.polyval(np.polyfit(grid, x, 2), grid)

    max_lag = int(np.ceil(n / 3))
    covariances = np.array(
        [np.mean(detrended[:-lag] * detrended[lag:]) for lag in range(1, max_lag + 1)]
    )
    peaks = []
    troughs = []
    for i in range(1, len(covariances) - 1):
        if covariances[i] > covariances[i - 1] and covariances[i] > covariances[i + 1]:
            peaks.append(i)
        if covariances[i] < covariances[i - 1] and covariances[i] < covariances[i + 1]:
            troughs.append(i)
    for peak in peaks:
        previous = [t for t in troughs if t < peak]
        if not previous:
            continue
        trough = previous[-1]
        if covariances[peak] > 0 and covariances[peak] - covariances[trough] >= 0.01:
            return float(peak)
    return 0.0


def _embedding_distance(x: np.ndarray, tau: int) -> float:
    tau = max(1, min(int(tau), max(1, len(x) // 10)))
    if len(x) - tau - 1 < 4:
        return 0.0
    d = np.sqrt(np.diff(x[:-tau]) ** 2 + np.diff(x[tau:]) ** 2)
    if len(d) < 3 or np.std(d) < 1e-3 or np.mean(d) == 0:
        return 0.0
    q75, q25 = np.percentile(d, [75, 25])
    width = 2 * (q75 - q25) / np.cbrt(len(d))
    if width <= 0:
        bins = max(2, int(np.sqrt(len(d))))
    else:
        bins = max(2, int(np.ceil((d.max() - d.min()) / width)))
    density, edges = np.histogram(d, bins=bins, density=True)
    centres = (edges[:-1] + edges[1:]) / 2
    expected = np.exp(-centres / np.mean(d)) / np.mean(d)
    return float(np.mean(np.abs(density - expected)))


def _ami_first_minimum(x: np.ndarray) -> float:
    maximum = min(40, int(np.ceil(len(x) / 2)))
    ami = []
    for lag in range(1, maximum + 1):
        left, right = x[:-lag], x[lag:]
        if np.std(left) == 0 or np.std(right) == 0:
            ami.append(np.nan)
            continue
        corr = np.corrcoef(left, right)[0, 1]
        corr = float(np.clip(corr, -0.999999, 0.999999))
        ami.append(-0.5 * np.log(1 - corr * corr))
    ami = np.asarray(ami)
    for i in range(1, len(ami) - 1):
        if np.isfinite(ami[i]) and ami[i] < ami[i - 1] and ami[i] < ami[i + 1]:
            return float(i)
    return float(maximum)


def _local_mean_residual(x: np.ndarray, length: int) -> np.ndarray:
    if len(x) <= length:
        return np.array([], dtype=float)
    return np.array([x[i + length] - np.mean(x[i : i + length]) for i in range(len(x) - length)])


def _tauresrat(x: np.ndarray, acfz: int) -> float:
    residual = _local_mean_residual(x, 1)
    if residual.size < 2:
        return 0.0
    tau = _first_zero(_autocorrelation(residual))
    return float(tau / max(1, acfz))


def _outlier_timing(x: np.ndarray, positive: bool = True) -> float:
    z = _z_normalise(x)
    if not positive:
        z = -z
    positive_values = z[z >= 0]
    if positive_values.size < 2 or np.max(positive_values) < 0.01:
        return 0.0
    thresholds = np.arange(0, np.max(positive_values) + 0.005, 0.01)
    medians = []
    useful = []
    for threshold in thresholds:
        positions = np.where(z >= threshold)[0] + 1
        if positions.size < 2:
            continue
        distances = np.diff(positions)
        density = 100 * len(distances) / max(1, len(positive_values))
        if density > 2:
            medians.append(np.median(positions) / (len(z) / 2) - 1)
            useful.append(threshold)
    return float(np.median(medians)) if medians else 0.0


def _welch_features(x: np.ndarray) -> tuple[float, float]:
    nfft = 1 << int(np.ceil(np.log2(len(x))))
    fft = np.fft.fft(x - np.mean(x), n=nfft)
    half = nfft // 2 + 1
    power = np.abs(fft[:half]) ** 2 / len(x) / (2 * np.pi)
    if half > 2:
        power[1:-1] *= 2
    omega = np.arange(half) / nfft * 2 * np.pi
    cutoff = int(np.floor(half / 5))
    area = float(np.sum(power[:cutoff]) * (omega[1] - omega[0])) if cutoff > 0 else 0.0
    cumulative = np.cumsum(power)
    if cumulative[-1] <= 0:
        centroid = 0.0
    else:
        idx = int(np.searchsorted(cumulative, cumulative[-1] / 2, side="right"))
        centroid = float(omega[min(idx, half - 1)])
    return area, centroid


def _longstretch_decreasing(x: np.ndarray) -> float:
    return float(_longest_run((np.diff(x) < 0).astype(int), 1))


def _motif_entropy(x: np.ndarray) -> float:
    labels = _quantile_labels(x, 3)
    counts = np.zeros((3, 3), dtype=float)
    for a, b in zip(labels[:-1], labels[1:]):
        counts[a, b] += 1
    probs = counts / max(1, len(labels) - 1)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def _linear_fit_error(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or not np.all(np.isfinite(y)):
        return np.inf
    coeff = np.polyfit(x, y, 1)
    return float(np.sqrt(np.sum((np.polyval(coeff, x) - y) ** 2)))


def _fluctuation_proportion(x: np.ndarray, original_length: int, dfa: bool) -> float:
    scales = np.unique(np.round(np.exp(np.linspace(np.log(5), np.log(original_length / 2), 50))).astype(int))
    scales = scales[scales >= 5]
    if len(scales) < 12:
        return 0.0
    fluctuations = []
    valid_scales = []
    for scale in scales:
        segments = len(x) // scale
        if segments < 1:
            continue
        values = []
        grid = np.arange(1, scale + 1, dtype=float)
        for segment in range(segments):
            window = x[segment * scale : (segment + 1) * scale]
            coeff = np.polyfit(grid, window, 1)
            residual = window - np.polyval(coeff, grid)
            if dfa:
                values.append(np.mean(residual ** 2))
            else:
                values.append((np.max(residual) - np.min(residual)) ** 2)
        fluct = np.sqrt(np.mean(values))
        if fluct > 0 and np.isfinite(fluct):
            valid_scales.append(scale)
            fluctuations.append(fluct)
    if len(valid_scales) < 12:
        return 0.0
    log_scale = np.log(np.asarray(valid_scales, dtype=float))
    log_fluct = np.log(np.asarray(fluctuations, dtype=float))
    errors = []
    candidates = range(6, len(log_scale) - 5)
    for split in candidates:
        error = _linear_fit_error(log_scale[:split], log_fluct[:split])
        error += _linear_fit_error(log_scale[split - 1 :], log_fluct[split - 1 :])
        errors.append(error)
    if not errors:
        return 0.0
    split = int(np.argmin(errors)) + 6
    return float(split / len(log_scale))


def _rs_range(x: np.ndarray) -> float:
    integrated = np.cumsum(x)
    return _fluctuation_proportion(integrated, len(x), False)


def _dfa(x: np.ndarray) -> float:
    integrated = np.cumsum(x[::2])
    return _fluctuation_proportion(integrated, len(x), True)


def _forecast_error(x: np.ndarray) -> float:
    residual = _local_mean_residual(x, 3)
    return float(np.std(residual, ddof=1)) if residual.size >= 3 else 0.0


def catch22_all(values: Iterable[float], short_names: bool = False) -> dict[str, list]:
    """Calcula las 22 características para una serie univariada."""
    raw = _as_float_array(values)
    # catch22 describe dinámica y no nivel. La normalización también evita que
    # las características de ventanas con escalas distintas dominen el análisis.
    x = _z_normalise(raw)
    ac = _autocorrelation(x)
    acfz = _first_zero(ac)
    low_frequency_power, centroid_frequency = _welch_features(x)

    features = [
        _histogram_mode(x, 5),
        _histogram_mode(x, 10),
        _first_1e_crossing(ac),
        _first_minimum(ac),
        _histogram_ami(x, 2, 5),
        _time_reversibility(x),
        _pnn40(x),
        _longstretch_above_mean(x),
        _transition_matrix_covariance(x, acfz),
        _periodicity_wang(x),
        _embedding_distance(x, acfz),
        _ami_first_minimum(x),
        _tauresrat(x, acfz),
        _outlier_timing(x, True),
        _outlier_timing(x, False),
        low_frequency_power,
        _longstretch_decreasing(x),
        _motif_entropy(x),
        _rs_range(x),
        _dfa(x),
        centroid_frequency,
        _forecast_error(x),
    ]
    features = np.nan_to_num(np.asarray(features, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "names": SHORT_NAMES.copy() if short_names else FEATURE_NAMES.copy(),
        "values": features.tolist(),
    }
