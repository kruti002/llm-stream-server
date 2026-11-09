//! Small, self-contained statistics helpers.
//!
//! The benchmark only needs a handful of summary numbers (percentiles and a
//! mean). Pulling in a heavy stats/histogram crate would be harder to justify
//! in an interview than a dozen well-commented lines, and an exact
//! sort-then-index percentile is both correct and easy to reason about at the
//! sample sizes this tool uses (hundreds to thousands of requests per level).
//!
//! # Percentile method: nearest-rank on sorted samples
//!
//! We use the *nearest-rank* method. Given `n` sorted samples and a percentile
//! `p` in `[0, 100]`, the rank is `ceil(p/100 * n)`, clamped to `[1, n]`, and
//! we return the value at 1-based position `rank` (i.e. 0-based `rank - 1`).
//! This always returns an actual observed sample (no interpolation), which is
//! the honest thing to report for latency: every number in the table is a real
//! measurement that happened, not a synthesized value between two samples.

/// Nearest-rank percentile of a slice of `f64` samples.
///
/// `p` is a percentile in `[0.0, 100.0]`. Returns `None` for an empty slice.
/// The input is cloned and sorted internally, so callers keep their ordering.
pub fn percentile(samples: &[f64], p: f64) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    let mut sorted = samples.to_vec();
    // Total order on f64 via `total_cmp` — benchmark timings are always finite
    // and non-negative, but `total_cmp` avoids the `partial_cmp` unwrap and is
    // well-defined even if a NaN ever sneaks in.
    sorted.sort_by(|a, b| a.total_cmp(b));

    let n = sorted.len();
    // rank = ceil(p/100 * n), clamped into [1, n], then converted to a 0-based
    // index. Using the ceiling keeps p100 mapped to the maximum sample.
    let rank = (p / 100.0 * n as f64).ceil() as usize;
    let rank = rank.clamp(1, n);
    Some(sorted[rank - 1])
}

/// Arithmetic mean of a slice of samples. `None` for an empty slice.
pub fn mean(samples: &[f64]) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    let sum: f64 = samples.iter().sum();
    Some(sum / samples.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_empty_is_none() {
        assert_eq!(percentile(&[], 50.0), None);
        assert_eq!(mean(&[]), None);
    }

    #[test]
    fn percentile_single_sample() {
        // With one sample every percentile is that sample.
        assert_eq!(percentile(&[42.0], 50.0), Some(42.0));
        assert_eq!(percentile(&[42.0], 95.0), Some(42.0));
        assert_eq!(percentile(&[42.0], 0.0), Some(42.0));
    }

    #[test]
    fn percentile_nearest_rank_known_values() {
        // Samples 1..=10. Nearest-rank: rank = ceil(p/100 * 10).
        let s: Vec<f64> = (1..=10).map(|v| v as f64).collect();
        // p50 -> ceil(5.0) = 5 -> value at index 4 = 5.0
        assert_eq!(percentile(&s, 50.0), Some(5.0));
        // p95 -> ceil(9.5) = 10 -> value at index 9 = 10.0
        assert_eq!(percentile(&s, 95.0), Some(10.0));
        // p90 -> ceil(9.0) = 9 -> value at index 8 = 9.0
        assert_eq!(percentile(&s, 90.0), Some(9.0));
        // p100 -> ceil(10.0) = 10 -> max
        assert_eq!(percentile(&s, 100.0), Some(10.0));
        // p10 -> ceil(1.0) = 1 -> min
        assert_eq!(percentile(&s, 10.0), Some(1.0));
    }

    #[test]
    fn percentile_is_order_independent() {
        // Unsorted input must give the same result as sorted input.
        let unsorted = [9.0, 1.0, 5.0, 3.0, 7.0];
        let sorted = [1.0, 3.0, 5.0, 7.0, 9.0];
        for p in [0.0, 25.0, 50.0, 75.0, 95.0, 100.0] {
            assert_eq!(percentile(&unsorted, p), percentile(&sorted, p));
        }
    }

    #[test]
    fn percentile_does_not_mutate_input() {
        let s = vec![3.0, 1.0, 2.0];
        let _ = percentile(&s, 50.0);
        // Caller's slice ordering is preserved (we clone internally).
        assert_eq!(s, vec![3.0, 1.0, 2.0]);
    }

    #[test]
    fn mean_basic() {
        assert_eq!(mean(&[2.0, 4.0, 6.0]), Some(4.0));
        assert_eq!(mean(&[5.0]), Some(5.0));
    }
}
