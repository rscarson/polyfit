# rational module — shortfalls and TODO

A record of what works, what's weak, and what remains. Written 2026-04-14
after a devil's-advocate review of the first deployment (linear-srgb PR #8).

## What ships

- **SK → LM pipeline** with analytical Jacobian, Nielsen damping, deterministic
  multi-restart (sin/cos phases, no RNG).
- **`optimize_f32`** — truncate-to-f32 then local coordinate-descent ULP search.
  Supports exhaustive f32 sweep or uniform sampling via `F32SearchConfig.samples`.
- **`refine_from_f32`** — warm-start LM from existing f32 coefficients.
- **`piecewise_gap`** — measures ULP gap between a linear branch and the
  polynomial at their join. Distinct from `boundary_ulp` (poly-vs-reference).
- **`measure_roundtrip`** — composed `f⁻¹(f(v))` error with u8/u16 step counts.
- **`F32SearchResult.exhaustive` + `values_scored`** — callers can tell the
  measurement regime at a glance.
- **Self-consistency test** — guards against scorer/measurement drift.
- **Absolute error scoring** (`F32ScoreMetric::AbsoluteError`) — for output-
  crosses-zero functions where ULP is misleading (log2 near mantissa=1).

Deployed once so far: linear-srgb base 4/4 S2L/L2S coefficients (PR #8, still
under review). EXT_L2S 6/6 previously landed via a separate iteration in that
crate's PR #6.

## Known shortfalls

### 1. f32 search isn't peerless against hand-tuned coefficients

log2 mantissa (libjxl, 3/3 rational on [−0.33, 0.33]):

| source | max_abs_err |
|---|---|
| libjxl | 4.53e-6 |
| polyfit fresh fit | 6.08e-6 (34% worse) |
| polyfit warm-start LM | diverged |

libjxl's coefficients are tuned for absolute error near zero specifically;
polyfit's Chebyshev-node fit overweights the domain edges. We don't currently
have a way to weight the residuals by *region* — only by the global
ULP / absolute / relative metric.

### 2. Warm-start LM diverges from f32 coefficients for tight fits

`refine_from_f32` promotes f32 → f64, runs LM. For very well-tuned starting
points (PQ inverse small, log2 lowp) the f32 rounding produces a residual
structure LM chases into a much worse basin. Current mitigation: try both
fresh and warm-start, keep the better. No root-cause fix.

### 3. Joint (multi-polynomial) optimization is ad-hoc

For roundtrip-sensitive functions (S2L + L2S) the individual-fit optimum is
not the composed-optimum. The example `joint_roundtrip.rs` shows marginal
improvement (rt_fwd 16→15, rt_inv 26→24 ULP) after brute-force coordinate
descent on 13 coefficients. There's no library API for "jointly optimize
two rationals with a user-supplied composite objective."

### 4. Constraint modes are tricky

- `SnapOnly` works for boundary x near zero (small p[0] perturbation).
- `Both` with high weight destabilizes LM (constraint dominates J^T J).
- `SkThenSnap` pulls SK into a worse basin that LM can't recover from.

Callers currently have to try several modes + weights and pick by outcome.
This is the cherry-picking risk the devil's advocate flagged. No principled
weight-selection heuristic.

### 5. Extended-range wide-domain fits are slow

Exhaustive f32 sweep on [0.003, 1.0] after sqrt transform is ~70M points.
`optimize_f32` with exhaustive mode + 13 coefficients × 6 deltas × 5 rounds
takes minutes per fit. Sampled mode (`samples: Some(N)`) is the practical
workaround but may miss the true worst-case f32 value. No early-termination
once a clear winner is found.

### 6. No cross-architecture validation

All measurements were done on x86-64 with FMA. `mul_add` in the Horner loops
goes through `num_traits::Float` → libm software fallback on non-FMA targets
with different rounding. polyfit doesn't spot-check aarch64 / wasm32
behavior. The linear-srgb PR inherited this hole: the "perfect monotonicity"
claim is untested outside x86-64.

### 7. F32 coefficient printing is caller's problem

`{c:.10e}` on an f32 prints 10 digits when only ~7 are representable. The
*stored* value is correct (f32), but pasted into source looks like it has
more precision than it does — triggering `#[allow(clippy::excessive_precision)]`
downstream. No helper that formats f32 at minimum-sufficient precision.

### 8. No "fit script committed alongside output" pattern enforced

polyfit provides the fitter; users write their own scripts calling it. There's
nothing stopping a caller from running many restart/weight combinations,
keeping the best, and shipping the coefficients without the recipe. The
linear-srgb PR #8 has exactly this problem — coefficients land in source,
the generator stays in another repo.

## TODO

- [ ] **Region-weighted residuals** — let callers specify `w(x)` in the fit
      objective so log2-mantissa-near-zero stays competitive with libjxl
- [ ] **`optimize_pair`** — library API for joint multi-polynomial optimization
      with user-supplied composite objective; generalize
      `examples/joint_roundtrip.rs`
- [ ] **Diagnostic on warm-start divergence** — `refine_from_f32` should
      return both original and refined metrics so callers can reject
      regressions (currently returns only the refined result)
- [ ] **Constraint-aware LM damping** — scale λ by the data residual only,
      not by constraint rows. Would make `ConstraintMode::Both` with high
      weights actually usable.
- [ ] **`F32SearchConfig.radius` auto-tuning** — start narrow, widen if no
      improvement; currently fixed at 3 ULP/coeff which misses larger
      basins for tight fits
- [ ] **Parallel restart evaluation** behind a feature flag — 4–8 restarts
      are cheap individually, embarrassingly parallel
- [ ] **f32 coefficient formatter** — `fmt_f32_min_precision(v)` returning a
      string that parses back to the exact same f32 bits with minimum digits
- [ ] **Architecture crosscheck helper** — `measure_roundtrip` variant that
      runs the same sweep with libm's `fmaf` emulation vs native FMA,
      reports divergence
- [ ] **CHANGELOG + fitter-script-in-repo template** — document the
      expected pattern: fit script lives next to the coefficients it produces,
      with seed/inputs pinned
- [ ] **Nielsen damping parameter exposure** — LM currently hardcodes
      `τ=1e-3`, `ν₀=2`. Callers with pathological fits may want to tune.
- [ ] **Better scoring for near-zero outputs** — current ULP metric reports
      billions for log2 mantissa = 0 crossings. The `AbsoluteError` metric
      exists but its interaction with `F32SearchConfig` isn't unified across
      scoring and final-measurement paths
- [ ] **Exhaustive sweep speedup** — SIMD the Horner eval in `f32_local_search`.
      Currently scalar; a 4× / 8× speedup would move wide-domain exhaustive
      sweeps from minutes to seconds

## Non-goals / deliberate omissions

- **General nonlinear optimization** — polyfit isn't scipy; stick to
  rational polynomial fits
- **Symbolic simplification** — we return numeric coefficients, not
  factored forms
- **Rational function root finding** — users can call `Polynomial::roots`
  on the numerator/denominator separately if they need this
- **`no_std`** — nalgebra needs std; we inherit that constraint

## Devil's-advocate findings NOT yet addressed

From the review of linear-srgb PR #8 specifically:
- `#[allow(clippy::excessive_precision)]` spreading — needs #7 above
- Fitter-script-not-committed in the target repo — needs #9 above
- No aarch64 / wasm32 validation — needs the architecture crosscheck helper
- README drift in downstream crates — that's a caller discipline issue,
  but polyfit could output the accuracy table in a copy-pasteable format
  so callers are less tempted to skip updating docs
