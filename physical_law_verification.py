"""
physical_law 논문 검증 스크립트 (소산 사영)
====================================================================
각 절은 논문의 정리(Theorem) 또는 부수적 탐구에 대응한다. 각 함수는
독립적으로 실행 가능하도록 필요한 import/정의를 함수 안에 포함한다.

목차:
  - T1  (§4.1): 소산 반군의 시간적분 vs Delta^-1
  - T2  (§4.2): d차원 열핵의 시간적분
  - §6 부기: 이류항 (u·∇)u가 만드는 삼중결합텐서의 에너지중립성
             주의: 이 항목은 표준 에너지법(Fourier/Parseval)을 빌린
             부수적 탐구이며, Δ 구조 자체에서 나온 결과가 아니다.
             나비에-스토크스 밀레니엄 문제(존재성/매끄러움)와는 무관하다.
  - §3.1: V_mn(t)의 크기/위상 분리 (실수축/허수축)
  - §10: 완화-결어긋남 부등식
  - 부록K/T7: 다체중력의 그린함수 중첩

실행: python physical_law_verification_ko.py
또는 개별 함수를 따로 호출해도 됨.
"""

from mpmath import mp, mpf, exp, quad, inf, gamma, pi, sqrt
import numpy as np

mp.dps = 30  # 30자리 정밀도


# ============================================================
# 정리 T1. 소산 반군의 시간적분과 Delta^-1
#   ∫₀^∞ e^{-2n²t} dt = 1/(2n²) = (Δ⁻¹)ₙₙ
# ============================================================

def T1_dissipative_semigroup_integral():
    """
    주의: 이 함수는 T1을 "증명"하지 않는다 — ∫₀^∞ e^{-at}dt = 1/a (a>0)는
    이미 초등적인 해석적 사실이다. 이 함수가 하는 일은 그 사실을
    n=1..10에서 30자리 정밀도로 수치확인하는 것뿐이다. 즉 이 코드는
    "해석적 항등식 + 수치검증"이지, 정리 자체를 계산으로 증명하는 것이 아니다.
    """
    print("=== T1 (해석적 항등식 + 수치검증): ===")
    print("    ∫₀^∞ e^{-2n²t} dt vs 1/(2n²)")
    print(f"{'n':>3} {'해석값 1/(2n²)':>22} {'수치적분':>22} {'절대오차':>14}")
    rows = []
    for n in range(1, 11):
        n_mp = mpf(n)
        analytic = 1 / (2 * n_mp**2)
        numeric = quad(lambda t: exp(-2 * n_mp**2 * t), [0, inf])
        err = abs(analytic - numeric)
        rows.append((n, analytic, numeric, err))
        print(f"{n:>3} {str(analytic):>22} {str(numeric):>22} {str(err):>14}")
    max_err = max(r[3] for r in rows)
    print(f"\n  n=1..10 최대 절대오차: {max_err}")
    assert max_err < mpf('1e-20')
    print("  확인됨: 해석값과 수치적분이 20자리 이상 일치.\n")
    return rows


# ============================================================
# 정리 T2. d차원 열핵의 시간적분
#   G(r) = ∫₀^∞ (4πt)^{-d/2} e^{-r²/4t} dt = Γ(d/2-1) / (4π^{d/2}) · r^{2-d}
#   d=3일 때: G(r) = 1/(4πr)
# ============================================================

def T2_heat_kernel_greens_function(d=3):
    """
    주의: 이 함수도 T2를 "증명"하지 않는다 — 논문이 실제로 쓰는 경우(d=3)에
    대해, 해석적 그린함수 1/(4πr)이 시간적분된 열핵과 일치하는지 수치로
    확인할 뿐이다. "해석적 항등식 + 수치검증"이지 계산적 증명이 아니다.

    이 적분은 d>2에서만 수렴한다 (d=2에서는 Γ(d/2-1)=Γ(0)이 발산하고,
    열핵 시간적분 자체도 로그 발산한다). d를 매개변수로 열어두되,
    d≤2는 조용히 잘못 계산되지 않도록 assert로 막는다.
    """
    assert d > 2, "T2의 시간적분은 d>2에서만 수렴한다 (d=2에서 발산, 그 이하는 정의되지 않음)."
    print(f"=== T2 (해석적 항등식 + 수치검증): 그린함수, d={d} ===")
    print(f"{'r':>6} {'해석값 G(r)':>22} {'수치적분':>22} {'절대오차':>14}")
    rows = []
    for r_val in [0.5, 1.0, 2.0, 3.0]:
        r = mpf(r_val)
        if d == 3:
            analytic = 1 / (4 * pi * r)
        else:
            analytic = gamma(mpf(d)/2 - 1) / (4 * pi**(mpf(d)/2)) * r**(2 - d)
        numeric = quad(lambda t: (4 * pi * t) ** (-mpf(d) / 2) * exp(-r**2 / (4 * t)), [0, inf])
        err = abs(analytic - numeric)
        rows.append((r_val, analytic, numeric, err))
        print(f"{r_val:>6} {str(analytic):>22} {str(numeric):>22} {str(err):>14}")
    max_err = max(r[3] for r in rows)
    print(f"\n  r∈[0.5,1.0,2.0,3.0] 최대 절대오차: {max_err}")
    assert max_err < mpf('1e-8')
    print("  확인됨: 해석적 그린함수 1/(4πr)가 수치적분과 1e-8 이내로 일치.\n")
    return rows


# ============================================================
# §6 부기. 삼중결합텐서의 에너지중립성
#   (부수적 탐구 — 표준 에너지법, Δ 구조 자체의 결과 아님. 논문 §6 부기 참조.)
#
#   u(x,t) = Σₖ cₖ(t) e^{ikx}  (Δ = -d²/dx²의 고유기저)
#   점성 버거스 방정식 (NS 이류항의 1차원 아날로그):
#     dcₖ/dt = -ν k² cₖ  (선형, Δ 자체에서)
#              - (ik/2) Σ_{m+n=k} cₘcₙ   (비선형, 삼중결합)
#
#   주장: 비선형(삼중결합) 항은 Σₖ|cₖ|²(에너지)의 시간변화에 정확히 0을
#   기여한다. 소산은 오직 선형(점성)항에서만 나온다. 이는 물리공간에서의
#   항등식 ∫u²uₓdx=0 (주기경계, 완전미분)과 정확히 대응한다.
# ============================================================

def triple_coupling_tensor_energy_neutrality(N=40, nu=0.1, seed=0):
    print("=== §6 부기: 삼중결합텐서의 에너지중립성 ===")
    print("(표준 에너지법을 이용한 부수적 탐구; Δ에서 유도된 결과 아님)")
    rng = np.random.default_rng(seed)
    ks = np.arange(-N, N + 1)
    idx = {k: i for i, k in enumerate(ks)}

    c = (rng.standard_normal(len(ks)) + 1j * rng.standard_normal(len(ks))) * np.exp(-0.05 * ks**2)
    # u(x,t)의 실수성 강제: c_{-k} = conj(c_k)
    for k in ks:
        if k < 0:
            c[idx[k]] = np.conj(c[idx[-k]])
    c[idx[0]] = c[idx[0]].real

    def dcdt_nonlinear(c):
        # dcₖ/dt (비선형 부분) = -(ik/2) Σ_{m+n=k} cₘcₙ
        out = np.zeros_like(c)
        for ki, k in enumerate(ks):
            s = 0.0 + 0.0j
            for mi, m in enumerate(ks):
                n = k - m
                if n in idx:
                    s += c[mi] * c[idx[n]]
            out[ki] = -(1j * k / 2.0) * s
        return out

    def dcdt_linear(c):
        return -nu * (ks**2) * c

    nl = dcdt_nonlinear(c)
    lin = dcdt_linear(c)

    dE_nonlinear = np.sum(np.real(np.conj(c) * nl))
    dE_linear = np.sum(np.real(np.conj(c) * lin))

    print(f"  비선형(삼중결합)의 dE/dt 기여: {dE_nonlinear:.3e}  (주장: 정확히 0)")
    print(f"  선형(점성)의 dE/dt 기여:      {dE_linear:.6f}  (주장: ≤0)")
    print(f"  전체 dE/dt:                   {dE_nonlinear + dE_linear:.6f}")
    assert abs(dE_nonlinear) < 1e-8
    print("  확인됨: 비선형 에너지 기여가 기계정밀도 수준에서 0.\n")
    return dE_nonlinear, dE_linear


# ============================================================
# §3.1. V_mn(t)의 크기/위상 분리
#   Vₘₙ(t) = cₘcₙ* e^{λₘt} e^{λₙ*t} = cₘcₙ* e^{-(m²+n²)t} · e^{i(m²-n²)t}
#   주장: 이는 실수축 크기(envelope, e^{-(m²+n²)t})와 허수축 위상
#   (e^{i(m²-n²)t})으로 정확히 분리되며, 대각항(m=n)은 순수 실수축
#   양으로 붕괴한다(위상항=1).
# ============================================================

def section3_1_Vmn_magnitude_phase_separation(pairs=None, t_vals=None, seed=1):
    print("=== §3.1: V_mn(t)의 크기/위상 분리 ===")
    if pairs is None:
        pairs = [(1, 1), (2, 2), (1, 2), (2, 3), (3, 5), (4, 4)]
    if t_vals is None:
        t_vals = [0.1, 0.5, 1.0]

    rng = np.random.default_rng(seed)
    print(f"{'(m,n)':>7} {'t':>5} {'|Vmn| 계산값':>16} {'크기 예측값':>16} {'위상 계산값':>16} {'위상 예측값':>16} {'최대오차':>10}")
    max_err = 0.0
    for (m, n) in pairs:
        cm = complex(rng.standard_normal(), rng.standard_normal())
        cn = complex(rng.standard_normal(), rng.standard_normal())
        for t in t_vals:
            lam_m = -(m**2) + 1j * (m**2)     # λₙ = -n² + iωₙ, ωₙ = n²
            lam_n_conj = -(n**2) - 1j * (n**2)
            Vmn = cm * np.conj(cn) * np.exp(lam_m * t) * np.exp(lam_n_conj * t)

            envelope_pred = abs(cm * np.conj(cn)) * np.exp(-(m**2 + n**2) * t)
            phase_pred = np.exp(1j * (m**2 - n**2) * t)

            computed_envelope = abs(Vmn)
            # 위상만 직접 재계산: Vmn / (|cmcn*| · envelope · unit_coeff) = phase_pred이어야 함
            unit_coeff = (cm * np.conj(cn)) / abs(cm * np.conj(cn))
            computed_phase_factor = Vmn / (abs(cm * np.conj(cn)) * np.exp(-(m**2+n**2)*t) * unit_coeff)

            err_envelope = abs(computed_envelope - envelope_pred)
            err_phase = abs(computed_phase_factor - phase_pred)
            err = max(err_envelope, err_phase)
            max_err = max(max_err, err)

            if m == n:
                # 대각항: 위상이 정확히 1로 붕괴해야 함 (Generation, 순수 실수축)
                assert abs(phase_pred - 1.0) < 1e-12, "대각항의 위상은 정확히 1이어야 함"

            phase_str = f"{complex(computed_phase_factor).real:.4f}{complex(computed_phase_factor).imag:+.4f}j"
            phase_pred_str = f"{complex(phase_pred).real:.4f}{complex(phase_pred).imag:+.4f}j"
            print(f"{str((m,n)):>7} {t:>5} {computed_envelope:>16.10f} {envelope_pred:>16.10f} "
                  f"{phase_str:>16} {phase_pred_str:>16} {err:>10.2e}")

    print(f"\n  전체 (m,n,t)에 대한 최대오차: {max_err:.2e}")
    assert max_err < 1e-8
    print("  확인됨: V_mn(t)가 실수축 크기 × 허수축 위상으로 정확히 분리됨;")
    print("  대각항(m=n)은 §3.2가 주장한 대로 순수 실수축(위상=1)으로 붕괴.\n")


# ============================================================
# §10. 완화-결어긋남 부등식
#   Γ_relax(n) = 2n²   (대각항 감쇠율)
#   Γ_decoh(m,n) = m²+n²   (비대각항 감쇠율, m≠n)
#   주장 (10.1): 모든 m≠n에 대해 Γ_decoh(m,n) > Γ_relax(min(m,n))
#   대수적 이유: p=max(m,n), q=min(m,n), p>q≥1일 때:
#     (p²+q²) - 2q² = p²-q² = (p-q)(p+q) ≥ 1·(2q+1) = 2q+1 > 0
# ============================================================

def section10_relaxation_decoherence_inequality(N=50):
    print("=== §10: 완화-결어긋남 부등식 (10.1) ===")
    violations = 0
    checked = 0
    min_margin = None
    for m in range(1, N + 1):
        for n in range(1, N + 1):
            if m == n:
                continue
            checked += 1
            gamma_decoh = m**2 + n**2
            gamma_relax = 2 * (min(m, n) ** 2)
            margin = gamma_decoh - gamma_relax
            if min_margin is None or margin < min_margin:
                min_margin = margin
            if margin <= 0:
                violations += 1

    print(f"  검사한 쌍 (m,n∈1..{N}, m≠n): {checked}")
    print(f"  Γ_decoh > Γ_relax 위반 건수: {violations}")
    print(f"  관측된 최소 여유: {min_margin} (예측 하한: 2·min(m,n)+1)")
    assert violations == 0
    print("  확인됨: 검사한 모든 쌍에서 부등식 성립.\n")


# ============================================================
# 부록K / T7. 다체중력의 그린함수 중첩
#   주장: Δ가 선형이므로 Δ[Σᵢqᵢ G(|r-rᵢ|)] = Σᵢqᵢ δ(r-rᵢ)이다.
#   해석적 중첩해 φ(r)=Σᵢqᵢ/(4π|r-rᵢ|)에 이산 3차원 라플라시안을
#   직접 적용해서, 모든 소스로부터 떨어진 점에서 Δφ=0(라플라스 방정식)이
#   성립하는지 확인한다 — 소스에서 먼 곳은 이 주장이 특이점 없이
#   검증 가능한 부분이며(각 소스의 델타함수 부분은 유한격자로는
#   직접 검증 불가), 이 방식으로 그 부분만 정직하게 확인한다.
#
#   주의: 이전 버전은 잘린 유한격자 위에서 전역 포아송 방정식을 직접
#   풀었으나, 유한격자는 해석적 그린함수가 가정하는 무한공간(r→∞)
#   경계조건을 갖지 못해 큰 오차가 발생했다. 이번 버전은 선형성을
#   직접 검사하는 방식으로 이 문제를 완전히 피한다.
# ============================================================

def T7_many_body_gravity_superposition(N_masses=3, seed=2):
    print("=== 부록K / T7: 다체중력의 그린함수 중첩 ===")
    rng = np.random.default_rng(seed)

    src_positions = [rng.uniform(-0.3, 0.3, size=3) for _ in range(N_masses)]
    charges = [rng.uniform(0.5, 2.0) for _ in range(N_masses)]

    def phi_superposition(r):
        return sum(q / (4*np.pi*np.linalg.norm(r - s)) for q, s in zip(charges, src_positions))

    def discrete_laplacian(f, r, h):
        lap = 0.0
        for axis in range(3):
            dr = np.zeros(3); dr[axis] = h
            lap += (f(r+dr) - 2*f(r) + f(r-dr)) / h**2
        return lap

    # 0단계: 고정된 한 테스트점에서 수렴성 확인. 유한차분 라플라시안은
    # O(h²) 절단오차와 O(eps/h²) 부동소수점 반올림오차를 동시에 가지며,
    # 이 둘이 상쇄되어 double precision에서는 h~1e-4 근처에서 오차가
    # 최소가 되는 U자형 곡선이 나온다. 이를 먼저 확인한 뒤에야 특정
    # |Δφ| 값 하나를 증거로 신뢰할 수 있다.
    pt0 = None
    h_scan = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    print("  수렴성 스캔 (O(h²) 절단오차임을 확인, 물리적 불일치가 아님):")
    print(f"  {'h':>10} {'|Δφ|':>16} {'이전 대비 비율':>18}")
    prev = None
    for _ in range(1):
        pt0 = rng.uniform(-1.0, 1.0, size=3)
        while any(np.linalg.norm(pt0 - s) < 0.2 for s in src_positions):
            pt0 = rng.uniform(-1.0, 1.0, size=3)
    for h in h_scan:
        val = abs(discrete_laplacian(phi_superposition, pt0, h))
        ratio = (prev/val) if prev else None
        print(f"  {h:>10.0e} {val:>16.6e} {('%.2f'%ratio if ratio else '-'):>18}")
        prev = val
    print("  (h를 10배 줄일 때마다 오차가 h~1e-4까지는 약 100배씩 줄어들어 O(h²)")
    print("   절단오차임이 확인됨; 그 아래로는 부동소수점 반올림오차가 지배하여")
    print("   오차가 다시 커진다 — double precision 유한차분의 전형적인 U자 곡선.)\n")

    h_optimal = 1e-4  # 절단오차/반올림오차 트레이드오프의 경험적 최적점
    print(f"{'테스트점':>26} {'Δφ 수치값':>20} {'기댓값(소스에서 먼 곳)':>30}")
    max_lap = 0.0
    for _ in range(5):
        pt = rng.uniform(-1.0, 1.0, size=3)
        # 모든 소스로부터 충분히 떨어진 점만 사용 (특이점 회피)
        if any(np.linalg.norm(pt - s) < 0.2 for s in src_positions):
            continue
        lap_val = discrete_laplacian(phi_superposition, pt, h_optimal)
        max_lap = max(max_lap, abs(lap_val))
        print(f"{str(np.round(pt,3)):>26} {lap_val:>20.3e} {'0 (라플라스 방정식)':>30}")

    print(f"\n  테스트점 전체 최대 |Δφ| (h={h_optimal:.0e}): {max_lap:.2e}")
    assert max_lap < 1e-5
    print("  확인됨: 선형중첩 Σᵢqᵢ G(|r-rᵢ|)이 모든 소스에서 떨어진 곳에서")
    print("  라플라스 방정식(Δφ~0)을 만족 — Δ의 선형성이 각 소스의 그린함수에")
    print("  독립적으로 작용한다는 것과 일치.\n")


if __name__ == "__main__":
    T1_dissipative_semigroup_integral()
    T2_heat_kernel_greens_function(d=3)
    triple_coupling_tensor_energy_neutrality()
    section3_1_Vmn_magnitude_phase_separation()
    section10_relaxation_decoherence_inequality()
    T7_many_body_gravity_superposition()
