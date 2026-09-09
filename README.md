# delta-pl

Optical Eyez XL, *소산 사영(dissipative projection)* 논문(physical_law / [PL])의
검증 스크립트 모음. 이 논문은 자매논문 *Preservation of Distinguishability*
([PD])에서 확립된 단일 자기수반 소산 생성자 Δeₙ=n²eₙ으로부터 고전장의
귀결들을 도출한다.

한글 주석판과 영문 번역판을 모두 포함한다.

## 파일 구성

| 파일 | 내용 |
|---|---|
| `physical_law_verification.py` / `physical_law_verification_en.py` | 다음을 검증한다: T1/T2(소산 반군의 시간적분과 d=3 뉴턴/쿨롱 그린함수), §6.2(나비에-스토크스 방정식의 Δ 부호 검증 및 Taylor-Green 압력해 재검산 — 이전 판 압력해의 오류 확인 포함), §3.1(V_mn(t)이 실수축 크기와 허수축 위상으로 분리됨), §10(완화-결어긋남 부등식, 2450개 모드쌍 전수검사), 부록K/T7(다체중력을 그린함수 중첩으로 — 수렴성까지 확인한 유한차분 라플라시안으로 검증), §5.4(T3와 전하보존으로부터 변위전류가 대수적으로 강제됨), 그리고 (Δ 구조 자체에서 나온 결과가 아니라 **표준 에너지법을 이용한 부수적 탐구**인) 이류항이 만드는 삼중결합텐서의 에너지중립성(§6 부기). |

## 실행 방법

```bash
pip install mpmath numpy sympy
python physical_law_verification.py
```

스크립트 안의 모든 assert 문은 논문에서 주장하는 수치적 결과에 대응한다 —
`AssertionError` 없이 실행이 끝나면 논문에 보고된 모든 결과가 재현된 것이다.
