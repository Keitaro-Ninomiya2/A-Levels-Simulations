# Simulation Specification: Type-Specific Demographic Slopes in Hierarchical Logit

## Problem
When estimating a global hierarchical logit with EB shrinkage, the non-GCSE demographic slopes (gender, race, SES) are estimated mostly from state-school data. Applying these slopes to independent-school students (who have very different demographic compositions) produces incorrect baseline probabilities. This biases the probability-scale risk reductions for independent schools.

## Current Model (Eq. 1 in paper)
```
z = α_j + X_i·η + γ_k + Σ_τ D_τ·(Δα_{j,τ} + X_i·Δη_τ + Δγ_{k,τ})
```
- i = student, j = school, k = subject, t = year
- Y = 1{top grade (A/A*)}
- α_j = school FE, γ_k = subject FE
- η = common slopes on student controls X_i (GCSE, gender, race, SES)
- Δα_{j,τ} = school-specific treatment effect (EB-shrunk)
- F = logistic CDF

## Key Identification Constraint
- **GCSE slope must be common** across school types. Exam boards calibrate grade boundaries to GCSE, so the GCSE→grade mapping is institutionally the same everywhere.
- **Non-GCSE demographic slopes** (gender, race, SES) have no such institutional anchor and may legitimately differ by school type.

## Proposed Fix
Split X_i into GCSE and demographic components. Interact demographics with school-type indicators:

```
z = α_j 
    + X_i^{GCSE} · η^{GCSE}           [common, not penalized]
    + X_i^{demo} · η^{demo}            [common baseline, not penalized]
    + X_i^{demo} · 1[type_s] · η_s     [type-specific deviations, LASSO-penalized]
    + γ_k 
    + treatment terms
```

## LASSO Strategy
- **Unpenalized (penalty.factor = 0):** GCSE slopes, school FEs, subject FEs, treatment terms (Δα, Δη, Δγ), common demographic slopes
- **Penalized:** demographic × school-type interactions only
- **Effect:** LASSO zeros out interactions where the common slope suffices, keeps interactions where types genuinely differ
- **CV:** within-sample cross-validation for λ selection

## What to Verify in Simulation
1. **DGP:** Generate data where demographic slopes differ by school type but GCSE slopes are common. Independent schools have higher baseline probabilities (~70-80%) vs state schools (~20-30%).
2. **Estimator A (global model):** Common η for all X_i. Compute risk reductions. Check if independent-school effects are inflated.
3. **Estimator B (split by type):** Separate models by type. Check correct risk reductions but note scale incomparability.
4. **Estimator C (proposed fix):** Joint model with LASSO on demographic × type interactions. Verify:
   - Risk reductions for independents are corrected
   - All estimates remain on one logit scale
   - Treatment effects (Δα) are comparable across types
   - GCSE slope stays common (identification preserved)

## Key References
- Wooldridge (2010) Ch. 15.7.1 -- neglected heterogeneity in nonlinear models
- Yatchew & Griliches (1985) -- specification error in probit
- Chetty & Hendren (2018) -- EB with group-specific priors
- Angrist, Hull, Pathak & Walters (2017) -- school VA with sector-specific EB
