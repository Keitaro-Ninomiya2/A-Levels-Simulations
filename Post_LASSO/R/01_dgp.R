# ==============================================================================
# 01_dgp.R — Data Generation Process
# ==============================================================================
#
# Simulates an A-level panel: ~200k students/year x 8 years x ~3,000 schools.
#   Years 1-7: Stationary exam-based grading.
#   Year 8:    COVID teacher-assessed grades (inflation + school-specific bias).
#
# TRUE DGP (only 4 covariates matter):
#   Historical:  P(Y=1) = Lambda(alpha_j + X'eta)
#   COVID:       P(Y=1) = Lambda(alpha_j + Delta_alpha_j
#                                + X'(eta + Delta_eta)
#                                + beta_ses_j * SES + beta_min_j * Minority)
#
# ESTIMATION covariates (30 total = 4 true + 1 poly + 6 interactions + 19 noise)
#   Only the 4 true covariates have nonzero DGP coefficients.
#   The extra 26 create the overfitting opportunity that the LASSO corrects.
#
# Exports:
#   default_params()               - Full parameter list (modify before passing)
#   generate_school_structure()    - Fixed school data; call ONCE per simulation
#   generate_panel()               - Student panel; call ONCE PER REPLICATION
#   get_covariate_names()          - Character vector of the 30 estimation covariates
#   verify_dgp()                   - Print ground-truth diagnostics
# ==============================================================================

.libPaths(c(file.path(Sys.getenv("HOME"), "A-Levels", "R_libs"), .libPaths()))
library(data.table)

# ------------------------------------------------------------------------------
# 1. DEFAULT PARAMETERS
# ------------------------------------------------------------------------------
default_params <- function() {
  list(
    # ---- School structure ----
    J = 3000L,
    type_shares = c(State = 0.60, Academy = 0.33, Independent = 0.07),

    # Students per school per year ~ Uniform(lo, hi)
    size_range = list(
      State       = c(30L, 150L),
      Academy     = c(25L, 120L),
      Independent = c(15L, 80L)
    ),

    # School quality: alpha_j ~ N(mu, sigma^2), by type
    alpha_dist = list(
      State       = c(mu = -0.30, sigma = 0.50),
      Academy     = c(mu =  0.10, sigma = 0.60),
      Independent = c(mu =  1.20, sigma = 0.30)
    ),

    # ---- Timeline ----
    T_hist  = 7L,
    T_covid = 1L,
    
    # Students per year (FIXED back to full scale)
    n_students_per_year = 200000L,

    # ---- True coefficients (eta) ----
    eta = c(GCSE_std = 0.80, SES = 0.40, Minority = -0.20, Gender = 0.10),

    # ---- COVID inflation ----
    # Delta_alpha_j = delta_0 + delta_1 * alpha_j + nu_j
    delta_0  = 1.0,    # general upward shift (logit scale)
    delta_1  = -0.3,   # ceiling effect: high-alpha schools inflate less
    sigma_nu = 0.3,    # idiosyncratic school-level noise

    # Aggregate covariate shift during COVID
    delta_eta = c(GCSE_std = 0.15, SES = 0.10, Minority = 0.00, Gender = 0.00),

    # School-specific COVID bias (NOT in estimation model -> confound)
    bias_ses = list(
      State       = c(mu = 0.15, sigma = 0.10),
      Academy     = c(mu = 0.20, sigma = 0.12),
      Independent = c(mu = 0.35, sigma = 0.10)
    ),
    bias_min = list(
      State       = c(mu = -0.05, sigma = 0.08),
      Academy     = c(mu = -0.10, sigma = 0.08),
      Independent = c(mu = -0.15, sigma = 0.10)
    ),

    # ---- Student attribute distributions ----
    p_ses      = c(State = 0.30, Academy = 0.40, Independent = 0.80),
    p_minority = c(State = 0.30, Academy = 0.25, Independent = 0.15),
    gcse_slope = 0.3,  # E[GCSE_std | school j] = gcse_slope * alpha_j

    # ---- Estimation extras ----
    n_noise = 19L,     # iid N(0,1) noise covariates (true coeff = 0)

    # ---- Monte Carlo ----
    R = 50L
  )
}


# ------------------------------------------------------------------------------
# 2. GENERATE SCHOOL STRUCTURE  (call once; fixed across replications)
# ------------------------------------------------------------------------------
generate_school_structure <- function(params = default_params(), seed = 42L) {
  set.seed(seed)

  J      <- params$J
  shares <- params$type_shares
  types  <- names(shares)

  # --- Allocate schools to types ---
  n_per_type <- setNames(round(J * shares), types)
  n_per_type[1] <- J - sum(n_per_type[-1])   # absorb rounding residual

  type_vec <- rep(types, times = n_per_type)

  # --- School sizes, quality, and COVID bias (vectorised by type) ---
  size_vec  <- integer(J)
  alpha_vec <- numeric(J)
  bses_vec  <- numeric(J)
  bmin_vec  <- numeric(J)

  for (tp in types) {
    idx <- which(type_vec == tp)
    n   <- length(idx)

    # Size
    rng <- params$size_range[[tp]]
    size_vec[idx] <- sample(rng[1]:rng[2], n, replace = TRUE)

    # School quality
    ad <- params$alpha_dist[[tp]]
    alpha_vec[idx] <- rnorm(n, mean = ad["mu"], sd = ad["sigma"])

    # School-specific COVID bias on SES
    bs <- params$bias_ses[[tp]]
    bses_vec[idx] <- rnorm(n, mean = bs["mu"], sd = bs["sigma"])

    # School-specific COVID bias on Minority
    bm <- params$bias_min[[tp]]
    bmin_vec[idx] <- rnorm(n, mean = bm["mu"], sd = bm["sigma"])
  }

  # --- COVID school-level inflation ---
  # Delta_alpha_j = delta_0 + delta_1 * alpha_j + nu_j
  nu_vec     <- rnorm(J, mean = 0, sd = params$sigma_nu)
  dalpha_vec <- params$delta_0 + params$delta_1 * alpha_vec + nu_vec

  # --- Assemble school data.table ---
  school_dt <- data.table(
    school_id     = 1L:J,
    school_type   = factor(type_vec, levels = types),
    school_size   = size_vec,
    alpha_j       = alpha_vec,
    delta_alpha_j = dalpha_vec,
    beta_ses_j    = bses_vec,
    beta_min_j    = bmin_vec
  )

  # --- Ground truth: inflation for a reference student (X* = 0) ---
  school_dt[, baseline_prob    := plogis(alpha_j)]
  school_dt[, covid_prob       := plogis(alpha_j + delta_alpha_j)]
  school_dt[, true_inflation   := covid_prob - baseline_prob]
  school_dt[, true_rrr         := true_inflation / (1 - baseline_prob)]
  school_dt[, alpha_percentile := frank(alpha_j) / .N]

  school_dt[]
}


# ------------------------------------------------------------------------------
# 3. GENERATE STUDENT PANEL  (call once per replication)
# ------------------------------------------------------------------------------
generate_panel <- function(school_dt, params = default_params(), seed = NULL) {
  if (!is.null(seed)) set.seed(seed)

  T_total <- params$T_hist + params$T_covid
  eta     <- params$eta
  deta    <- params$delta_eta
  J       <- nrow(school_dt)
  
  # Use the n_students_per_year parameter if available, otherwise fallback/calculate
  # The original code calculated N_per_yr based on school sizes. 
  # However, to support exact N scaling, we can adjust the logic or just use the school sizes 
  # defined in generate_school_structure which sum to approx 200k if configured right.
  # But typically in these simulations we want to control N exactly.
  # Let's stick to the structure implied by school sizes to keep it consistent 
  # with the user's provided code structure, where N depends on school_size sum.
  
  # -- Pre-extract school-level vectors (avoid repeated $ lookup) --
  s_id     <- school_dt$school_id
  s_size   <- school_dt$school_size
  s_alpha  <- school_dt$alpha_j
  s_dalpha <- school_dt$delta_alpha_j
  s_bses   <- school_dt$beta_ses_j
  s_bmin   <- school_dt$beta_min_j
  s_type   <- school_dt$school_type

  # Per-school probabilities for SES / Minority draws (lookup once)
  p_ses_sch <- params$p_ses[as.character(s_type)]
  p_min_sch <- params$p_minority[as.character(s_type)]

  # Student-to-school index (same structure every year)
  sch_idx  <- rep(seq_len(J), times = s_size)
  N_per_yr <- length(sch_idx)

  # Expand school-level quantities to student-level (reused every year)
  st_alpha  <- s_alpha[sch_idx]
  st_dalpha <- s_dalpha[sch_idx]
  st_bses   <- s_bses[sch_idx]
  st_bmin   <- s_bmin[sch_idx]
  st_sid    <- s_id[sch_idx]
  st_p_ses  <- p_ses_sch[sch_idx]
  st_p_min  <- p_min_sch[sch_idx]

  # -- Build panel year by year --
  year_list <- vector("list", T_total)

  for (t in seq_len(T_total)) {
    is_covid <- (t > params$T_hist)

    # Draw student covariates
    GCSE <- rnorm(N_per_yr, mean = params$gcse_slope * st_alpha, sd = 1)
    SES  <- rbinom(N_per_yr, 1L, prob = st_p_ses)
    MIN  <- rbinom(N_per_yr, 1L, prob = st_p_min)
    GEN  <- rbinom(N_per_yr, 1L, prob = 0.5)

    # Latent index: alpha_j + X'eta
    latent <- st_alpha +
      eta["GCSE_std"] * GCSE +
      eta["SES"]      * SES  +
      eta["Minority"] * MIN  +
      eta["Gender"]   * GEN

    # COVID additions (year 8 only)
    if (is_covid) {
      latent <- latent +
        st_dalpha +                          # school-level inflation
        deta["GCSE_std"] * GCSE +        # aggregate covariate shift
        deta["SES"]      * SES  +
        deta["Minority"] * MIN  +
        deta["Gender"]   * GEN  +
        st_bses * SES +                  # school-specific SES bias
        st_bmin * MIN                    # school-specific minority bias
    }

    # Binary outcome
    Y <- rbinom(N_per_yr, 1L, prob = plogis(latent))

    year_list[[t]] <- data.table(
      year      = as.integer(t),
      school_id = st_sid,
      D_covid   = as.integer(is_covid),
      Y         = Y,
      GCSE_std  = GCSE,
      SES       = SES,
      Minority  = MIN,
      Gender    = GEN
    )
  }

  # Stack years
  panel <- rbindlist(year_list, use.names = TRUE)
  rm(year_list)

  # -- Add estimation covariates (polynomial + interactions) --
  panel[, GCSE_std_sq    := GCSE_std * GCSE_std]
  panel[, GCSE_SES       := GCSE_std * SES]
  panel[, GCSE_Minority  := GCSE_std * Minority]
  panel[, GCSE_Gender    := GCSE_std * Gender]
  panel[, SES_Minority   := SES * Minority]
  panel[, SES_Gender     := SES * Gender]
  panel[, Minority_Gender := Minority * Gender]

  # -- Noise covariates: iid N(0,1), zero true effect --
  N <- nrow(panel)
  for (k in seq_len(params$n_noise)) {
    set(panel, j = paste0("Z_", k), value = rnorm(N))
  }

  gc(verbose = FALSE)
  panel[]
}


# ------------------------------------------------------------------------------
# 4. HELPER: estimation covariate names
# ------------------------------------------------------------------------------
get_covariate_names <- function(params = default_params()) {
  base  <- c("GCSE_std", "SES", "Minority", "Gender",
             "GCSE_std_sq",
             "GCSE_SES", "GCSE_Minority", "GCSE_Gender",
             "SES_Minority", "SES_Gender", "Minority_Gender")
  noise <- paste0("Z_", seq_len(params$n_noise))
  c(base, noise)
}


# ------------------------------------------------------------------------------
# 5. VERIFICATION
# ------------------------------------------------------------------------------
verify_dgp <- function(school_dt, params = default_params()) {
  cat("====================================================================\n")
  cat("  DGP VERIFICATION\n")
  cat("====================================================================\n\n")

  # -- School counts --
  cat("[1] School counts by type:\n")
  print(school_dt[, .N, by = school_type])

  # -- Total students per year --
  cat(sprintf("\n[2] Total students per year: %s\n",
              format(sum(school_dt$school_size), big.mark = ",")))

  # -- Alpha distribution --
  cat("\n[3] School quality (alpha_j) by type:\n")
  print(school_dt[, .(mean    = round(mean(alpha_j), 3),
                        sd      = round(sd(alpha_j), 3),
                        q10     = round(quantile(alpha_j, 0.10), 3),
                        median = round(median(alpha_j), 3),
                        q90     = round(quantile(alpha_j, 0.90), 3)),
                  by = school_type])

  # -- Critical check: Independent schools concentrated at the top --
  n_indep       <- school_dt[school_type == "Independent", .N]
  n_indep_top30 <- school_dt[school_type == "Independent" &
                               alpha_percentile >= 0.70, .N]
  pct_top30 <- n_indep_top30 / n_indep

  cat(sprintf(
    "\n[4] CRITICAL — Independent schools above 70th percentile: %d / %d = %.1f%%",
    n_indep_top30, n_indep, 100 * pct_top30))
  cat(ifelse(pct_top30 >= 0.80, "  [PASS]\n", "  [FAIL]\n"))

  # -- Implied baseline pass rates --
  cat("\n[5] Implied baseline pass rate at X*=0 (reference student):\n")
  print(school_dt[, .(mean_baseline = round(mean(baseline_prob), 3),
                        mean_covid    = round(mean(covid_prob), 3),
                        mean_inflation_pp = round(mean(true_inflation), 3),
                        mean_rrr      = round(mean(true_rrr), 3)),
                  by = school_type])

  # -- Inflation summary --
  cat("\n[6] True inflation (prob. points) distribution by type:\n")
  print(school_dt[, .(mean = round(mean(true_inflation), 3),
                        sd   = round(sd(true_inflation), 3),
                        min  = round(min(true_inflation), 3),
                        max  = round(max(true_inflation), 3)),
                  by = school_type])

  cat("\n====================================================================\n")
  cat("  VERIFICATION COMPLETE\n")
  cat("====================================================================\n")
  invisible(NULL)
}


# ------------------------------------------------------------------------------
# 6. STANDALONE EXECUTION (runs only when called as: Rscript 01_dgp.R)
# ------------------------------------------------------------------------------
if (sys.nframe() == 0L) {
  params <- default_params()

  cat("--- Generating school structure ---\n")
  school_dt <- generate_school_structure(params)
  verify_dgp(school_dt, params)

  cat("\n--- Generating one panel (seed = 1) ---\n")
  t0    <- proc.time()
  panel <- generate_panel(school_dt, params, seed = 1L)
  elapsed <- (proc.time() - t0)["elapsed"]
  cat(sprintf("Panel: %s rows x %d cols  [%.1f sec]\n",
              format(nrow(panel), big.mark = ","), ncol(panel), elapsed))

  cat("\nOutcome rate by year:\n")
  print(panel[, .(pass_rate = round(mean(Y), 4), N = format(.N, big.mark = ",")),
              by = year])

  cat("\nOutcome rate by school type x COVID:\n")
  tmp <- merge(panel[, .(pass_rate = mean(Y), N = .N),
                     by = .(school_id, D_covid)],
               school_dt[, .(school_id, school_type)],
               by = "school_id")
  print(tmp[, .(mean_pass = round(mean(pass_rate), 4)), by = .(school_type, D_covid)])
  rm(tmp)

  cat("\nEstimation covariate names (p = ", length(get_covariate_names(params)), "):\n")
  cat(paste(get_covariate_names(params), collapse = ", "), "\n")

  cat("\nDone.\n")
}
