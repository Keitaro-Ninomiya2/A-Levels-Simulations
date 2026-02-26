# ==============================================================================
# 02_estimator.R — Three Estimators for Type-Specific Slopes DGP
# ==============================================================================
#
# Estimator A: Global Model — common demographic slopes (misspecified)
# Estimator B: Split by Type — separate models per school type
# Estimator C: Proposed Fix — LASSO on Demo × Type interactions
#
# Uses the sparse block IRLS solver from run_single.R for efficient estimation
# with school fixed effects.
# ==============================================================================

.libPaths(c(file.path(Sys.getenv("HOME"), "A-Levels", "R_libs"), .libPaths()))
library(data.table)
library(Matrix)
library(glmnet)

source(file.path(Sys.getenv("HOME"), "A-Levels", "school_type_fix",
                 "Post_LASSO", "R", "01_dgp.R"))


# ==============================================================================
# 1. SPARSE BLOCK IRLS SOLVER
# ==============================================================================
# Exploits block-diagonal structure of X'WX for school FEs.
# Solves via Schur complement: O(p^3 + Jp^2) instead of O((2J+2p)^3)
#
# Sparse d_covid optimisation: d_covid is binary (1 for COVID year only,
# ~12.5% of rows). All COVID-weighted operations are restricted to the
# pre-indexed COVID subset, avoiding ~87.5% redundant computation.

sparse_block_irls <- function(sid, d_covid, Z_base, y,
                              use_covid_cov = TRUE,
                              ridge = 1e-3,
                              tol = 1e-8, maxit = 200) {
  N <- length(y)
  J <- max(sid)
  p <- if (is.null(Z_base)) 0L else ncol(Z_base)
  has_cov <- p > 0

  # Pre-index COVID rows once (binary treatment => sparse subset)
  ci <- which(d_covid == 1L)
  N_c <- length(ci)
  sid_c <- sid[ci]
  Z_c <- if (has_cov) Z_base[ci, , drop = FALSE] else NULL

  # Safe rowsum for COVID subset: guarantees J-length output even if some
  # schools have zero COVID students (fills those with 0)
  all_schools_in_c <- (length(unique(sid_c)) == J)
  rowsum_c <- if (all_schools_in_c) {
    function(x, ...) drop(rowsum(x, sid_c, reorder = TRUE))
  } else {
    function(x, ...) {
      raw <- rowsum(x, sid_c, reorder = TRUE)
      out <- numeric(J)
      out[as.integer(rownames(raw))] <- drop(raw)
      out
    }
  }
  # Matrix version (for wZ_c -> U_c)
  rowsum_c_mat <- if (all_schools_in_c) {
    function(x) rowsum(x, sid_c, reorder = TRUE)
  } else {
    function(x) {
      raw <- rowsum(x, sid_c, reorder = TRUE)
      out <- matrix(0, J, ncol(x))
      out[as.integer(rownames(raw)), ] <- raw
      out
    }
  }

  alpha <- rep(0, J)
  delta <- rep(0, J)
  gamma_b <- if (has_cov) rep(0, p) else numeric(0)
  gamma_c <- if (has_cov && use_covid_cov) rep(0, p) else numeric(0)

  for (iter in seq_len(maxit)) {
    # Linear predictor: only add delta on COVID rows
    eta <- alpha[sid]
    if (has_cov) eta <- eta + drop(Z_base %*% gamma_b)
    eta[ci] <- eta[ci] + delta[sid_c]
    if (has_cov && use_covid_cov) eta[ci] <- eta[ci] + drop(Z_c %*% gamma_c)

    mu <- plogis(eta)
    w  <- pmax(mu * (1 - mu), 1e-10)
    z  <- eta + (y - mu) / w
    wz <- w * z

    # School-level sufficient statistics (COVID sums use subset)
    w_j     <- drop(rowsum(w, sid, reorder = TRUE))
    w_j_c   <- rowsum_c(w[ci])
    f_alpha <- drop(rowsum(wz, sid, reorder = TRUE))
    f_delta <- rowsum_c(wz[ci])

    # A^{-1} (J independent 2x2 blocks with ridge)
    w_j_r   <- w_j + ridge
    w_j_c_r <- w_j_c + ridge
    det_j <- pmax(w_j_r * w_j_c_r - w_j_c^2, 1e-12)
    ai11 <- w_j_c_r / det_j
    ai12 <- -w_j_c / det_j
    ai22 <- w_j_r / det_j

    if (has_cov) {
      # Full-sample covariate block
      wZ_all <- w * Z_base
      U_b  <- rowsum(wZ_all, sid, reorder = TRUE)
      B11  <- crossprod(Z_base, wZ_all)
      g_b  <- drop(crossprod(Z_base, wz))

      # COVID-subset covariate block (replaces d_covid * wZ over all N)
      w_c  <- w[ci]
      wz_c <- wz[ci]
      wZ_c <- w_c * Z_c
      U_c  <- rowsum_c_mat(wZ_c)
      B12  <- crossprod(Z_c, wZ_c)
      g_c  <- if (use_covid_cov) drop(crossprod(Z_c, wz_c)) else numeric(0)
      rm(wZ_all, wZ_c)

      if (use_covid_cov) {
        U_full <- cbind(U_b, U_c)
        V_full <- cbind(U_c, U_c)
        B_full <- rbind(cbind(B11, B12), cbind(B12, B12))
        g_full <- c(g_b, g_c)

        CtAiC <- crossprod(U_full, ai11 * U_full) +
                 crossprod(U_full, ai12 * V_full) +
                 crossprod(V_full, ai12 * U_full) +
                 crossprod(V_full, ai22 * V_full)
        S <- B_full - CtAiC
        diag(S) <- diag(S) + 1e-10

        Ainv_fa <- ai11 * f_alpha + ai12 * f_delta
        Ainv_fd <- ai12 * f_alpha + ai22 * f_delta
        CtAif <- drop(crossprod(U_full, Ainv_fa) + crossprod(V_full, Ainv_fd))
        gamma_full_new <- drop(solve(S, g_full - CtAif))
        gamma_b_new <- gamma_full_new[1:p]
        gamma_c_new <- gamma_full_new[(p + 1):(2 * p)]
        Cg_a <- drop(U_full %*% gamma_full_new)
        Cg_d <- drop(V_full %*% gamma_full_new)
      } else {
        CtAiC <- crossprod(U_b, ai11 * U_b) +
                 crossprod(U_b, ai12 * U_c) +
                 crossprod(U_c, ai12 * U_b) +
                 crossprod(U_c, ai22 * U_c)
        S <- B11 - CtAiC
        diag(S) <- diag(S) + 1e-10

        Ainv_fa <- ai11 * f_alpha + ai12 * f_delta
        Ainv_fd <- ai12 * f_alpha + ai22 * f_delta
        CtAif <- drop(crossprod(U_b, Ainv_fa) + crossprod(U_c, Ainv_fd))
        gamma_b_new <- drop(solve(S, g_b - CtAif))
        gamma_c_new <- numeric(0)
        Cg_a <- drop(U_b %*% gamma_b_new)
        Cg_d <- drop(U_c %*% gamma_b_new)
      }

      alpha_new <- ai11 * (f_alpha - Cg_a) + ai12 * (f_delta - Cg_d)
      delta_new <- ai12 * (f_alpha - Cg_a) + ai22 * (f_delta - Cg_d)
    } else {
      alpha_new <- ai11 * f_alpha + ai12 * f_delta
      delta_new <- ai12 * f_alpha + ai22 * f_delta
      gamma_b_new <- numeric(0)
      gamma_c_new <- numeric(0)
    }

    max_change <- max(abs(c(alpha_new - alpha, delta_new - delta,
                            gamma_b_new - gamma_b, gamma_c_new - gamma_c)))
    alpha   <- alpha_new
    delta   <- delta_new
    gamma_b <- gamma_b_new
    gamma_c <- gamma_c_new
    if (max_change < tol) break
  }

  converged <- (max_change < tol)
  cat(sprintf("      %s in %d iters (max change: %.2e)\n",
              if (converged) "Converged" else "WARNING: not converged",
              iter, max_change))

  list(alpha = alpha, delta = delta, gamma_b = gamma_b, gamma_c = gamma_c,
       iter = iter, converged = converged)
}


# ==============================================================================
# 2. STANDARD ERRORS (block Schur complement)
# ==============================================================================
# Same sparse d_covid optimisation as the solver above.
compute_block_se <- function(sid, d_covid, Z_base, fit,
                             use_covid_cov = TRUE, ridge = 1e-3) {
  J <- max(sid)

  # Pre-index COVID rows
  ci    <- which(d_covid == 1L)
  sid_c <- sid[ci]
  Z_c   <- Z_base[ci, , drop = FALSE]

  # Safe COVID-subset rowsum (handles schools with no COVID students)
  all_schools_in_c <- (length(unique(sid_c)) == J)
  rowsum_c <- if (all_schools_in_c) {
    function(x) drop(rowsum(x, sid_c, reorder = TRUE))
  } else {
    function(x) {
      raw <- rowsum(x, sid_c, reorder = TRUE)
      out <- numeric(J)
      out[as.integer(rownames(raw))] <- drop(raw)
      out
    }
  }
  rowsum_c_mat <- if (all_schools_in_c) {
    function(x) rowsum(x, sid_c, reorder = TRUE)
  } else {
    function(x) {
      raw <- rowsum(x, sid_c, reorder = TRUE)
      out <- matrix(0, J, ncol(x))
      out[as.integer(rownames(raw)), ] <- raw
      out
    }
  }

  eta <- fit$alpha[sid]
  if (length(fit$gamma_b) > 0)
    eta <- eta + drop(Z_base %*% fit$gamma_b)
  eta[ci] <- eta[ci] + fit$delta[sid_c]
  if (use_covid_cov && length(fit$gamma_c) > 0)
    eta[ci] <- eta[ci] + drop(Z_c %*% fit$gamma_c)

  mu <- plogis(eta)
  w  <- pmax(mu * (1 - mu), 1e-10)

  w_j     <- drop(rowsum(w, sid, reorder = TRUE))
  w_j_c   <- rowsum_c(w[ci])
  w_j_r   <- w_j + ridge
  w_j_c_r <- w_j_c + ridge
  det_j   <- pmax(w_j_r * w_j_c_r - w_j_c^2, 1e-12)
  ai11 <- w_j_c_r / det_j
  ai12 <- -w_j_c / det_j
  ai22 <- w_j_r / det_j

  p <- ncol(Z_base)
  if (!is.null(Z_base) && p > 0) {
    w_c  <- w[ci]

    wZ_all <- w * Z_base
    U_b  <- rowsum(wZ_all, sid, reorder = TRUE)
    B11  <- crossprod(Z_base, wZ_all)
    rm(wZ_all)

    wZ_c <- w_c * Z_c
    U_c  <- rowsum_c_mat(wZ_c)
    B12  <- crossprod(Z_c, wZ_c)
    rm(wZ_c)

    if (use_covid_cov) {
      U_full <- cbind(U_b, U_c)
      V_full <- cbind(U_c, U_c)
      B_full <- rbind(cbind(B11, B12), cbind(B12, B12))
      CtAiC <- crossprod(U_full, ai11 * U_full) +
               crossprod(U_full, ai12 * V_full) +
               crossprod(V_full, ai12 * U_full) +
               crossprod(V_full, ai22 * V_full)
      S <- B_full - CtAiC
      diag(S) <- diag(S) + 1e-10
      Sinv <- solve(S)
      T1 <- ai11 * U_full + ai12 * V_full
      T2 <- ai12 * U_full + ai22 * V_full
    } else {
      CtAiC <- crossprod(U_b, ai11 * U_b) +
               crossprod(U_b, ai12 * U_c) +
               crossprod(U_c, ai12 * U_b) +
               crossprod(U_c, ai22 * U_c)
      S <- B11 - CtAiC
      diag(S) <- diag(S) + 1e-10
      Sinv <- solve(S)
      T1 <- ai11 * U_b + ai12 * U_c
      T2 <- ai12 * U_b + ai22 * U_c
    }
    var_alpha <- ai11 + rowSums((T1 %*% Sinv) * T1)
    var_delta <- ai22 + rowSums((T2 %*% Sinv) * T2)
  } else {
    var_alpha <- ai11
    var_delta <- ai22
  }

  list(se_alpha = sqrt(pmax(var_alpha, 0)),
       se_delta = sqrt(pmax(var_delta, 0)))
}


# ==============================================================================
# 2b. POST-HOC FIRTH BIAS CORRECTION
# ==============================================================================
# One-step correction: β_F = β_MLE + (X'WX)^{-1} X' h (0.5 - μ)
# where h_ii is the hat matrix diagonal.  Reduces O(1/n_j) logit bias
# for school-level alpha_j and delta_j without re-running IRLS.
firth_correct <- function(sid, d_covid, Z_base, fit, use_covid_cov = TRUE) {
  J <- max(sid)
  N <- length(sid)
  p <- ncol(Z_base)

  ci    <- which(d_covid == 1L)
  sid_c <- sid[ci]
  Z_c   <- Z_base[ci, , drop = FALSE]

  all_schools_in_c <- (length(unique(sid_c)) == J)
  rowsum_c <- if (all_schools_in_c) {
    function(x) drop(rowsum(x, sid_c, reorder = TRUE))
  } else {
    function(x) {
      raw <- rowsum(x, sid_c, reorder = TRUE)
      out <- numeric(J); out[as.integer(rownames(raw))] <- drop(raw); out
    }
  }
  rowsum_c_mat <- if (all_schools_in_c) {
    function(x) rowsum(x, sid_c, reorder = TRUE)
  } else {
    function(x) {
      raw <- rowsum(x, sid_c, reorder = TRUE)
      out <- matrix(0, J, ncol(x)); out[as.integer(rownames(raw)), ] <- raw; out
    }
  }

  # Fitted values at MLE
  eta <- fit$alpha[sid]
  if (p > 0) eta <- eta + drop(Z_base %*% fit$gamma_b)
  eta[ci] <- eta[ci] + fit$delta[sid_c]
  if (use_covid_cov && p > 0) eta[ci] <- eta[ci] + drop(Z_c %*% fit$gamma_c)

  mu <- plogis(eta)
  w  <- pmax(mu * (1 - mu), 1e-10)

  # A^{-1} blocks (no ridge — exact Fisher information)
  w_j   <- drop(rowsum(w, sid, reorder = TRUE))
  w_j_c <- rowsum_c(w[ci])
  det_j <- pmax(w_j * w_j_c - w_j_c^2, 1e-12)
  ai11  <- w_j_c / det_j
  ai12  <- -w_j_c / det_j
  ai22  <- w_j / det_j

  # Coupling matrices
  w_c    <- w[ci]
  wZ_all <- w * Z_base
  U_b    <- rowsum(wZ_all, sid, reorder = TRUE)
  B11    <- crossprod(Z_base, wZ_all)
  rm(wZ_all)
  wZ_c <- w_c * Z_c
  U_c  <- rowsum_c_mat(wZ_c)
  B12  <- crossprod(Z_c, wZ_c)
  rm(wZ_c)

  if (use_covid_cov) {
    U_full <- cbind(U_b, U_c)
    V_full <- cbind(U_c, U_c)
    B_full <- rbind(cbind(B11, B12), cbind(B12, B12))
    CtAiC  <- crossprod(U_full, ai11 * U_full) +
              crossprod(U_full, ai12 * V_full) +
              crossprod(V_full, ai12 * U_full) +
              crossprod(V_full, ai22 * V_full)
    S <- B_full - CtAiC
    diag(S) <- diag(S) + 1e-10
    Sinv <- solve(S)

    T1     <- ai11 * U_full + ai12 * V_full
    T2     <- ai12 * U_full + ai22 * V_full
    T1Sinv <- T1 %*% Sinv
    T2Sinv <- T2 %*% Sinv

    # Full marginal variances per school
    var_a  <- ai11 + rowSums(T1Sinv * T1)
    var_d  <- ai22 + rowSums(T2Sinv * T2)
    cov_ad <- ai12 + rowSums(T1Sinv * T2)

    # Sinv sub-blocks for global quadratic forms
    Sinv_bb  <- Sinv[1:p, 1:p, drop = FALSE]
    Sinv_bc  <- Sinv[1:p, (p + 1):(2 * p), drop = FALSE]
    Sinv_cc  <- Sinv[(p + 1):(2 * p), (p + 1):(2 * p), drop = FALSE]
    Sinv_sum <- Sinv_bb + Sinv_bc + t(Sinv_bc) + Sinv_cc

    # Cross-term matrices per school (J x p)
    q_nc    <- -T1Sinv[, 1:p, drop = FALSE]
    T12Sinv <- T1Sinv + T2Sinv
    q_cv    <- -(T12Sinv[, 1:p, drop = FALSE] +
                 T12Sinv[, (p + 1):(2 * p), drop = FALSE])

    # h_ii for non-COVID students
    nci    <- setdiff(seq_len(N), ci)
    sid_nc <- sid[nci]
    Z_nc   <- Z_base[nci, , drop = FALSE]
    h      <- numeric(N)
    h[nci] <- w[nci] * (var_a[sid_nc] +
                         2 * rowSums(Z_nc * q_nc[sid_nc, , drop = FALSE]) +
                         rowSums((Z_nc %*% Sinv_bb) * Z_nc))

    # h_ii for COVID students
    h[ci] <- w[ci] * ((var_a + 2 * cov_ad + var_d)[sid_c] +
                        2 * rowSums(Z_c * q_cv[sid_c, , drop = FALSE]) +
                        rowSums((Z_c %*% Sinv_sum) * Z_c))

    # Firth adjustment vector
    adj <- h * (0.5 - mu)

    # Bias = (X'WX)^{-1} X' adj  (solve via Schur complement)
    f_a    <- drop(rowsum(adj, sid, reorder = TRUE))
    f_d    <- rowsum_c(adj[ci])
    g_full <- c(drop(crossprod(Z_base, adj)), drop(crossprod(Z_c, adj[ci])))

    Ainv_fa <- ai11 * f_a + ai12 * f_d
    Ainv_fd <- ai12 * f_a + ai22 * f_d
    CtAif   <- drop(crossprod(U_full, Ainv_fa) + crossprod(V_full, Ainv_fd))
    gamma_bias <- drop(Sinv %*% (g_full - CtAif))
    Cg_a <- drop(U_full %*% gamma_bias)
    Cg_d <- drop(V_full %*% gamma_bias)

    alpha_bias <- ai11 * (f_a - Cg_a) + ai12 * (f_d - Cg_d)
    delta_bias <- ai12 * (f_a - Cg_a) + ai22 * (f_d - Cg_d)

  } else {
    T1 <- ai11 * U_b + ai12 * U_c
    T2 <- ai12 * U_b + ai22 * U_c
    CtAiC <- crossprod(U_b, T1) + crossprod(U_c, T2)
    S <- B11 - CtAiC
    diag(S) <- diag(S) + 1e-10
    Sinv   <- solve(S)
    T1Sinv <- T1 %*% Sinv
    T2Sinv <- T2 %*% Sinv

    var_a  <- ai11 + rowSums(T1Sinv * T1)
    var_d  <- ai22 + rowSums(T2Sinv * T2)
    cov_ad <- ai12 + rowSums(T1Sinv * T2)

    q_nc <- -T1Sinv
    q_cv <- -(T1Sinv + T2Sinv)

    nci    <- setdiff(seq_len(N), ci)
    sid_nc <- sid[nci]
    Z_nc   <- Z_base[nci, , drop = FALSE]
    h      <- numeric(N)
    h[nci] <- w[nci] * (var_a[sid_nc] +
                         2 * rowSums(Z_nc * q_nc[sid_nc, , drop = FALSE]) +
                         rowSums((Z_nc %*% Sinv) * Z_nc))
    h[ci]  <- w[ci] * ((var_a + 2 * cov_ad + var_d)[sid_c] +
                         2 * rowSums(Z_c * q_cv[sid_c, , drop = FALSE]) +
                         rowSums((Z_c %*% Sinv) * Z_c))

    adj <- h * (0.5 - mu)
    f_a <- drop(rowsum(adj, sid, reorder = TRUE))
    f_d <- rowsum_c(adj[ci])
    g_b <- drop(crossprod(Z_base, adj))

    Ainv_fa <- ai11 * f_a + ai12 * f_d
    Ainv_fd <- ai12 * f_a + ai22 * f_d
    CtAif   <- drop(crossprod(U_b, Ainv_fa) + crossprod(U_c, Ainv_fd))
    gamma_bias <- drop(Sinv %*% (g_b - CtAif))
    Cg_a <- drop(U_b %*% gamma_bias)
    Cg_d <- drop(U_c %*% gamma_bias)

    alpha_bias <- ai11 * (f_a - Cg_a) + ai12 * (f_d - Cg_d)
    delta_bias <- ai12 * (f_a - Cg_a) + ai22 * (f_d - Cg_d)
  }

  list(alpha = fit$alpha + alpha_bias,
       delta = fit$delta + delta_bias)
}


# ==============================================================================
# 3. EMPIRICAL BAYES SHRINKAGE (type-specific, relaxed)
# ==============================================================================
# tau2_floor: minimum between-unit variance to prevent full collapse to mean.
#   When var(est) - mean(SE^2) < 0, standard EB sets tau2=0 and shrinks all
#   to the mean. A floor (e.g. 0.02) preserves some heterogeneity.
eb_shrink <- function(est, se, verbose = FALSE, tau2_floor = 0.02) {
  mu   <- mean(est)
  V    <- se^2
  tau2 <- max(tau2_floor, var(est) - mean(V))
  if (verbose)
    cat(sprintf("      EB: var(est)=%.4f, mean(SE^2)=%.4f, tau2=%.4f\n",
                var(est), mean(V), tau2))
  B  <- tau2 / (tau2 + V)
  (1 - B) * mu + B * est
}


# ==============================================================================
# 4. ESTIMATOR A: Global Model (common slopes — misspecified)
# ==============================================================================
# Type-specific EB shrinkage so Independent/Academy aren't shrunk toward grand mean.
fit_global <- function(sid, d_covid, Z_base, y, cov_names, school_dt) {
  cat("    [A] Global model (common slopes)...\n")
  t0 <- proc.time()

  J <- nrow(school_dt)
  fit <- sparse_block_irls(sid, d_covid, Z_base, y, use_covid_cov = TRUE)
  se  <- compute_block_se(sid, d_covid, Z_base, fit,
                          use_covid_cov = TRUE, ridge = 0)

  eb_alpha <- numeric(J)
  eb_delta <- numeric(J)
  for (stype in c("State", "Academy", "Independent")) {
    idx <- which(school_dt$school_type == stype)
    eb_alpha[idx] <- eb_shrink(fit$alpha[idx], se$se_alpha[idx], tau2_floor = 0.02)
    eb_delta[idx] <- eb_shrink(fit$delta[idx], se$se_delta[idx], tau2_floor = 0.02)
  }

  cat(sprintf("    [A] Done [%.1fs]\n", (proc.time() - t0)["elapsed"]))
  list(fit = fit, eb_alpha = eb_alpha, eb_delta = eb_delta, se = se)
}


# ==============================================================================
# 5. ESTIMATOR B: Split by Type (separate models per school type)
# ==============================================================================
fit_split <- function(sid, d_covid, Z_base, y, school_dt, cov_names) {
  cat("    [B] Split-by-type models...\n")
  t0 <- proc.time()

  J <- nrow(school_dt)
  student_type <- school_dt$school_type[sid]

  eb_alpha_all    <- numeric(J)
  eb_delta_all    <- numeric(J)
  raw_alpha_all   <- numeric(J)
  raw_delta_all   <- numeric(J)
  se_delta_all    <- numeric(J)
  firth_alpha_all <- numeric(J)
  firth_delta_all <- numeric(J)
  fit_list        <- list()

  for (stype in c("State", "Academy", "Independent")) {
    cat(sprintf("      %s: ", stype))

    # Subset
    s_mask     <- which(student_type == stype)
    s_orig_ids <- sort(unique(sid[s_mask]))
    J_s        <- length(s_orig_ids)
    id_map     <- match(sid[s_mask], s_orig_ids)

    s_sid  <- id_map
    s_dcov <- d_covid[s_mask]
    s_y    <- y[s_mask]
    s_Z    <- Z_base[s_mask, , drop = FALSE]

    # Fit
    fit_s <- sparse_block_irls(s_sid, s_dcov, s_Z, s_y, use_covid_cov = TRUE)
    se_s  <- compute_block_se(s_sid, s_dcov, s_Z, fit_s,
                              use_covid_cov = TRUE, ridge = 0)
    eb_a <- eb_shrink(fit_s$alpha, se_s$se_alpha, tau2_floor = 0.02)
    eb_d <- eb_shrink(fit_s$delta, se_s$se_delta, tau2_floor = 0.02)

    # Firth bias correction
    fc <- firth_correct(s_sid, s_dcov, s_Z, fit_s, use_covid_cov = TRUE)

    # Map back to global school IDs
    eb_alpha_all[s_orig_ids]    <- eb_a
    eb_delta_all[s_orig_ids]    <- eb_d
    raw_alpha_all[s_orig_ids]   <- fit_s$alpha
    raw_delta_all[s_orig_ids]   <- fit_s$delta
    se_delta_all[s_orig_ids]    <- se_s$se_delta
    firth_alpha_all[s_orig_ids] <- fc$alpha
    firth_delta_all[s_orig_ids] <- fc$delta

    fit_list[[stype]] <- list(fit = fit_s, se = se_s,
                               eb_alpha = eb_a, eb_delta = eb_d,
                               school_ids = s_orig_ids)
  }

  cat(sprintf("    [B] Done [%.1fs]\n", (proc.time() - t0)["elapsed"]))
  list(eb_alpha = eb_alpha_all, eb_delta = eb_delta_all,
       raw_alpha = raw_alpha_all, raw_delta = raw_delta_all,
       se_delta = se_delta_all,
       firth_alpha = firth_alpha_all, firth_delta = firth_delta_all,
       fits = fit_list)
}


# ==============================================================================
# 6. ESTIMATOR C: Proposed Fix (LASSO on Demo × Type interactions)
# ==============================================================================
fit_typed_lasso <- function(sid, d_covid, Z_base, y, school_dt,
                            cov_names, col_means) {
  cat("    [C] LASSO on Demo x Type interactions...\n")
  t0 <- proc.time()

  N <- length(y)
  J <- nrow(school_dt)
  p <- ncol(Z_base)
  student_type <- school_dt$school_type[sid]

  # --- Build school type indicators at student level ---
  is_acad  <- as.numeric(student_type == "Academy")
  is_indep <- as.numeric(student_type == "Independent")

  # --- Construct Demo × Type interaction columns ---
  # SES=col2, Minority=col3, Gender=col4 in Z_base (centered)
  demo_acad  <- Z_base[, 2:4, drop = FALSE] * is_acad
  demo_indep <- Z_base[, 2:4, drop = FALSE] * is_indep
  colnames(demo_acad)  <- paste0(c("SES", "Minority", "Gender"), "_Acad")
  colnames(demo_indep) <- paste0(c("SES", "Minority", "Gender"), "_Indep")

  # --- Build full design matrix for glmnet ---
  # School FEs
  school_sp <- sparseMatrix(i = seq_len(N), j = sid, x = 1, dims = c(N, J))
  covid_idx <- which(d_covid == 1L)
  covid_school_sp <- sparseMatrix(i = covid_idx, j = sid[covid_idx],
                                  x = 1, dims = c(N, J))

  # Core covariates: GCSE(1) + Demo(3) = cols 1:4
  Z_core <- Z_base[, 1:4, drop = FALSE]
  Z_core_covid <- d_covid * Z_core

  # Type interactions (the LASSO targets)
  Z_type_int <- cbind(demo_acad, demo_indep)

  # Remaining covariates: cols 5:p (polynomials + noise)
  Z_rest <- Z_base[, 5:p, drop = FALSE]
  Z_rest_covid <- d_covid * Z_rest

  # Combine
  X <- cbind(
    school_sp,           # J cols (unpenalized)
    covid_school_sp,     # J cols (unpenalized)
    as(Z_core, "dgCMatrix"),        # 4 cols (unpenalized): GCSE, SES, MIN, GEN
    as(Z_core_covid, "dgCMatrix"),  # 4 cols (unpenalized): COVID × core
    as(Z_type_int, "dgCMatrix"),    # 6 cols (PENALIZED): Demo × Acad/Indep
    as(Z_rest, "dgCMatrix"),        # (p-4) cols (penalized): poly + noise
    as(Z_rest_covid, "dgCMatrix")   # (p-4) cols (penalized): COVID × poly/noise
  )

  rm(school_sp, covid_school_sp, Z_core, Z_core_covid, Z_type_int,
     Z_rest, Z_rest_covid, demo_acad, demo_indep)
  gc(verbose = FALSE)

  # --- Penalty factor ---
  n_core   <- 4L   # GCSE, SES, MIN, GEN
  n_type   <- 6L   # Demo × {Acad, Indep}
  n_rest   <- p - n_core
  pf <- c(
    rep(0, 2L * J),     # school FEs + COVID × school FEs
    rep(0, n_core),      # core covariates
    rep(0, n_core),      # COVID × core covariates
    rep(0, n_type),      # Demo × Type interactions (unpenalized — structural)
    rep(1, n_rest),      # poly + noise base
    rep(1, n_rest)       # COVID × poly/noise
  )

  cat(sprintf("      X: %s x %d, penalized: %d/%d cov cols\n",
              format(N, big.mark = ","), ncol(X),
              sum(pf > 0), length(pf) - 2L * J))

  # --- LASSO fit (AIC selection — less conservative than BIC for large N) ---
  cat("      Fitting glmnet (AIC selection)...\n")
  glm_fit <- glmnet(
    x = X, y = y, family = "binomial",
    alpha = 1, penalty.factor = pf, intercept = FALSE,
    standardize = FALSE, thresh = 1e-7, maxit = 1e5
  )

  dev_v <- deviance(glm_fit)
  df_v  <- glm_fit$df
  aic_v <- dev_v + 2 * df_v
  best_idx <- which.min(aic_v)

  cf    <- coef(glm_fit, s = glm_fit$lambda[best_idx])
  coefs <- as.numeric(cf)[-1]

  # --- Extract coefficients ---
  idx_start <- 2L * J + 1L
  gamma_core      <- coefs[idx_start:(idx_start + n_core - 1)]
  gamma_core_cov  <- coefs[(idx_start + n_core):(idx_start + 2*n_core - 1)]
  gamma_type      <- coefs[(idx_start + 2*n_core):(idx_start + 2*n_core + n_type - 1)]
  gamma_rest      <- coefs[(idx_start + 2*n_core + n_type):(idx_start + 2*n_core + n_type + n_rest - 1)]
  gamma_rest_cov  <- coefs[(idx_start + 2*n_core + n_type + n_rest):length(coefs)]

  # Report which type interactions were selected
  type_names <- c("SES_Acad", "MIN_Acad", "GEN_Acad",
                  "SES_Indep", "MIN_Indep", "GEN_Indep")
  selected_type <- which(gamma_type != 0)
  cat(sprintf("      LASSO kept %d/6 type interactions: %s\n",
              length(selected_type),
              paste(type_names[selected_type], collapse = ", ")))

  # --- Post-LASSO: refit using exact block IRLS solver ---
  all_cov_coefs <- coefs[(2L * J + 1):length(coefs)]
  n_total_cov   <- length(all_cov_coefs)
  selected_cov  <- which(all_cov_coefs != 0)

  # Determine which rest columns (5:p in Z_base) LASSO selected
  rest_base_start  <- 2L * n_core + n_type + 1L
  rest_base_end    <- 2L * n_core + n_type + n_rest
  rest_covid_start <- rest_base_end + 1L
  rest_covid_end   <- n_total_cov

  sel_rest_base  <- intersect(selected_cov, rest_base_start:rest_base_end) - rest_base_start + 1L
  sel_rest_covid <- intersect(selected_cov, rest_covid_start:rest_covid_end) - rest_covid_start + 1L
  sel_rest_local <- sort(unique(c(sel_rest_base, sel_rest_covid)))

  cat(sprintf("      LASSO selected %d / %d rest columns\n",
              length(sel_rest_local), n_rest))

  # Build augmented Z for block IRLS: [core(4) | type_int(6) | selected_rest]
  type_int_mat <- cbind(
    Z_base[, 2] * is_acad,  Z_base[, 3] * is_acad,  Z_base[, 4] * is_acad,
    Z_base[, 2] * is_indep, Z_base[, 3] * is_indep, Z_base[, 4] * is_indep
  )

  if (length(sel_rest_local) > 0) {
    sel_rest_zcols <- 4L + sel_rest_local
    Z_post <- cbind(Z_base[, 1:4], type_int_mat, Z_base[, sel_rest_zcols, drop = FALSE])
  } else {
    Z_post <- cbind(Z_base[, 1:4], type_int_mat)
    sel_rest_zcols <- integer(0)
  }
  p_post <- ncol(Z_post)

  rm(X, type_int_mat)
  gc(verbose = FALSE)

  cat(sprintf("      Post-LASSO refit via block IRLS: %d covariate columns\n", p_post))
  fit_post <- sparse_block_irls(sid, d_covid, Z_post, y, use_covid_cov = TRUE)

  # SE and EB
  se_post <- compute_block_se(sid, d_covid, Z_post, fit_post,
                              use_covid_cov = TRUE, ridge = 0)
  # Type-specific EB shrinkage with tau2 floor to avoid full collapse
  eb_alpha <- numeric(J)
  eb_delta <- numeric(J)
  for (stype in c("State", "Academy", "Independent")) {
    idx <- which(school_dt$school_type == stype)
    eb_alpha[idx] <- eb_shrink(fit_post$alpha[idx], se_post$se_alpha[idx], tau2_floor = 0.02)
    eb_delta[idx] <- eb_shrink(fit_post$delta[idx], se_post$se_delta[idx], tau2_floor = 0.02)
  }

  # Extract type interaction coefficients from post-LASSO
  gamma_type_post <- fit_post$gamma_b[5:10]

  cat(sprintf("    [C] Done [%.1fs]\n", (proc.time() - t0)["elapsed"]))

  list(
    eb_alpha       = eb_alpha,
    eb_delta       = eb_delta,
    fit_post       = fit_post,
    gamma_type     = gamma_type_post,
    type_names     = type_names,
    sel_rest_zcols = sel_rest_zcols,
    p_post         = p_post,
    se             = se_post
  )
}


# ==============================================================================
# 7. COMPUTE RRR FOR ALL ESTIMATORS
# ==============================================================================
# split_only: if TRUE, only compute B (Split-by-Type); res_A and res_C can be NULL
compute_rrr <- function(school_dt, params, res_A, res_B, res_C,
                        sid, d_covid, Z_base, col_means,
                        split_only = FALSE) {
  J <- nrow(school_dt)
  p <- ncol(Z_base)
  stypes <- as.character(school_dt$school_type)
  student_type <- school_dt$school_type[sid]

  # --- Global reference student (fixed across all school types) ---
  # Uses the overall sample mean to remove composition differences.
  # Type-specific slopes (truth + LASSO estimates) still vary by school type.
  ref_raw      <- col_means                        # global mean on raw scale
  ref_centered <- rep(0, p)                        # zero on centered scale

  # --- Clamping: reduce GCSE if max p_base > 0.85 ---
  # Comparable outcome: no demo effect on baseline, so only GCSE matters
  P_BASE_CAP <- 0.85
  alpha_max_all <- max(school_dt$alpha_j)
  max_p <- plogis(alpha_max_all + params$eta_GCSE * ref_raw[1])

  if (max_p > P_BASE_CAP) {
    clamp_fn <- function(z) {
      plogis(alpha_max_all + params$eta_GCSE * z) - P_BASE_CAP
    }
    gcse_clamp <- uniroot(clamp_fn, interval = c(-5, ref_raw[1]), tol = 1e-8)$root
    ref_raw[1] <- gcse_clamp
    ref_raw[5] <- gcse_clamp^2
    ref_raw[6] <- gcse_clamp * ref_raw[2]
    ref_raw[7] <- gcse_clamp * ref_raw[3]
    ref_raw[8] <- gcse_clamp * ref_raw[4]
    ref_centered <- ref_raw - col_means
  }

  # --- Truth: comparable outcome — GCSE only in baseline; COVID adds grading bias ---
  true_base  <- numeric(J)
  true_covid <- numeric(J)
  r <- ref_raw
  for (stype in c("State", "Academy", "Independent")) {
    idx <- which(stypes == stype)

    base_s <- params$eta_GCSE * r[1]

    covid_s <- (params$eta_GCSE + params$delta_eta["GCSE_std"]) * r[1] +
               school_dt$beta_ses_j[idx] * r[2] +
               school_dt$beta_min_j[idx] * r[3]

    true_base[idx]  <- base_s
    true_covid[idx] <- covid_s
  }

  rc <- ref_centered

  # --- Estimator B (Split): type-specific gamma, fixed student ---
  B_base  <- numeric(J)
  B_covid <- numeric(J)
  for (stype in c("State", "Academy", "Independent")) {
    idx  <- which(stypes == stype)
    f    <- res_B$fits[[stype]]$fit
    B_base[idx]  <- drop(rc %*% f$gamma_b)
    B_covid[idx] <- drop(rc %*% (f$gamma_b + f$gamma_c))
  }

  # --- Build results table (B always; A and C only when not split_only) ---
  results <- data.table(
    school_id   = school_dt$school_id,
    school_type = school_dt$school_type,
    true_alpha  = school_dt$alpha_j,
    true_delta  = school_dt$delta_alpha_j,

    # Truth
    true_p0  = plogis(school_dt$alpha_j + true_base),
    true_p1  = plogis(school_dt$alpha_j + school_dt$delta_alpha_j + true_covid),

    # Estimator B (Split) — EB-shrunk
    B_alpha = res_B$eb_alpha,
    B_delta = res_B$eb_delta,
    B_p0    = plogis(res_B$eb_alpha + B_base),
    B_p1    = plogis(res_B$eb_alpha + res_B$eb_delta + B_covid),

    # Estimator B (Split) — raw (unshrunk) and standard errors
    B_raw_alpha = res_B$raw_alpha,
    B_raw_delta = res_B$raw_delta,
    B_se_delta  = res_B$se_delta,

    # Estimator B (Split) — Firth bias-corrected (same gamma, corrected alpha/delta)
    B_firth_alpha = res_B$firth_alpha,
    B_firth_delta = res_B$firth_delta,
    B_firth_p0    = plogis(res_B$firth_alpha + B_base),
    B_firth_p1    = plogis(res_B$firth_alpha + res_B$firth_delta + B_covid)
  )

  if (!split_only) {
    # --- Estimator A (Global): common gamma, fixed student ---
    A_base  <- rep(drop(rc %*% res_A$fit$gamma_b), J)
    A_covid <- rep(drop(rc %*% (res_A$fit$gamma_b + res_A$fit$gamma_c)), J)

    # --- Estimator C (Fix): fixed student, type interactions from LASSO ---
    C_base  <- numeric(J)
    C_covid <- numeric(J)
    p_post         <- res_C$p_post
    sel_rest_zcols <- res_C$sel_rest_zcols
    fit_post       <- res_C$fit_post
    for (stype in c("State", "Academy", "Independent")) {
      idx <- which(stypes == stype)

      rc_post <- numeric(p_post)
      rc_post[1:4] <- rc[1:4]
      if (stype == "Academy") {
        rc_post[5:7] <- rc[2:4]
      } else if (stype == "Independent") {
        rc_post[8:10] <- rc[2:4]
      }
      if (length(sel_rest_zcols) > 0) {
        rc_post[11:p_post] <- rc[sel_rest_zcols]
      }

      C_base[idx]  <- drop(rc_post %*% fit_post$gamma_b)
      C_covid[idx] <- drop(rc_post %*% (fit_post$gamma_b + fit_post$gamma_c))
    }

    results[, `:=`(
      A_alpha = res_A$eb_alpha,
      A_delta = res_A$eb_delta,
      A_p0    = plogis(res_A$eb_alpha + A_base),
      A_p1    = plogis(res_A$eb_alpha + res_A$eb_delta + A_covid),
      C_alpha = res_C$eb_alpha,
      C_delta = res_C$eb_delta,
      C_p0    = plogis(res_C$eb_alpha + C_base),
      C_p1    = plogis(res_C$eb_alpha + res_C$eb_delta + C_covid)
    )]
  }

  # Derived quantities
  prefixes <- if (split_only) c("true", "B", "B_firth") else c("true", "A", "B", "C", "B_firth")
  for (prefix in prefixes) {
    p0_col  <- paste0(prefix, "_p0")
    p1_col  <- paste0(prefix, "_p1")
    raw_col <- paste0(prefix, "_raw")
    rrr_col <- paste0(prefix, "_rrr")
    results[, (raw_col) := get(p1_col) - get(p0_col)]
    results[, (rrr_col) := get(raw_col) / (1 - get(p0_col))]
  }

  # Binning
  results[, alpha_pctl := frank(true_alpha) / .N]
  results[, bin := cut(alpha_pctl, breaks = seq(0, 1, 0.05),
                       labels = FALSE, include.lowest = TRUE)]
  results[, bin_center := (bin - 0.5) / 20]

  results[]
}


# ==============================================================================
# 8. COMPUTE HYPERPARAMETERS AND SCHOOL-LEVEL AGGREGATE ERROR
# ==============================================================================
# split_only: if TRUE, only compute for B_Split
# Reports both EB-shrunk and raw (unshrunk) delta hyperparameters.
# School-level metrics: MAE (mean |est - true|), RMSE (sqrt(mean (est-true)^2))
compute_hyperparams <- function(results, split_only = FALSE) {
  hp_list <- list()
  true_delta <- results$true_delta
  true_rrr   <- results$true_rrr

  # Helper: compute RRR stats for a given estimator's RRR column
  rrr_stats <- function(idx, rrr_col) {
    if (!rrr_col %in% names(results)) return(list(mu = NA_real_, mae = NA_real_, rmse = NA_real_))
    est_rrr  <- results[[rrr_col]][idx]
    tru_rrr  <- true_rrr[idx]
    rrr_err  <- est_rrr - tru_rrr
    list(mu = mean(est_rrr), mae = mean(abs(rrr_err)), rmse = sqrt(mean(rrr_err^2)))
  }

  # EB-shrunk estimators
  ests <- if (split_only) "B" else c("A", "B", "C")
  est_labels <- c(A = "A_Global", B = "B_Split", C = "C_Fix")

  for (est in ests) {
    est_label <- est_labels[[est]]
    delta_col <- paste0(est, "_delta")
    rrr_col   <- paste0(est, "_rrr")
    if (!delta_col %in% names(results)) next

    est_vals <- results[[delta_col]]

    for (grp in c("Overall", "State", "Academy", "Independent")) {
      idx <- if (grp == "Overall") seq_len(nrow(results))
             else which(results$school_type == grp)
      vals   <- est_vals[idx]
      tvals  <- true_delta[idx]
      err    <- vals - tvals
      rs     <- rrr_stats(idx, rrr_col)

      hp_list[[length(hp_list) + 1L]] <- data.table(
        group      = grp,
        estimator  = est_label,
        mu_delta   = mean(vals),
        tau2_delta = var(vals),
        mae_delta  = mean(abs(err)),
        rmse_delta = sqrt(mean(err^2)),
        mu_rrr     = rs$mu,
        rrr_mae    = rs$mae,
        rrr_rmse   = rs$rmse
      )
    }
  }

  # Raw (unshrunk) delta — B_Split only
  raw_col <- "B_raw_delta"
  if (raw_col %in% names(results)) {
    raw_vals <- results[[raw_col]]
    for (grp in c("Overall", "State", "Academy", "Independent")) {
      idx <- if (grp == "Overall") seq_len(nrow(results))
             else which(results$school_type == grp)
      vals   <- raw_vals[idx]
      tvals  <- true_delta[idx]
      err    <- vals - tvals

      hp_list[[length(hp_list) + 1L]] <- data.table(
        group      = grp,
        estimator  = "B_Raw",
        mu_delta   = mean(vals),
        tau2_delta = var(vals),
        mae_delta  = mean(abs(err)),
        rmse_delta = sqrt(mean(err^2)),
        mu_rrr     = NA_real_,
        rrr_mae    = NA_real_,
        rrr_rmse   = NA_real_
      )
    }
  }

  # DerSimonian-Laird estimator — uses raw deltas and their SEs
  se_col <- "B_se_delta"
  if (raw_col %in% names(results) && se_col %in% names(results)) {
    raw_vals <- results[[raw_col]]
    se_vals  <- results[[se_col]]
    has_rrr  <- "B_rrr" %in% names(results)
    for (grp in c("Overall", "State", "Academy", "Independent")) {
      idx <- if (grp == "Overall") seq_len(nrow(results))
             else which(results$school_type == grp)
      d_j   <- raw_vals[idx]
      se_j  <- se_vals[idx]
      tvals <- true_delta[idx]
      k     <- length(d_j)

      # Step 1: tau2 via DL (Cochran's Q)
      w_fe  <- 1 / se_j^2
      mu_fe <- sum(w_fe * d_j) / sum(w_fe)
      Q     <- sum(w_fe * (d_j - mu_fe)^2)
      c_dl  <- sum(w_fe) - sum(w_fe^2) / sum(w_fe)
      tau2  <- max(0, (Q - (k - 1)) / c_dl)

      # Step 2: mu via precision-weighted mean
      w_re  <- 1 / (se_j^2 + tau2)
      mu_dl <- sum(w_re * d_j) / sum(w_re)

      err  <- d_j - tvals

      # Precision-weighted mean RRR using the same RE weights
      mu_rrr_dl  <- NA_real_
      rrr_mae_dl <- NA_real_
      rrr_rmse_dl <- NA_real_
      if (has_rrr) {
        est_rrr <- results[["B_rrr"]][idx]
        tru_rrr <- true_rrr[idx]
        mu_rrr_dl  <- sum(w_re * est_rrr) / sum(w_re)
        rrr_err    <- est_rrr - tru_rrr
        rrr_mae_dl <- mean(abs(rrr_err))
        rrr_rmse_dl <- sqrt(mean(rrr_err^2))
      }

      hp_list[[length(hp_list) + 1L]] <- data.table(
        group      = grp,
        estimator  = "B_DL",
        mu_delta   = mu_dl,
        tau2_delta = tau2,
        mae_delta  = mean(abs(err)),
        rmse_delta = sqrt(mean(err^2)),
        mu_rrr     = mu_rrr_dl,
        rrr_mae    = rrr_mae_dl,
        rrr_rmse   = rrr_rmse_dl
      )
    }
  }

  # Firth bias-corrected delta — simple sample moments + RRR
  firth_col <- "B_firth_delta"
  if (firth_col %in% names(results)) {
    firth_vals <- results[[firth_col]]
    for (grp in c("Overall", "State", "Academy", "Independent")) {
      idx <- if (grp == "Overall") seq_len(nrow(results))
             else which(results$school_type == grp)
      vals  <- firth_vals[idx]
      tvals <- true_delta[idx]
      err   <- vals - tvals
      rs    <- rrr_stats(idx, "B_firth_rrr")

      hp_list[[length(hp_list) + 1L]] <- data.table(
        group      = grp,
        estimator  = "B_Firth",
        mu_delta   = mean(vals),
        tau2_delta = var(vals),
        mae_delta  = mean(abs(err)),
        rmse_delta = sqrt(mean(err^2)),
        mu_rrr     = rs$mu,
        rrr_mae    = rs$mae,
        rrr_rmse   = rs$rmse
      )
    }
  }

  # Firth + DerSimonian-Laird: DL on Firth-corrected deltas
  if (firth_col %in% names(results) && se_col %in% names(results)) {
    firth_vals <- results[[firth_col]]
    se_vals    <- results[[se_col]]
    has_frrr   <- "B_firth_rrr" %in% names(results)
    for (grp in c("Overall", "State", "Academy", "Independent")) {
      idx <- if (grp == "Overall") seq_len(nrow(results))
             else which(results$school_type == grp)
      d_j   <- firth_vals[idx]
      se_j  <- se_vals[idx]
      tvals <- true_delta[idx]
      k     <- length(d_j)

      w_fe  <- 1 / se_j^2
      mu_fe <- sum(w_fe * d_j) / sum(w_fe)
      Q     <- sum(w_fe * (d_j - mu_fe)^2)
      c_dl  <- sum(w_fe) - sum(w_fe^2) / sum(w_fe)
      tau2  <- max(0, (Q - (k - 1)) / c_dl)
      w_re  <- 1 / (se_j^2 + tau2)
      mu_dl <- sum(w_re * d_j) / sum(w_re)
      err   <- d_j - tvals

      mu_rrr_fdl  <- NA_real_
      rrr_mae_fdl <- NA_real_
      rrr_rmse_fdl <- NA_real_
      if (has_frrr) {
        est_rrr <- results[["B_firth_rrr"]][idx]
        tru_rrr <- true_rrr[idx]
        mu_rrr_fdl  <- sum(w_re * est_rrr) / sum(w_re)
        rrr_err     <- est_rrr - tru_rrr
        rrr_mae_fdl <- mean(abs(rrr_err))
        rrr_rmse_fdl <- sqrt(mean(rrr_err^2))
      }

      hp_list[[length(hp_list) + 1L]] <- data.table(
        group      = grp,
        estimator  = "B_Firth_DL",
        mu_delta   = mu_dl,
        tau2_delta = tau2,
        mae_delta  = mean(abs(err)),
        rmse_delta = sqrt(mean(err^2)),
        mu_rrr     = mu_rrr_fdl,
        rrr_mae    = rrr_mae_fdl,
        rrr_rmse   = rrr_rmse_fdl
      )
    }
  }

  rbindlist(hp_list)
}


# ==============================================================================
# 9. ORCHESTRATOR
# ==============================================================================
# split_only: if TRUE, only run Split-by-Type (B); avoids extrapolation bias
estimate_all <- function(panel, school_dt, params, split_only = FALSE) {
  J         <- nrow(school_dt)
  cov_names <- get_covariate_names(params)
  p         <- length(cov_names)
  sid       <- panel$school_id
  d_covid   <- panel$D_covid
  y         <- panel$Y

  # Mean-centre covariates
  Z_base <- as.matrix(panel[, ..cov_names])
  Z_base <- scale(Z_base, center = TRUE, scale = FALSE)
  col_means <- attr(Z_base, "scaled:center")

  rm(panel)
  gc(verbose = FALSE)

  cat(sprintf("    J = %d schools, p = %d covariates, N = %s\n",
              J, p, format(length(y), big.mark = ",")))

  # --- Estimator B: Split by Type (always run) ---
  res_B <- fit_split(sid, d_covid, Z_base, y, school_dt, cov_names)

  if (split_only) {
    # --- Compute RRR for B only ---
    results <- compute_rrr(school_dt, params, NULL, res_B, NULL,
                          sid, d_covid, Z_base, col_means, split_only = TRUE)
    hyperparams <- compute_hyperparams(results, split_only = TRUE)
  } else {
    # --- Estimator A: Global ---
    res_A <- fit_global(sid, d_covid, Z_base, y, cov_names, school_dt)

    # --- Estimator C: Proposed Fix ---
    res_C <- fit_typed_lasso(sid, d_covid, Z_base, y, school_dt,
                             cov_names, col_means)

    # --- Compute RRR for all ---
    results <- compute_rrr(school_dt, params, res_A, res_B, res_C,
                          sid, d_covid, Z_base, col_means, split_only = FALSE)
    hyperparams <- compute_hyperparams(results, split_only = FALSE)
  }

  rm(Z_base, sid, d_covid, y)
  gc(verbose = FALSE)

  list(results = results, hyperparams = hyperparams)
}
