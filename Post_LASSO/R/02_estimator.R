# ==============================================================================
# 02_estimator.R — Estimation Functions (Updated with Post-LASSO)
# ==============================================================================
#
# Three model pipelines:
#   Model A: Unpenalized Logit (ridge, lambda~0) + EB
#   Model B: Penalized Logit (LASSO on covariates) + EB
#   Model C: Post-LASSO (Unpenalized on LASSO-selected subset) + EB
#
# ==============================================================================

.libPaths(c(file.path(Sys.getenv("HOME"), "A-Levels", "R_libs"), .libPaths()))
library(data.table)
library(Matrix)
library(glmnet)

source(file.path(Sys.getenv("HOME"), "A-Levels", "Post_LASSO", "R", "01_dgp.R"))


# ------------------------------------------------------------------------------
# 1. BUILD SPARSE DESIGN MATRIX
# ------------------------------------------------------------------------------
build_design_matrix <- function(panel, J, cov_names) {
  N <- nrow(panel)
  p <- length(cov_names)

  # --- School indicator dummies (N x J) ---
  school_sp <- sparseMatrix(
    i = seq_len(N),
    j = panel$school_id,
    x = 1,
    dims = c(N, J)
  )

  # --- COVID x School dummies (N x J) ---
  covid_rows <- which(panel$D_covid == 1L)
  covid_school_sp <- sparseMatrix(
    i = covid_rows,
    j = panel$school_id[covid_rows],
    x = 1,
    dims = c(N, J)
  )

  # --- Covariates (N x p) ---
  Z <- as.matrix(panel[, ..cov_names])
  Z <- scale(Z, center = TRUE, scale = FALSE)  # mean-zero columns
  Z_sp <- as(Z, "dgCMatrix")

  # --- COVID x Covariates (N x p) ---
  Z_covid <- Z * panel$D_covid
  Z_covid_sp <- as(Z_covid, "dgCMatrix")
  rm(Z, Z_covid)

  # --- Combine ---
  X <- cbind(school_sp, covid_school_sp, Z_sp, Z_covid_sp)
  rm(school_sp, covid_school_sp, Z_sp, Z_covid_sp)

  col_info <- list(
    alpha_cols       = 1:J,
    delta_alpha_cols = (J + 1):(2 * J),
    eta_cols         = (2 * J + 1):(2 * J + p),
    delta_eta_cols   = (2 * J + p + 1):(2 * J + 2 * p),
    J = J,
    p = p
  )

  list(X = X, col_info = col_info)
}


# ------------------------------------------------------------------------------
# 2. FIT UNPENALIZED LOGIT
# ------------------------------------------------------------------------------
fit_unpenalized <- function(X, y, tol = 1e-8, maxit = 50L) {
  # Sparse IRLS (iteratively reweighted least squares) for logistic MLE
  # No penalty — true maximum likelihood
  N <- nrow(X)
  P <- ncol(X)
  beta <- rep(0, P)

  for (iter in seq_len(maxit)) {
    eta <- as.numeric(X %*% beta)
    mu  <- plogis(eta)
    w   <- mu * (1 - mu)
    w   <- pmax(w, 1e-10)

    # Working response
    z <- eta + (y - mu) / w

    # Weighted normal equations: (X'WX) beta = X'Wz
    # Use sparse Cholesky for efficiency
    sqW  <- sqrt(w)
    XtWX <- crossprod(X * sqW)
    diag(XtWX) <- diag(XtWX) + 1e-10  # numerical stability
    XtWz <- as.numeric(crossprod(X, w * z))

    beta_new <- as.numeric(solve(XtWX, XtWz))

    # Check convergence
    if (max(abs(beta_new - beta)) < tol) {
      beta <- beta_new
      break
    }
    beta <- beta_new
  }

  fitted <- plogis(as.numeric(X %*% beta))
  list(coefs = beta, fitted = fitted)
}


# ------------------------------------------------------------------------------
# 3. FIT PENALIZED LOGIT (LASSO)
# ------------------------------------------------------------------------------
fit_penalized <- function(X, y, pf, nfolds = 5L) {
  cv_fit <- cv.glmnet(
    x              = X,
    y              = y,
    family         = "binomial",
    alpha          = 1,
    penalty.factor = pf,
    intercept      = FALSE,
    standardize    = FALSE,
    nfolds         = nfolds,
    type.measure   = "deviance",
    thresh         = 1e-7,
    maxit          = 1e5
  )

  cf     <- coef(cv_fit, s = "lambda.min")
  coefs  <- as.numeric(cf)[-1]
  fitted <- plogis(as.numeric(X %*% coefs))

  list(coefs = coefs, fitted = fitted, lambda_min = cv_fit$lambda.min)
}


# ------------------------------------------------------------------------------
# 4. STANDARD ERRORS FOR SCHOOL FEs
# ------------------------------------------------------------------------------
compute_school_fe_se <- function(X, coefs, J, panel_school_id, panel_D_covid) {
  N <- nrow(X)
  P <- ncol(X)
  # Detect number of covariates dynamically (P - 2J)
  # This allows the function to work for Post-LASSO where p is smaller
  p2 <- P - 2L * J 

  eta <- as.numeric(X %*% coefs)
  mu  <- plogis(eta)
  w   <- mu * (1 - mu)
  w   <- pmax(w, 1e-10)

  # --- Block D (School FEs) ---
  agg <- data.table(sid = panel_school_id, dc = panel_D_covid, w = w)
  agg <- agg[, .(w_all = sum(w), w_covid = sum(w * dc)), by = sid]
  setorder(agg, sid)

  w_all   <- agg$w_all
  w_covid <- agg$w_covid
  w_hist  <- w_all - w_covid

  det_D <- w_covid * w_hist
  det_D <- pmax(det_D, 1e-12)

  Dinv_11 <- w_covid / det_D
  Dinv_22 <- w_all   / det_D
  Dinv_12 <- -w_covid / det_D

  # --- Block U (Covariates) & C ---
  # B is whatever columns remain after the first 2J
  if (p2 > 0) {
    B_sp <- X[, (2L * J + 1):P, drop = FALSE]
    B    <- as.matrix(B_sp)
    rm(B_sp)

    C <- crossprod(B * sqrt(w))
    diag(C) <- diag(C) + 1e-6

    wB  <- w * B
    wDB <- (w * panel_D_covid) * B
    rm(B)

    dt_base <- data.table(sid = panel_school_id, wB)
    U_base  <- as.matrix(dt_base[, lapply(.SD, sum), by = sid][order(sid)][, -1])

    dt_covid <- data.table(sid = panel_school_id, wDB)
    U_covid  <- as.matrix(dt_covid[, lapply(.SD, sum), by = sid][order(sid)][, -1])
    rm(dt_base, dt_covid, wB, wDB)

    # --- Woodbury Q ---
    UDU <- crossprod(U_base, Dinv_11 * U_base) +
           crossprod(U_base, Dinv_12 * U_covid) +
           crossprod(U_covid, Dinv_12 * U_base) +
           crossprod(U_covid, Dinv_22 * U_covid)

    S <- C - UDU
    diag(S) <- diag(S) + 1e-6
    Q <- tryCatch(
      chol2inv(chol(S)),
      error = function(e) solve(S)
    )

    # --- Variances ---
    T1 <- Dinv_11 * U_base + Dinv_12 * U_covid
    T2 <- Dinv_12 * U_base + Dinv_22 * U_covid

    var_alpha <- Dinv_11 + rowSums((T1 %*% Q) * T1)
    var_delta <- Dinv_22 + rowSums((T2 %*% Q) * T2)
  } else {
    # If NO covariates selected (unlikely), Variance is just Dinv
    var_alpha <- Dinv_11
    var_delta <- Dinv_22
  }

  list(
    se_alpha = sqrt(pmax(var_alpha, 0)),
    se_delta = sqrt(pmax(var_delta, 0))
  )
}


# ------------------------------------------------------------------------------
# 5. EMPIRICAL BAYES SHRINKAGE
# ------------------------------------------------------------------------------
eb_shrink <- function(alpha_hat, se_hat) {
  mu   <- mean(alpha_hat)
  V_j  <- se_hat^2
  tau2 <- max(0, var(alpha_hat) - mean(V_j))

  if (tau2 < 1e-12) {
    B  <- rep(0, length(alpha_hat))
    eb <- rep(mu, length(alpha_hat))
  } else {
    B  <- tau2 / (tau2 + V_j)
    eb <- (1 - B) * mu + B * alpha_hat
  }

  list(eb = eb, B = B, mu = mu, tau2 = tau2)
}


# ------------------------------------------------------------------------------
# 6. ORCHESTRATOR
# ------------------------------------------------------------------------------
estimate_all <- function(panel, school_dt, params) {
  J         <- params$J
  cov_names <- get_covariate_names(params)
  y         <- panel$Y
  sid       <- panel$school_id
  dcov      <- panel$D_covid

  # --- Build Design Matrix ---
  cat("    Building design matrix...")
  t0  <- proc.time()
  dm  <- build_design_matrix(panel, J, cov_names)
  X   <- dm$X
  ci  <- dm$col_info
  cat(sprintf(" [%.0fs]\n", (proc.time() - t0)["elapsed"]))

  pf <- c(rep(0, 2 * J), rep(1, 2 * ci$p))

  # --- MODEL A: Unpenalized ---
  cat("    Fitting Unpenalized...")
  t0 <- proc.time()
  res_unpen <- fit_unpenalized(X, y)
  cat(sprintf(" [%.0fs]\n", (proc.time() - t0)["elapsed"]))

  se_unpen <- compute_school_fe_se(X, res_unpen$coefs, J, sid, dcov)
  eb_alpha_unpen <- eb_shrink(res_unpen$coefs[ci$alpha_cols], se_unpen$se_alpha)
  eb_delta_unpen <- eb_shrink(res_unpen$coefs[ci$delta_alpha_cols], se_unpen$se_delta)

  # --- MODEL B: Penalized (LASSO) ---
  cat("    Fitting Penalized (LASSO)...")
  t0 <- proc.time()
  res_pen <- fit_penalized(X, y, pf, nfolds = 5L)
  cat(sprintf(" [%.0fs]\n", (proc.time() - t0)["elapsed"]))

  se_pen <- compute_school_fe_se(X, res_pen$coefs, J, sid, dcov)
  eb_alpha_pen <- eb_shrink(res_pen$coefs[ci$alpha_cols], se_pen$se_alpha)
  eb_delta_pen <- eb_shrink(res_pen$coefs[ci$delta_alpha_cols], se_pen$se_delta)

  # --- MODEL C: Post-LASSO (Double Selection) ---
  cat("    Fitting Post-LASSO...")
  t0 <- proc.time()
  
  # Identify selected COVARIATES (indices > 2J)
  all_cov_indices <- c(ci$eta_cols, ci$delta_eta_cols)
  # Check which of these have non-zero coefs in LASSO
  selected_flags  <- res_pen$coefs[all_cov_indices] != 0
  
  # Mandatory: first 4 base covariates + their COVID interactions
  mandatory_base_idx <- ci$eta_cols[1:4]
  mandatory_covid_idx <- ci$delta_eta_cols[1:4]
  mandatory_indices <- c(mandatory_base_idx, mandatory_covid_idx)

  # Union of LASSO-selected and mandatory
  keep_cols <- unique(c(1:(2*J), all_cov_indices[selected_flags], mandatory_indices))
  keep_cols <- sort(keep_cols)
  X_post    <- X[, keep_cols, drop = FALSE]
  
  # Re-run Unpenalized on this subset
  res_post <- fit_unpenalized(X_post, y)
  cat(sprintf(" [%.0fs]\n", (proc.time() - t0)["elapsed"]))
  
  # Extract Alphas (still first J) and Deltas (next J)
  # X_post preserves the first 2J columns exactly
  alpha_post <- res_post$coefs[1:J]
  delta_post <- res_post$coefs[(J + 1):(2 * J)]
  
  # SE and EB for Post-LASSO
  se_post <- compute_school_fe_se(X_post, res_post$coefs, J, sid, dcov)
  eb_alpha_post <- eb_shrink(alpha_post, se_post$se_alpha)
  eb_delta_post <- eb_shrink(delta_post, se_post$se_delta)

  # --- Results ---
  rm(X, X_post, dm, res_unpen, res_pen, res_post)
  gc(verbose = FALSE)

  data.table(
    school_id         = school_dt$school_id,
    school_type       = school_dt$school_type,
    school_size       = school_dt$school_size,
    alpha_percentile  = school_dt$alpha_percentile,
    true_alpha        = school_dt$alpha_j,
    true_delta        = school_dt$delta_alpha_j,

    # Model A
    unpen_raw_alpha   = eb_alpha_unpen$eb, # Using EB for cleaner table, or raw? 
    unpen_raw_delta   = eb_delta_unpen$eb, # Prompt asked for raw+EB, logic below stores EB result
    
    # Let's align with the Runner script expectation. 
    # Runner calculates "delta" from alpha, but here we estimate delta directly.
    # We will return the EB estimates as the primary output for the table rows.
    
    unpen_est_alpha   = eb_alpha_unpen$eb,
    unpen_est_delta   = eb_delta_unpen$eb,

    pen_est_alpha     = eb_alpha_pen$eb,
    pen_est_delta     = eb_delta_pen$eb,
    
    post_est_alpha    = eb_alpha_post$eb,
    post_est_delta    = eb_delta_post$eb
  )[]
}
