# ==============================================================================
# run_single.R — Single-sample comparison
#   Model A: Unpenalized MLE  (custom sparse block-IRLS)
#   Model B: Penalized MLE    (LASSO via glmnet)
#   Model C: Post-LASSO       (LASSO selection → unpenalized MLE on subset)
#
# The custom IRLS solver exploits the block-diagonal structure of X'WX:
#   School FE columns are sparse indicators → 2×2 diagonal blocks
#   Covariate columns are low-dimensional (2p ≈ 60)
# Solves via Schur complement: O(p^3 + Jp^2) instead of O((2J+2p)^3)
# Memory: O(Np) instead of O((2J+2p)^2)
# ==============================================================================

.libPaths(c(file.path(Sys.getenv("HOME"), "A-Levels", "R_libs"), .libPaths()))

source(file.path(Sys.getenv("HOME"), "A-Levels", "Post_LASSO", "R", "01_dgp.R"))
library(Matrix)
library(glmnet)
library(ggplot2)
library(data.table)


# ==============================================================================
# 1. SPARSE BLOCK IRLS SOLVER
# ==============================================================================
#
# Model: logit P(Y=1) = alpha_j + delta_j * D_covid + Z %*% gamma
#
# The (2J + q) × (2J + q) weighted normal equations have structure:
#
#   [ A   C ] [ theta ]   [ f ]
#   [ C'  B ] [ gamma ] = [ g ]
#
# where:
#   A = block-diag of J 2×2 blocks (from school FEs)
#   B = Z'WZ (q × q, small dense)
#   C = cross-terms (2J × q, but stored as two J × q matrices)
#   theta = (alpha, delta) stacked
#
# Solution via Schur complement:
#   S = B - C' A^{-1} C          (q × q dense system)
#   gamma = S^{-1} (g - C' A^{-1} f)
#   theta = A^{-1} (f - C gamma)  (J independent 2×2 solves)
#
# Memory-efficient: only materialises Z_base (N × p), computes COVID
# interactions on the fly to avoid storing Z_full (N × 2p).
# ==============================================================================

sparse_block_irls <- function(sid, d_covid, Z_base, y,
                              use_covid_cov = TRUE,
                              ridge = 1e-3,
                              tol = 1e-8, maxit = 200) {
  # sid:      integer school IDs (1..J)
  # d_covid:  binary COVID indicator (0/1)
  # Z_base:   N × p base covariate matrix (mean-centred)
  # y:        binary outcome
  # use_covid_cov: if TRUE, include COVID × covariate interactions
  #
  # Returns: list(alpha, delta, gamma_b, gamma_c, iter, converged)

  N <- length(y)
  J <- max(sid)
  p <- if (is.null(Z_base)) 0L else ncol(Z_base)
  has_cov <- p > 0

  alpha <- rep(0, J)
  delta <- rep(0, J)
  gamma_b <- if (has_cov) rep(0, p) else numeric(0)
  gamma_c <- if (has_cov && use_covid_cov) rep(0, p) else numeric(0)
  q <- length(gamma_b) + length(gamma_c)  # total covariate params

  for (iter in seq_len(maxit)) {

    # --- Linear predictor ---
    eta <- alpha[sid] + delta[sid] * d_covid
    if (has_cov) {
      eta <- eta + drop(Z_base %*% gamma_b)
      if (use_covid_cov) eta <- eta + d_covid * drop(Z_base %*% gamma_c)
    }

    mu <- plogis(eta)
    w  <- mu * (1 - mu)
    w  <- pmax(w, 1e-10)

    # --- Working response ---
    z  <- eta + (y - mu) / w
    wz <- w * z

    # --- School-level sufficient statistics ---
    w_j   <- drop(rowsum(w, sid, reorder = TRUE))
    w_j_c <- drop(rowsum(w * d_covid, sid, reorder = TRUE))
    w_j_h <- w_j - w_j_c

    f_alpha <- drop(rowsum(wz, sid, reorder = TRUE))
    f_delta <- drop(rowsum(wz * d_covid, sid, reorder = TRUE))

    # --- A^{-1} components (J independent 2×2 blocks) ---
    # A_j + ridge*I = [[w_j + r, w_j^c], [w_j^c, w_j^c + r]]
    # Ridge prevents separation-induced divergence for small schools.
    # In penalized IRLS: (X'WX + R) beta_new = X'Wz
    # Ridge only enters the Hessian (LHS), not the RHS.
    w_j_r   <- w_j + ridge
    w_j_c_r <- w_j_c + ridge
    det_j <- w_j_r * w_j_c_r - w_j_c^2
    det_j <- pmax(det_j, 1e-12)
    ai11 <- w_j_c_r / det_j
    ai12 <- -w_j_c / det_j
    ai22 <- w_j_r / det_j

    if (has_cov) {
      # --- Build cross-terms and covariate block ---
      # Memory-efficient: compute wZ, use it, then overwrite for wdZ
      wZ <- w * Z_base                                  # N × p (largest temp)
      U_b <- rowsum(wZ, sid, reorder = TRUE)            # J × p
      B11 <- crossprod(Z_base, wZ)                      # p × p

      wzd <- wz * d_covid
      g_b <- drop(crossprod(Z_base, wz))                # p-vector
      g_c <- if (use_covid_cov) drop(crossprod(Z_base, wzd)) else numeric(0)

      # Overwrite wZ → w*d*Z to save memory
      wZ <- d_covid * wZ                                # now w*d*Z_base
      U_c <- rowsum(wZ, sid, reorder = TRUE)            # J × p
      B12 <- crossprod(Z_base, wZ)                      # p × p
      rm(wZ); # free N × p temp

      # Full block matrices for Schur complement:
      # U_full = [U_b | U_c]  (J × 2p)  — alpha rows of C
      # V_full = [U_c | U_c]  (J × 2p)  — delta rows of C
      # B_full = [[B11, B12], [B12, B12]]  (2p × 2p)
      # (B22 = B12 because d^2 = d for binary d_covid)

      if (use_covid_cov) {
        # Form the full q × q system
        U_full <- cbind(U_b, U_c)    # J × 2p
        V_full <- cbind(U_c, U_c)    # J × 2p
        B_full <- rbind(cbind(B11, B12), cbind(B12, B12))  # 2p × 2p
        g_full <- c(g_b, g_c)

        # Schur complement: S = B - C' A^{-1} C
        CtAiC <- crossprod(U_full, ai11 * U_full) +
                 crossprod(U_full, ai12 * V_full) +
                 crossprod(V_full, ai12 * U_full) +
                 crossprod(V_full, ai22 * V_full)

        S <- B_full - CtAiC
        diag(S) <- diag(S) + 1e-10

        # RHS: g - C' A^{-1} f
        Ainv_fa <- ai11 * f_alpha + ai12 * f_delta
        Ainv_fd <- ai12 * f_alpha + ai22 * f_delta
        CtAif <- drop(crossprod(U_full, Ainv_fa) + crossprod(V_full, Ainv_fd))

        gamma_full_new <- drop(solve(S, g_full - CtAif))
        gamma_b_new <- gamma_full_new[1:p]
        gamma_c_new <- gamma_full_new[(p + 1):(2 * p)]

        # Back-substitute: theta = A^{-1}(f - C gamma)
        Cg_a <- drop(U_full %*% gamma_full_new)
        Cg_d <- drop(V_full %*% gamma_full_new)

      } else {
        # Only base covariates, no COVID interactions
        # U = U_b, V = U_c (delta rows still cross with base covariates)
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
      # No covariates — just solve the 2×2 blocks
      alpha_new <- ai11 * f_alpha + ai12 * f_delta
      delta_new <- ai12 * f_alpha + ai22 * f_delta
      gamma_b_new <- numeric(0)
      gamma_c_new <- numeric(0)
    }

    # --- Convergence ---
    max_change <- max(abs(c(alpha_new - alpha, delta_new - delta,
                            gamma_b_new - gamma_b, gamma_c_new - gamma_c)))
    alpha   <- alpha_new
    delta   <- delta_new
    gamma_b <- gamma_b_new
    gamma_c <- gamma_c_new

    if (max_change < tol) break
  }

  # Final log-likelihood
  eta_final <- alpha[sid] + delta[sid] * d_covid
  if (has_cov) {
    eta_final <- eta_final + drop(Z_base %*% gamma_b)
    if (use_covid_cov) eta_final <- eta_final + d_covid * drop(Z_base %*% gamma_c)
  }
  mu_final <- plogis(eta_final)
  ll <- sum(y * log(mu_final + 1e-15) + (1 - y) * log(1 - mu_final + 1e-15))

  converged <- (max_change < tol)
  cat(sprintf("      %s in %d iters (max change: %.2e, loglik: %.1f)\n",
              if (converged) "Converged" else "WARNING: not converged",
              iter, max_change, ll))

  list(alpha = alpha, delta = delta, gamma_b = gamma_b, gamma_c = gamma_c,
       iter = iter, converged = converged)
}


# ==============================================================================
# 2. STANDARD ERRORS (block Schur complement)
# ==============================================================================
# Var(theta) from the inverse Fisher information, exploiting the same
# block structure as the solver. Only returns SE for school FEs.

compute_block_se <- function(sid, d_covid, Z_base, fit,
                             use_covid_cov = TRUE, ridge = 1e-3) {
  N <- length(sid)
  J <- max(sid)

  eta <- fit$alpha[sid] + fit$delta[sid] * d_covid
  if (length(fit$gamma_b) > 0) {
    eta <- eta + drop(Z_base %*% fit$gamma_b)
    if (use_covid_cov && length(fit$gamma_c) > 0)
      eta <- eta + d_covid * drop(Z_base %*% fit$gamma_c)
  }

  mu <- plogis(eta)
  w  <- mu * (1 - mu)
  w  <- pmax(w, 1e-10)

  w_j   <- drop(rowsum(w, sid, reorder = TRUE))
  w_j_c <- drop(rowsum(w * d_covid, sid, reorder = TRUE))

  # Match the ridge-augmented A block from the solver
  w_j_r   <- w_j + ridge
  w_j_c_r <- w_j_c + ridge
  det_j   <- pmax(w_j_r * w_j_c_r - w_j_c^2, 1e-12)

  ai11 <- w_j_c_r / det_j
  ai12 <- -w_j_c / det_j
  ai22 <- w_j_r / det_j

  p <- ncol(Z_base)
  if (!is.null(Z_base) && p > 0) {
    wZ <- w * Z_base
    U_b <- rowsum(wZ, sid, reorder = TRUE)
    B11 <- crossprod(Z_base, wZ)
    wZ  <- d_covid * wZ
    U_c <- rowsum(wZ, sid, reorder = TRUE)
    B12 <- crossprod(Z_base, wZ)
    rm(wZ)

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

      # T1_j = (A^{-1} C)_{alpha-row, j} — used for Var(alpha_j)
      # T2_j = (A^{-1} C)_{delta-row, j} — used for Var(delta_j)
      T1 <- ai11 * U_full + ai12 * V_full   # J × 2p
      T2 <- ai12 * U_full + ai22 * V_full   # J × 2p

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

    # Var(alpha_j) = ai11_j + T1_j' Sinv T1_j  (Woodbury correction)
    # Var(delta_j) = ai22_j + T2_j' Sinv T2_j
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
# 3. EMPIRICAL BAYES SHRINKAGE
# ==============================================================================
eb_shrink <- function(est, se, verbose = FALSE) {
  mu   <- mean(est)
  V    <- se^2
  var_est <- var(est)
  mean_V  <- mean(V)
  tau2 <- max(0, var_est - mean_V)

  if (verbose) {
    cat(sprintf("      EB: var(est)=%.4f, mean(SE^2)=%.4f, tau2=%.4f\n",
                var_est, mean_V, tau2))
  }

  if (tau2 < 1e-12) return(rep(mu, length(est)))
  B  <- tau2 / (tau2 + V)
  (1 - B) * mu + B * est
}


# ==============================================================================
# 4. MAIN SIMULATION
# ==============================================================================

cat("========================================\n")
cat("  Single-Sample Simulation\n")
cat("  Post-LASSO vs MLE vs Penalized MLE\n")
cat("========================================\n\n")

# --- 4a. Generate data (scaled down for single-sample feasibility) ---
cat("[1] Generating data...\n")
params      <- default_params()
params$J    <- 500L
params$size_range <- list(
  State       = c(80L, 250L),
  Academy     = c(60L, 200L),
  Independent = c(40L, 120L)
)
params$n_noise <- 10L   # fewer noise covariates (still enough to test selection)

school_dt <- generate_school_structure(params)

t0    <- proc.time()
panel <- generate_panel(school_dt, params, seed = 1L)
cat(sprintf("    Panel: %s rows x %d cols [%.1fs]\n",
            format(nrow(panel), big.mark = ","), ncol(panel),
            (proc.time() - t0)["elapsed"]))

# --- 4b. Extract estimation inputs, then free panel ---
J         <- params$J
cov_names <- get_covariate_names(params)
p         <- length(cov_names)

sid     <- panel$school_id
d_covid <- panel$D_covid
y       <- panel$Y

Z_base <- as.matrix(panel[, ..cov_names])
Z_base <- scale(Z_base, center = TRUE, scale = FALSE)  # mean-centre

rm(panel)
gc(verbose = FALSE)

cat(sprintf("    J = %d schools, p = %d covariates\n", J, p))
cat(sprintf("    Z_base: %.0f MB\n\n", object.size(Z_base) / 1e6))


# ==============================================================================
# MODEL A: Unpenalized MLE (sparse block IRLS)
# ==============================================================================
cat("[2] MODEL A: Unpenalized MLE (block IRLS)\n")
t0    <- proc.time()
fit_A <- sparse_block_irls(sid, d_covid, Z_base, y, use_covid_cov = TRUE)
time_A <- (proc.time() - t0)["elapsed"]
cat(sprintf("    Time: %.1fs\n", time_A))

cat("    Computing SEs...\n")
# Use ridge=0 for SE computation so EB reflects true sampling variability
se_A <- compute_block_se(sid, d_covid, Z_base, fit_A, use_covid_cov = TRUE, ridge = 0)
eb_alpha_A <- eb_shrink(fit_A$alpha, se_A$se_alpha, verbose = TRUE)
eb_delta_A <- eb_shrink(fit_A$delta, se_A$se_delta, verbose = TRUE)
cat(sprintf("    median SE(alpha) = %.4f, median SE(delta) = %.4f\n\n",
            median(se_A$se_alpha), median(se_A$se_delta)))


# ==============================================================================
# MODEL B: Penalized MLE (LASSO via glmnet)
# ==============================================================================
cat("[3] MODEL B: Penalized MLE (LASSO)\n")
cat("    Building sparse design matrix for glmnet...\n")
t0 <- proc.time()

N <- length(y)

# School FE indicators (sparse)
school_sp <- sparseMatrix(
  i = seq_len(N), j = sid, x = 1, dims = c(N, J)
)
# COVID × School FE indicators (sparse)
covid_idx <- which(d_covid == 1L)
covid_school_sp <- sparseMatrix(
  i = covid_idx, j = sid[covid_idx], x = 1, dims = c(N, J)
)

# Covariates: base + COVID interactions
Z_covid_sp <- as(d_covid * Z_base, "dgCMatrix")
Z_base_sp  <- as(Z_base, "dgCMatrix")

X_glmnet <- cbind(school_sp, covid_school_sp, Z_base_sp, Z_covid_sp)

rm(school_sp, covid_school_sp, Z_base_sp, Z_covid_sp)
gc(verbose = FALSE)

# Penalty: 0 on school FEs, 1 on covariates
pf <- c(rep(0, 2L * J), rep(1, 2L * p))

cat(sprintf("    X: %s x %d [%.1fs]\n",
            format(nrow(X_glmnet), big.mark = ","), ncol(X_glmnet),
            (proc.time() - t0)["elapsed"]))

cat("    Fitting glmnet (full path, BIC selection)...\n")
t0 <- proc.time()
glm_fit <- glmnet(
  x = X_glmnet, y = y, family = "binomial",
  alpha = 1, penalty.factor = pf, intercept = FALSE,
  standardize = FALSE, thresh = 1e-7, maxit = 1e5
)
time_B <- (proc.time() - t0)["elapsed"]

# BIC-based lambda selection (avoids costly CV)
# BIC = deviance + log(N) * df
N_obs <- length(y)
dev   <- deviance(glm_fit)
df    <- glm_fit$df
bic   <- dev + log(N_obs) * df
best_idx   <- which.min(bic)
lambda_bic <- glm_fit$lambda[best_idx]
cat(sprintf("    Time: %.1fs, lambda(BIC) = %.6f, df = %d\n",
            time_B, lambda_bic, df[best_idx]))

# Extract LASSO coefficients at BIC-optimal lambda
cf_B    <- coef(glm_fit, s = lambda_bic)
coefs_B <- as.numeric(cf_B)[-1]
alpha_B <- coefs_B[1:J]
delta_B <- coefs_B[(J + 1):(2 * J)]
gamma_B <- coefs_B[(2 * J + 1):(2 * J + 2 * p)]

# Identify selected covariate columns (nonzero in LASSO)
selected_cov <- which(gamma_B != 0)
cat(sprintf("    LASSO selected %d / %d covariate columns\n",
            length(selected_cov), 2 * p))

# SE and EB for LASSO
fit_B_list <- list(alpha = alpha_B, delta = delta_B,
                   gamma_b = gamma_B[1:p], gamma_c = gamma_B[(p + 1):(2 * p)])
se_B <- compute_block_se(sid, d_covid, Z_base, fit_B_list,
                         use_covid_cov = TRUE, ridge = 0)
eb_alpha_B <- eb_shrink(alpha_B, se_B$se_alpha)
eb_delta_B <- eb_shrink(delta_B, se_B$se_delta)

rm(X_glmnet, glm_fit, cf_B, coefs_B)
gc(verbose = FALSE)
cat("\n")


# ==============================================================================
# MODEL C: Post-LASSO (LASSO selection → unpenalized block IRLS on subset)
# ==============================================================================
cat("[4] MODEL C: Post-LASSO\n")

# Always include the 4 core covariates (base + COVID interactions)
# Columns 1..p are base, (p+1)..2p are COVID interactions
mandatory_base  <- 1:4
mandatory_covid <- (p + 1):(p + 4)
mandatory <- c(mandatory_base, mandatory_covid)

keep_cols <- sort(unique(c(selected_cov, mandatory)))
cat(sprintf("    Keeping %d / %d covariate columns (LASSO + mandatory)\n",
            length(keep_cols), 2 * p))

# Map selected columns back to base vs COVID-interaction structure
# Columns 1..p → base covariates; columns (p+1)..2p → COVID covariates
keep_base  <- keep_cols[keep_cols <= p]
keep_covid <- keep_cols[keep_cols > p] - p

# If the same base columns appear in both, great.
# Build the Post-LASSO Z matrices
Z_post_base  <- Z_base[, keep_base, drop = FALSE]

# For COVID interactions, we need Z_base columns for the kept COVID cols
# These may differ from keep_base
Z_post_covid <- Z_base[, keep_covid, drop = FALSE]

# We need a custom version for asymmetric base/COVID selection
# Simplification: merge to one covariate matrix [Z_post_base | d * Z_post_covid]
# and use the block solver without the structured COVID split
p_post_b <- length(keep_base)
p_post_c <- length(keep_covid)

cat(sprintf("    Base covariates: %d, COVID covariates: %d\n", p_post_b, p_post_c))

# Build combined covariate matrix for Post-LASSO
# [Z_base_selected | d_covid * Z_covid_selected]
Z_post_combined <- cbind(Z_post_base, d_covid * Z_post_covid)
q_post <- ncol(Z_post_combined)

# Use block solver with this combined matrix (no structured COVID split)
t0 <- proc.time()
fit_C <- sparse_block_irls(sid, d_covid, Z_post_combined, y,
                           use_covid_cov = FALSE)
time_C <- (proc.time() - t0)["elapsed"]
cat(sprintf("    Time: %.1fs\n", time_C))

# SE and EB
fit_C_for_se <- list(alpha = fit_C$alpha, delta = fit_C$delta,
                     gamma_b = fit_C$gamma_b, gamma_c = numeric(0))
se_C <- compute_block_se(sid, d_covid, Z_post_combined, fit_C_for_se,
                         use_covid_cov = FALSE, ridge = 0)
eb_alpha_C <- eb_shrink(fit_C$alpha, se_C$se_alpha)
eb_delta_C <- eb_shrink(fit_C$delta, se_C$se_delta)

rm(Z_post_base, Z_post_covid, Z_post_combined)
gc(verbose = FALSE)
cat("\n")


# ==============================================================================
# 5. RESULTS TABLE
# ==============================================================================
cat("========================================\n")
cat("  RESULTS\n")
cat("========================================\n\n")

cat(sprintf("  Model A (MLE):        %.1fs, %d IRLS iters\n", time_A, fit_A$iter))
cat(sprintf("  Model B (LASSO):      %.1fs\n", time_B))
cat(sprintf("  Model C (Post-LASSO): %.1fs, %d IRLS iters\n\n", time_C, fit_C$iter))

# Build results table with EB estimates of both alpha and delta
results <- data.table(
  school_id   = 1:J,
  school_type = school_dt$school_type,
  # Truth
  true_alpha  = school_dt$alpha_j,
  true_delta  = school_dt$delta_alpha_j,
  # EB-shrunk alpha (school quality)
  mle_alpha   = eb_alpha_A,
  lasso_alpha = eb_alpha_B,
  post_alpha  = eb_alpha_C,
  # EB-shrunk delta (COVID inflation, log-odds)
  mle_delta   = eb_delta_A,
  lasso_delta = eb_delta_B,
  post_delta  = eb_delta_C
)

# --- Compute probability-scale inflation measures ---
# Reference student (X=0 after centering), so:
#   p_baseline = logistic(alpha_j)
#   p_covid    = logistic(alpha_j + delta_j)
#   Raw increase = p_covid - p_baseline
#   RRR = (p_covid - p_baseline) / (1 - p_baseline)

# Truth
results[, true_p0  := plogis(true_alpha)]
results[, true_p1  := plogis(true_alpha + true_delta)]
results[, true_raw := true_p1 - true_p0]
results[, true_rrr := true_raw / pmax(1 - true_p0, 1e-6)]

# MLE
results[, mle_p0  := plogis(mle_alpha)]
results[, mle_p1  := plogis(mle_alpha + mle_delta)]
results[, mle_raw := mle_p1 - mle_p0]
results[, mle_rrr := mle_raw / pmax(1 - mle_p0, 1e-6)]

# LASSO
results[, lasso_p0  := plogis(lasso_alpha)]
results[, lasso_p1  := plogis(lasso_alpha + lasso_delta)]
results[, lasso_raw := lasso_p1 - lasso_p0]
results[, lasso_rrr := lasso_raw / pmax(1 - lasso_p0, 1e-6)]

# Post-LASSO
results[, post_p0  := plogis(post_alpha)]
results[, post_p1  := plogis(post_alpha + post_delta)]
results[, post_raw := post_p1 - post_p0]
results[, post_rrr := post_raw / pmax(1 - post_p0, 1e-6)]

# Bin by true school quality (overall percentile)
results[, alpha_pctl := frank(true_alpha) / .N]
results[, bin := cut(alpha_pctl, breaks = seq(0, 1, 0.05),
                     labels = FALSE, include.lowest = TRUE)]
results[, bin_center := (bin - 0.5) / 20]

# --- Summary statistics ---
cat("Mean inflation by school type (probability scale, EB estimates):\n")
summ <- results[, .(
  True_raw  = round(mean(true_raw), 4),
  MLE_raw   = round(mean(mle_raw), 4),
  Post_raw  = round(mean(post_raw), 4),
  True_RRR  = round(mean(true_rrr), 4),
  MLE_RRR   = round(mean(mle_rrr), 4),
  Post_RRR  = round(mean(post_rrr), 4)
), by = school_type]
print(summ)
cat("\n")

cat("Correlation of EB estimates with truth:\n")
cat(sprintf("  Raw increase:  MLE %.3f | LASSO %.3f | Post %.3f\n",
            cor(results$mle_raw, results$true_raw),
            cor(results$lasso_raw, results$true_raw),
            cor(results$post_raw, results$true_raw)))
cat(sprintf("  RRR:           MLE %.3f | LASSO %.3f | Post %.3f\n\n",
            cor(results$mle_rrr, results$true_rrr),
            cor(results$lasso_rrr, results$true_rrr),
            cor(results$post_rrr, results$true_rrr)))


# ==============================================================================
# 6. PLOTS
# ==============================================================================
cat("Generating plots...\n")

fig_dir <- file.path(Sys.getenv("HOME"), "A-Levels", "output", "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

model_colors <- c(True  = "#D55E00",
                  MLE   = "grey50",
                  LASSO = "black",
                  Post  = "#0072B2")
model_labels <- c(True  = "True (DGP)",
                  MLE   = "Unpenalized MLE + EB",
                  LASSO = "Penalized MLE (LASSO) + EB",
                  Post  = "Post-LASSO + EB")

# --- FIGURE 1: Raw Probability Increase (p1 - p0) ---
long_raw <- melt(
  results,
  id.vars      = c("bin_center", "school_type"),
  measure.vars = c("true_raw", "mle_raw", "lasso_raw", "post_raw"),
  variable.name = "Model",
  value.name    = "prob_increase"
)
long_raw[, Model := factor(
  sub("_raw$", "", Model),
  levels = c("true", "mle", "lasso", "post"),
  labels = c("True", "MLE", "LASSO", "Post")
)]

plot_raw <- long_raw[, .(mean_val = mean(prob_increase, na.rm = TRUE)),
                     by = .(bin_center, school_type, Model)]

plt_prob <- ggplot(plot_raw, aes(x = bin_center, y = mean_val,
                                color = Model, linetype = Model)) +
  geom_line(linewidth = 1) +
  geom_point(size = 1.5) +
  facet_wrap(~school_type) +
  scale_color_manual(values = model_colors, labels = model_labels) +
  scale_linetype_manual(
    values = c(True = "solid", MLE = "dashed", LASSO = "dotdash", Post = "solid"),
    labels = model_labels
  ) +
  labs(
    title    = "COVID Grade Inflation: Raw Probability Increase",
    subtitle = expression(
      "Inflation = " * P[covid] - P[baseline] *
      "  (reference student, EB-shrunk school effects)"
    ),
    x     = "School Quality Percentile",
    y     = expression(P[covid] - P[baseline]),
    color    = NULL,
    linetype = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position  = "bottom",
    strip.text       = element_text(face = "bold", size = 12),
    legend.text      = element_text(size = 10),
    plot.subtitle    = element_text(size = 10)
  )

out_prob <- file.path(fig_dir, "inflation_probability_increase.pdf")
ggsave(out_prob, plt_prob, width = 12, height = 6)
cat(sprintf("  Figure 1 saved: %s\n", out_prob))


# --- FIGURE 2: Relative Risk Reduction (p1 - p0) / (1 - p0) ---
long_rrr <- melt(
  results,
  id.vars      = c("bin_center", "school_type"),
  measure.vars = c("true_rrr", "mle_rrr", "lasso_rrr", "post_rrr"),
  variable.name = "Model",
  value.name    = "rrr"
)
long_rrr[, Model := factor(
  sub("_rrr$", "", Model),
  levels = c("true", "mle", "lasso", "post"),
  labels = c("True", "MLE", "LASSO", "Post")
)]

plot_rrr <- long_rrr[, .(mean_val = mean(rrr, na.rm = TRUE)),
                     by = .(bin_center, school_type, Model)]

plt_rrr <- ggplot(plot_rrr, aes(x = bin_center, y = mean_val,
                                color = Model, linetype = Model)) +
  geom_line(linewidth = 1) +
  geom_point(size = 1.5) +
  facet_wrap(~school_type) +
  scale_color_manual(values = model_colors, labels = model_labels) +
  scale_linetype_manual(
    values = c(True = "solid", MLE = "dashed", LASSO = "dotdash", Post = "solid"),
    labels = model_labels
  ) +
  labs(
    title    = "COVID Grade Inflation: Relative Risk Reduction",
    subtitle = expression(
      "RRR = " * frac(P[covid] - P[baseline], 1 - P[baseline]) *
      "  (fraction of 'room to inflate' consumed)"
    ),
    x     = "School Quality Percentile",
    y     = expression(frac(P[covid] - P[baseline], 1 - P[baseline])),
    color    = NULL,
    linetype = NULL
  ) +
  scale_y_continuous(labels = scales::percent_format()) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position  = "bottom",
    strip.text       = element_text(face = "bold", size = 12),
    legend.text      = element_text(size = 10),
    plot.subtitle    = element_text(size = 10)
  )

out_rrr <- file.path(fig_dir, "inflation_rrr.pdf")
ggsave(out_rrr, plt_rrr, width = 12, height = 6)
cat(sprintf("  Figure 2 saved: %s\n", out_rrr))


# --- Save results ---
res_dir <- file.path(Sys.getenv("HOME"), "A-Levels", "output", "results")
dir.create(res_dir, recursive = TRUE, showWarnings = FALSE)
saveRDS(results, file.path(res_dir, "single_sample_results.rds"))
saveRDS(school_dt, file.path(res_dir, "school_structure.rds"))

cat("\nDone!\n")
