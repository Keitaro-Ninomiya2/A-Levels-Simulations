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
col_means <- attr(Z_base, "scaled:center")              # save ALL centering constants
GCSE_grand_mean <- col_means[1]

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

# Build results table with EB estimates and SEs
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
  post_delta  = eb_delta_C,
  # Standard errors (needed for parametric bootstrap)
  mle_se_alpha   = se_A$se_alpha,
  mle_se_delta   = se_A$se_delta,
  lasso_se_alpha = se_B$se_alpha,
  lasso_se_delta = se_B$se_delta,
  post_se_alpha  = se_C$se_alpha,
  post_se_delta  = se_C$se_delta
)

# ==============================================================================
# 6. CALIBRATE "MARGINAL STUDENT" TO ZERO DENOMINATOR BIAS
# ==============================================================================
# The model's alpha_hat absorbs population-mean covariate effects, so it
# systematically differs from the DGP's alpha_j. At a given reference z-score,
# this creates a denominator bias: mean((1 - p_hat) - (1 - p_true)) != 0.
#
# We find z* such that this bias is exactly zero, ensuring a fair comparison
# between estimated and true RRR on the same probability scale.
#
# Truth uses the DGP formula:
#   p_true = logistic(alpha_j + eta_GCSE * z_raw)
# Estimates use:
#   p_hat  = logistic(alpha_hat_j + gamma_hat_GCSE * z_centered)
#         = logistic(alpha_hat_j + gamma_hat_GCSE * (z_raw - GCSE_grand_mean))
# ==============================================================================

cat("[5] Calibrating marginal student z-score...\n")
cat(sprintf("    GCSE grand mean (centering constant): %.4f\n", GCSE_grand_mean))

eta_GCSE <- params$eta["GCSE_std"]           # true DGP = 0.80
delta_eta_GCSE <- params$delta_eta["GCSE_std"] # true COVID shift = 0.15

# Build the FULL reference covariate vector for a student with
# raw profile: (GCSE=z, SES=0, Minority=0, Gender=0, interactions, noise=0)
# This properly accounts for alpha_hat absorbing ALL mean covariate effects.
build_ref_raw <- function(z_raw) {
  # 21-vector matching cov_names order:
  # GCSE_std, SES, Minority, Gender, GCSE_std_sq,
  # GCSE_SES, GCSE_Minority, GCSE_Gender, SES_Minority, SES_Gender, Minority_Gender,
  # Z_1 ... Z_n_noise
  ref_raw <- rep(0, p)
  ref_raw[1] <- z_raw           # GCSE_std
  ref_raw[5] <- z_raw^2         # GCSE_std_sq
  # All other terms are 0 (SES=0, MIN=0, GEN=0 → interactions=0, noise=0)
  ref_raw
}

# MLE (Model A) gamma for calibration — uses all covariates, most principled
calculate_denom_bias <- function(z_raw) {
  ref_raw      <- build_ref_raw(z_raw)
  ref_centered <- ref_raw - col_means  # centered scale

  # Truth: only the 4 true covariates matter
  #   latent_true = alpha_j + eta_GCSE * z + eta_SES * 0 + ... = alpha_j + 0.8 * z
  p_true <- plogis(school_dt$alpha_j + eta_GCSE * z_raw)

  # Estimate (Model A, MLE): alpha_hat + gamma_b' * Z_ref_centered
  #   alpha_hat already absorbed E[X|j] effects; gamma_b' * (X_ref - Xbar)
  #   undoes the absorbed mean, giving latent ≈ alpha_j + eta' * X_ref
  est_shift <- drop(ref_centered %*% fit_A$gamma_b)
  p_hat     <- plogis(eb_alpha_A + est_shift)

  mean((1 - p_hat) - (1 - p_true))
}

# Scan grid first
z_grid <- seq(-4.0, 2.0, by = 0.25)
bias_grid <- sapply(z_grid, calculate_denom_bias)
cat("    Denominator bias scan:\n")
print(data.table(z = z_grid, bias = round(bias_grid, 6))[abs(bias) < 0.02])

# Find the root
sign_changes <- which(diff(sign(bias_grid)) != 0)
if (length(sign_changes) > 0) {
  lo <- z_grid[sign_changes[1]]
  hi <- z_grid[sign_changes[1] + 1]
  opt <- uniroot(calculate_denom_bias, interval = c(lo, hi), tol = 1e-10)
  GCSE_OPT <- opt$root
  remaining_bias <- opt$f.root
  cat(sprintf("\n    ** Optimal z* = %.4f (remaining bias = %.2e) **\n",
              GCSE_OPT, remaining_bias))
} else {
  # If no sign change, pick the z with smallest absolute bias
  best_idx <- which.min(abs(bias_grid))
  GCSE_OPT <- z_grid[best_idx]
  remaining_bias <- bias_grid[best_idx]
  cat(sprintf("\n    ** No exact root. Best z* = %.2f (bias = %.5f) **\n",
              GCSE_OPT, remaining_bias))
}

# Compute model-specific reference shifts at the calibrated z
ref_raw_opt      <- build_ref_raw(GCSE_OPT)
ref_centered_opt <- ref_raw_opt - col_means

# Each model uses its OWN estimated gamma for a consistent reference
# MLE
mle_ref_base  <- drop(ref_centered_opt %*% fit_A$gamma_b)
mle_ref_covid <- drop(ref_centered_opt %*% (fit_A$gamma_b + fit_A$gamma_c))

# LASSO
lasso_ref_base  <- drop(ref_centered_opt %*% fit_B_list$gamma_b)
lasso_ref_covid <- drop(ref_centered_opt %*% (fit_B_list$gamma_b + fit_B_list$gamma_c))

# Post-LASSO: uses subset of covariates
post_ref_centered_base  <- ref_centered_opt[keep_base]
post_ref_centered_covid <- ref_centered_opt[keep_covid]
post_ref_base  <- drop(post_ref_centered_base %*% fit_C$gamma_b[1:p_post_b])
post_ref_covid <- post_ref_base + drop(post_ref_centered_covid %*%
                    fit_C$gamma_b[(p_post_b + 1):(p_post_b + p_post_c)])

# Truth: use true DGP coefficients
TRUE_REF_BASE  <- eta_GCSE * GCSE_OPT   # only GCSE matters for SES=0, MIN=0, GEN=0
TRUE_REF_COVID <- TRUE_REF_BASE + delta_eta_GCSE * GCSE_OPT

cat(sprintf("    MLE shifts:   base=%.3f, covid=%.3f\n", mle_ref_base, mle_ref_covid))
cat(sprintf("    Post shifts:  base=%.3f, covid=%.3f\n", post_ref_base, post_ref_covid))
cat(sprintf("    Truth shifts: base=%.3f, covid=%.3f\n", TRUE_REF_BASE, TRUE_REF_COVID))


# --- Compute probability-scale inflation measures ---
# Each model uses its own gamma to evaluate the SAME reference student (GCSE=z*)

# Truth
results[, true_p0  := plogis(true_alpha + TRUE_REF_BASE)]
results[, true_p1  := plogis(true_alpha + true_delta + TRUE_REF_COVID)]
results[, true_raw := true_p1 - true_p0]
results[, true_rrr := true_raw / (1 - true_p0)]

# MLE
results[, mle_p0  := plogis(mle_alpha + mle_ref_base)]
results[, mle_p1  := plogis(mle_alpha + mle_delta + mle_ref_covid)]
results[, mle_raw := mle_p1 - mle_p0]
results[, mle_rrr := mle_raw / (1 - mle_p0)]

# LASSO
results[, lasso_p0  := plogis(lasso_alpha + lasso_ref_base)]
results[, lasso_p1  := plogis(lasso_alpha + lasso_delta + lasso_ref_covid)]
results[, lasso_raw := lasso_p1 - lasso_p0]
results[, lasso_rrr := lasso_raw / (1 - lasso_p0)]

# Post-LASSO
results[, post_p0  := plogis(post_alpha + post_ref_base)]
results[, post_p1  := plogis(post_alpha + post_delta + post_ref_covid)]
results[, post_raw := post_p1 - post_p0]
results[, post_rrr := post_raw / (1 - post_p0)]

# Bin by true school quality
results[, alpha_pctl := frank(true_alpha) / .N]
results[, bin := cut(alpha_pctl, breaks = seq(0, 1, 0.05),
                     labels = FALSE, include.lowest = TRUE)]
results[, bin_center := (bin - 0.5) / 20]

# --- Summary statistics ---
cat("\nMean inflation by school type (calibrated marginal student):\n")
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

# --- Denominator bias verification ---
cat("Denominator bias verification (should be ~0 for Post-LASSO):\n")
for (model in c("mle", "lasso", "post")) {
  db <- mean((1 - results[[paste0(model, "_p0")]]) - (1 - results$true_p0))
  cat(sprintf("  %s: %.6f\n", model, db))
}
cat(sprintf("\np0 range: [%.3f, %.3f]\n\n",
            min(results$post_p0), max(results$post_p0)))


# ==============================================================================
# 6b. SECTOR-SPECIFIC REFERENCE PROFILES
# ==============================================================================
# Instead of one universal reference student, each sector (State, Academy,
# Independent) uses its own mean student profile from training data.
# This avoids extrapolation error for Independent schools, whose students
# have systematically different covariate distributions.
# ==============================================================================

cat("[6] Computing sector-specific reference profiles...\n")

# Map students to school types
student_type <- school_dt$school_type[sid]

# Compute per-sector mean covariate profiles (both raw and centered)
sector_ref_raw      <- list()
sector_ref_centered <- list()

for (stype in c("State", "Academy", "Independent")) {
  mask <- which(student_type == stype)
  centered_mean <- colMeans(Z_base[mask, , drop = FALSE])
  raw_mean      <- centered_mean + col_means
  sector_ref_raw[[stype]]      <- raw_mean
  sector_ref_centered[[stype]] <- centered_mean
}

cat("  Sector mean profiles (raw scale):\n")
for (stype in c("State", "Academy", "Independent")) {
  r <- sector_ref_raw[[stype]]
  cat(sprintf("    %s: GCSE=%.3f, SES=%.3f, MIN=%.3f, GEN=%.3f\n",
              stype, r[1], r[2], r[3], r[4]))
}

# --- Clamping: if max p_base > 0.85 for any sector, reduce GCSE ---
P_BASE_CAP <- 0.85

for (stype in c("State", "Academy", "Independent")) {
  s_ids <- which(school_dt$school_type == stype)
  s_raw <- sector_ref_raw[[stype]]

  non_gcse_shift <- params$eta["SES"]      * s_raw[2] +
                    params$eta["Minority"] * s_raw[3] +
                    params$eta["Gender"]   * s_raw[4]
  s_alpha_max <- max(school_dt$alpha_j[s_ids])
  s_max_p <- plogis(s_alpha_max + params$eta["GCSE_std"] * s_raw[1] + non_gcse_shift)

  cat(sprintf("\n  %s max p_base (unclamped): %.3f", stype, s_max_p))

  if (s_max_p > P_BASE_CAP) {
    clamp_fn <- function(z) {
      plogis(s_alpha_max + params$eta["GCSE_std"] * z + non_gcse_shift) - P_BASE_CAP
    }
    clamp_opt  <- uniroot(clamp_fn, interval = c(-5, s_raw[1]), tol = 1e-8)
    gcse_clamp <- clamp_opt$root

    cat(sprintf(" -> clamping GCSE from %.3f to %.3f", s_raw[1], gcse_clamp))

    s_raw_new    <- s_raw
    s_raw_new[1] <- gcse_clamp                        # GCSE_std
    s_raw_new[5] <- gcse_clamp^2                      # GCSE_std_sq
    s_raw_new[6] <- gcse_clamp * s_raw[2]             # GCSE_SES
    s_raw_new[7] <- gcse_clamp * s_raw[3]             # GCSE_Minority
    s_raw_new[8] <- gcse_clamp * s_raw[4]             # GCSE_Gender

    sector_ref_raw[[stype]]      <- s_raw_new
    sector_ref_centered[[stype]] <- s_raw_new - col_means

    new_p <- plogis(s_alpha_max + params$eta["GCSE_std"] * gcse_clamp + non_gcse_shift)
    cat(sprintf(", new max: %.3f", new_p))
  }
  cat("\n")
}

# --- Compute sector-specific shifts for each school ---
stypes <- as.character(school_dt$school_type)

# Truth: 4 true covariates + school-specific COVID biases
true_sec_base  <- numeric(J)
true_sec_covid <- numeric(J)
for (stype in c("State", "Academy", "Independent")) {
  idx <- which(stypes == stype)
  r   <- sector_ref_raw[[stype]]

  base_s <- params$eta["GCSE_std"] * r[1] +
            params$eta["SES"]      * r[2] +
            params$eta["Minority"] * r[3] +
            params$eta["Gender"]   * r[4]

  covid_s <- (params$eta["GCSE_std"] + params$delta_eta["GCSE_std"]) * r[1] +
             (params$eta["SES"]      + params$delta_eta["SES"])      * r[2] +
             (params$eta["Minority"] + params$delta_eta["Minority"]) * r[3] +
             (params$eta["Gender"]   + params$delta_eta["Gender"])   * r[4] +
             school_dt$beta_ses_j[idx] * r[2] +
             school_dt$beta_min_j[idx] * r[3]

  true_sec_base[idx]  <- base_s
  true_sec_covid[idx] <- covid_s
}

# MLE (Model A)
mle_sec_base  <- numeric(J)
mle_sec_covid <- numeric(J)
for (stype in c("State", "Academy", "Independent")) {
  idx <- which(stypes == stype)
  rc  <- sector_ref_centered[[stype]]
  mle_sec_base[idx]  <- drop(rc %*% fit_A$gamma_b)
  mle_sec_covid[idx] <- drop(rc %*% (fit_A$gamma_b + fit_A$gamma_c))
}

# LASSO (Model B)
lasso_sec_base  <- numeric(J)
lasso_sec_covid <- numeric(J)
for (stype in c("State", "Academy", "Independent")) {
  idx <- which(stypes == stype)
  rc  <- sector_ref_centered[[stype]]
  lasso_sec_base[idx]  <- drop(rc %*% fit_B_list$gamma_b)
  lasso_sec_covid[idx] <- drop(rc %*% (fit_B_list$gamma_b + fit_B_list$gamma_c))
}

# Post-LASSO (Model C): uses subset of covariates
post_sec_base  <- numeric(J)
post_sec_covid <- numeric(J)
for (stype in c("State", "Academy", "Independent")) {
  idx <- which(stypes == stype)
  rc  <- sector_ref_centered[[stype]]

  rc_b <- rc[keep_base]
  rc_c <- rc[keep_covid]
  b_s  <- drop(rc_b %*% fit_C$gamma_b[1:p_post_b])
  c_s  <- drop(rc_c %*% fit_C$gamma_b[(p_post_b + 1):(p_post_b + p_post_c)])

  post_sec_base[idx]  <- b_s
  post_sec_covid[idx] <- b_s + c_s
}

# --- Probabilities and RRR ---
results[, sec_true_p0  := plogis(true_alpha + true_sec_base)]
results[, sec_true_p1  := plogis(true_alpha + true_delta + true_sec_covid)]
results[, sec_true_raw := sec_true_p1 - sec_true_p0]
results[, sec_true_rrr := sec_true_raw / (1 - sec_true_p0)]

results[, sec_mle_p0  := plogis(mle_alpha + mle_sec_base)]
results[, sec_mle_p1  := plogis(mle_alpha + mle_delta + mle_sec_covid)]
results[, sec_mle_raw := sec_mle_p1 - sec_mle_p0]
results[, sec_mle_rrr := sec_mle_raw / (1 - sec_mle_p0)]

results[, sec_lasso_p0  := plogis(lasso_alpha + lasso_sec_base)]
results[, sec_lasso_p1  := plogis(lasso_alpha + lasso_delta + lasso_sec_covid)]
results[, sec_lasso_raw := sec_lasso_p1 - sec_lasso_p0]
results[, sec_lasso_rrr := sec_lasso_raw / (1 - sec_lasso_p0)]

results[, sec_post_p0  := plogis(post_alpha + post_sec_base)]
results[, sec_post_p1  := plogis(post_alpha + post_delta + post_sec_covid)]
results[, sec_post_raw := sec_post_p1 - sec_post_p0]
results[, sec_post_rrr := sec_post_raw / (1 - sec_post_p0)]

# --- Summary ---
cat("\nSector-specific mean inflation by school type:\n")
sec_summ <- results[, .(
  True_RRR = round(mean(sec_true_rrr), 4),
  MLE_RRR  = round(mean(sec_mle_rrr), 4),
  LASSO_RRR = round(mean(sec_lasso_rrr), 4),
  Post_RRR = round(mean(sec_post_rrr), 4),
  max_p0   = round(max(sec_true_p0), 3),
  min_1mp0 = round(min(1 - sec_true_p0), 3)
), by = school_type]
print(sec_summ)

cat("\nSector-specific denominator bias:\n")
for (stype in c("State", "Academy", "Independent")) {
  mask <- results$school_type == stype
  db_mle  <- mean((1 - results$sec_mle_p0[mask])  - (1 - results$sec_true_p0[mask]))
  db_post <- mean((1 - results$sec_post_p0[mask]) - (1 - results$sec_true_p0[mask]))
  cat(sprintf("  %s: MLE %.5f, Post-LASSO %.5f\n", stype, db_mle, db_post))
}

cat(sprintf("\nSector-specific p0 range: [%.3f, %.3f]\n",
            min(results$sec_true_p0), max(results$sec_true_p0)))
cat(sprintf("  (1-p0) >= 0.15 for %.0f%% of schools\n\n",
            100 * mean((1 - results$sec_true_p0) >= 0.15)))


# ==============================================================================
# 6c. SECTOR-SPECIFIC PLATT SCALING (SCALE CALIBRATION)
# ==============================================================================
# The global model may be "over-confident" for sectors with compressed grade
# distributions (e.g. Independent). Platt scaling fits:
#   logit(P(Y=1)) = delta_sector + k_sector * Z_raw
# within each sector, where k < 1 flattens the logistic curve.
#
# We then use bin-specific reference profiles (mean covariates per quality bin)
# to compute calibrated RRR at each point along the quality spectrum.
# ==============================================================================

cat("[7] Platt scaling calibration...\n")

# --- Step 1: Raw linear predictors for all students ---
# Model A (MLE)
Z_raw_A <- fit_A$alpha[sid] + fit_A$delta[sid] * d_covid +
           drop(Z_base %*% fit_A$gamma_b) +
           d_covid * drop(Z_base %*% fit_A$gamma_c)

# Model C (Post-LASSO) — uses subset of covariates
Z_post_base_part  <- drop(Z_base[, keep_base, drop = FALSE] %*%
                          fit_C$gamma_b[1:p_post_b])
Z_post_covid_part <- drop(Z_base[, keep_covid, drop = FALSE] %*%
                          fit_C$gamma_b[(p_post_b + 1):(p_post_b + p_post_c)])
Z_raw_C <- fit_C$alpha[sid] + fit_C$delta[sid] * d_covid +
           Z_post_base_part + d_covid * Z_post_covid_part
rm(Z_post_base_part, Z_post_covid_part)

# --- Step 2: Fit Platt scaling per sector ---
student_type <- school_dt$school_type[sid]
platt_A <- list()
platt_C <- list()

for (stype in c("State", "Academy", "Independent")) {
  mask <- which(student_type == stype)

  # MLE
  pfit_A <- glm(y[mask] ~ Z_raw_A[mask], family = binomial)
  platt_A[[stype]] <- coef(pfit_A)  # [1] = delta, [2] = k

  # Post-LASSO
  pfit_C <- glm(y[mask] ~ Z_raw_C[mask], family = binomial)
  platt_C[[stype]] <- coef(pfit_C)
}

cat("  Platt scale factors (k_sector):\n")
cat(sprintf("    MLE:        State=%.4f, Academy=%.4f, Independent=%.4f\n",
            platt_A[["State"]][2], platt_A[["Academy"]][2], platt_A[["Independent"]][2]))
cat(sprintf("    Post-LASSO: State=%.4f, Academy=%.4f, Independent=%.4f\n",
            platt_C[["State"]][2], platt_C[["Academy"]][2], platt_C[["Independent"]][2]))
cat("  Platt intercepts (delta_sector):\n")
cat(sprintf("    MLE:        State=%.4f, Academy=%.4f, Independent=%.4f\n",
            platt_A[["State"]][1], platt_A[["Academy"]][1], platt_A[["Independent"]][1]))
cat(sprintf("    Post-LASSO: State=%.4f, Academy=%.4f, Independent=%.4f\n",
            platt_C[["State"]][1], platt_C[["Academy"]][1], platt_C[["Independent"]][1]))

rm(Z_raw_A, Z_raw_C)
gc(verbose = FALSE)

# --- Step 3: Bin-specific reference profiles ---
# Map students to bins via their school's quality percentile bin
student_bin <- results$bin[match(sid, results$school_id)]

# Compute mean covariate profile per bin (centered scale)
n_bins <- 20L
bin_profiles_centered <- vector("list", n_bins)
bin_profiles_raw      <- vector("list", n_bins)
for (b in seq_len(n_bins)) {
  mask <- which(student_bin == b)
  if (length(mask) > 0) {
    bin_profiles_centered[[b]] <- colMeans(Z_base[mask, , drop = FALSE])
    bin_profiles_raw[[b]]      <- bin_profiles_centered[[b]] + col_means
  }
}

cat(sprintf("\n  Bin profiles computed: %d / %d bins populated\n",
            sum(!sapply(bin_profiles_centered, is.null)), n_bins))

# --- Step 4: Calibrated predictions per school ---
# For each school j: use its bin's mean profile + its own alpha_hat,
# then apply its sector's Platt calibration.

cal_mle_p0  <- numeric(J)
cal_mle_p1  <- numeric(J)
cal_post_p0 <- numeric(J)
cal_post_p1 <- numeric(J)
cal_true_p0 <- numeric(J)
cal_true_p1 <- numeric(J)

for (j in seq_len(J)) {
  b     <- results$bin[j]
  stype <- as.character(results$school_type[j])

  if (is.null(bin_profiles_centered[[b]])) next

  prof_c <- bin_profiles_centered[[b]]
  prof_r <- bin_profiles_raw[[b]]

  # --- MLE raw linear predictors ---
  z_base_A  <- eb_alpha_A[j] + drop(prof_c %*% fit_A$gamma_b)
  z_covid_A <- eb_alpha_A[j] + eb_delta_A[j] +
               drop(prof_c %*% (fit_A$gamma_b + fit_A$gamma_c))

  # Apply Platt calibration
  pa <- platt_A[[stype]]
  cal_mle_p0[j]  <- plogis(pa[1] + pa[2] * z_base_A)
  cal_mle_p1[j]  <- plogis(pa[1] + pa[2] * z_covid_A)

  # --- Post-LASSO raw linear predictors ---
  prof_c_base  <- prof_c[keep_base]
  prof_c_covid <- prof_c[keep_covid]
  z_base_C  <- eb_alpha_C[j] +
               drop(prof_c_base %*% fit_C$gamma_b[1:p_post_b])
  z_covid_C <- eb_alpha_C[j] + eb_delta_C[j] +
               drop(prof_c_base %*% fit_C$gamma_b[1:p_post_b]) +
               drop(prof_c_covid %*% fit_C$gamma_b[(p_post_b + 1):(p_post_b + p_post_c)])

  # Apply Platt calibration
  pc <- platt_C[[stype]]
  cal_post_p0[j] <- plogis(pc[1] + pc[2] * z_base_C)
  cal_post_p1[j] <- plogis(pc[1] + pc[2] * z_covid_C)

  # --- Truth (no calibration needed) ---
  true_base_s  <- params$eta["GCSE_std"] * prof_r[1] +
                  params$eta["SES"]      * prof_r[2] +
                  params$eta["Minority"] * prof_r[3] +
                  params$eta["Gender"]   * prof_r[4]
  true_covid_s <- (params$eta["GCSE_std"] + params$delta_eta["GCSE_std"]) * prof_r[1] +
                  (params$eta["SES"]      + params$delta_eta["SES"])      * prof_r[2] +
                  (params$eta["Minority"] + params$delta_eta["Minority"]) * prof_r[3] +
                  (params$eta["Gender"]   + params$delta_eta["Gender"])   * prof_r[4] +
                  school_dt$beta_ses_j[j] * prof_r[2] +
                  school_dt$beta_min_j[j] * prof_r[3]

  cal_true_p0[j] <- plogis(school_dt$alpha_j[j] + true_base_s)
  cal_true_p1[j] <- plogis(school_dt$alpha_j[j] + school_dt$delta_alpha_j[j] + true_covid_s)
}

# Store in results
results[, cal_true_p0  := cal_true_p0]
results[, cal_true_p1  := cal_true_p1]
results[, cal_true_raw := cal_true_p1 - cal_true_p0]
results[, cal_true_rrr := cal_true_raw / (1 - cal_true_p0)]

results[, cal_mle_p0  := cal_mle_p0]
results[, cal_mle_p1  := cal_mle_p1]
results[, cal_mle_raw := cal_mle_p1 - cal_mle_p0]
results[, cal_mle_rrr := cal_mle_raw / (1 - cal_mle_p0)]

results[, cal_post_p0  := cal_post_p0]
results[, cal_post_p1  := cal_post_p1]
results[, cal_post_raw := cal_post_p1 - cal_post_p0]
results[, cal_post_rrr := cal_post_raw / (1 - cal_post_p0)]

# --- Summary ---
cat("\nPlatt-calibrated RRR by school type:\n")
cal_summ <- results[, .(
  True_RRR = round(mean(cal_true_rrr), 4),
  MLE_RRR  = round(mean(cal_mle_rrr), 4),
  Post_RRR = round(mean(cal_post_rrr), 4),
  max_p0_true = round(max(cal_true_p0), 3),
  max_p0_mle  = round(max(cal_mle_p0), 3)
), by = school_type]
print(cal_summ)

cat("\nPlatt-calibrated denominator bias:\n")
for (stype in c("State", "Academy", "Independent")) {
  mask <- results$school_type == stype
  db_mle  <- mean((1 - results$cal_mle_p0[mask])  - (1 - results$cal_true_p0[mask]))
  db_post <- mean((1 - results$cal_post_p0[mask]) - (1 - results$cal_true_p0[mask]))
  cat(sprintf("  %s: MLE %.5f, Post-LASSO %.5f\n", stype, db_mle, db_post))
}
cat("\n")


# ==============================================================================
# 6d. SECTOR-SPECIFIC REGRESSIONS
# ==============================================================================
# The global model pools all sectors, so the loss function is dominated by
# State (60%) and Academy (33%). Independent schools (7%) may get poorly
# estimated covariate effects. Fitting separate models per sector ensures
# each sector's beta is optimised for its own grade distribution.
# ==============================================================================

cat("[8] Fitting sector-specific models...\n")

student_type <- school_dt$school_type[sid]
P_BASE_CAP_SEP <- 0.85
sector_model_results <- list()

for (stype in c("State", "Academy", "Independent")) {
  cat(sprintf("\n  --- %s ---\n", stype))

  # --- Subset data ---
  s_mask     <- which(student_type == stype)
  s_orig_ids <- sort(unique(sid[s_mask]))
  J_s        <- length(s_orig_ids)
  id_map     <- match(sid[s_mask], s_orig_ids)

  s_sid  <- id_map
  s_dcov <- d_covid[s_mask]
  s_y    <- y[s_mask]
  s_Z    <- Z_base[s_mask, , drop = FALSE]
  N_s    <- length(s_y)

  cat(sprintf("    N=%s students, J=%d schools\n",
              format(N_s, big.mark = ","), J_s))

  # --- Model A: Sector MLE ---
  cat("    MLE...")
  t0 <- proc.time()
  fit_sA <- sparse_block_irls(s_sid, s_dcov, s_Z, s_y, use_covid_cov = TRUE)
  cat(sprintf(" [%.1fs]\n", (proc.time() - t0)["elapsed"]))

  se_sA <- compute_block_se(s_sid, s_dcov, s_Z, fit_sA,
                            use_covid_cov = TRUE, ridge = 0)
  eb_alpha_sA <- eb_shrink(fit_sA$alpha, se_sA$se_alpha)
  eb_delta_sA <- eb_shrink(fit_sA$delta, se_sA$se_delta)

  # --- Model B: Sector LASSO ---
  cat("    LASSO...")
  t0 <- proc.time()

  school_sp_s <- sparseMatrix(i = seq_len(N_s), j = s_sid, x = 1,
                              dims = c(N_s, J_s))
  covid_idx_s <- which(s_dcov == 1L)
  covid_sp_s  <- sparseMatrix(i = covid_idx_s, j = s_sid[covid_idx_s],
                              x = 1, dims = c(N_s, J_s))
  Z_covid_sp_s <- as(s_dcov * s_Z, "dgCMatrix")
  Z_base_sp_s  <- as(s_Z, "dgCMatrix")
  X_s <- cbind(school_sp_s, covid_sp_s, Z_base_sp_s, Z_covid_sp_s)
  rm(school_sp_s, covid_sp_s, Z_base_sp_s, Z_covid_sp_s)

  pf_s <- c(rep(0, 2L * J_s), rep(1, 2L * p))
  glm_fit_s <- glmnet(x = X_s, y = s_y, family = "binomial",
                      alpha = 1, penalty.factor = pf_s, intercept = FALSE,
                      standardize = FALSE, thresh = 1e-7, maxit = 1e5)

  dev_s  <- deviance(glm_fit_s)
  df_s   <- glm_fit_s$df
  bic_s  <- dev_s + log(N_s) * df_s
  best_s <- which.min(bic_s)

  cf_s    <- coef(glm_fit_s, s = glm_fit_s$lambda[best_s])
  coefs_s <- as.numeric(cf_s)[-1]
  gamma_sB <- coefs_s[(2L * J_s + 1):(2L * J_s + 2L * p)]
  selected_cov_s <- which(gamma_sB != 0)

  cat(sprintf(" [%.1fs] selected %d/%d covs\n",
              (proc.time() - t0)["elapsed"], length(selected_cov_s), 2L * p))

  fit_sB <- list(alpha   = coefs_s[1:J_s],
                 delta   = coefs_s[(J_s + 1):(2L * J_s)],
                 gamma_b = gamma_sB[1:p],
                 gamma_c = gamma_sB[(p + 1):(2L * p)])
  se_sB <- compute_block_se(s_sid, s_dcov, s_Z, fit_sB,
                            use_covid_cov = TRUE, ridge = 0)
  eb_alpha_sB <- eb_shrink(fit_sB$alpha, se_sB$se_alpha)
  eb_delta_sB <- eb_shrink(fit_sB$delta, se_sB$se_delta)

  rm(X_s, glm_fit_s, cf_s, coefs_s)

  # --- Model C: Sector Post-LASSO ---
  cat("    Post-LASSO...")
  t0 <- proc.time()

  mandatory_s  <- c(1:4, (p + 1):(p + 4))
  keep_s       <- sort(unique(c(selected_cov_s, mandatory_s)))
  keep_base_s  <- keep_s[keep_s <= p]
  keep_covid_s <- keep_s[keep_s > p] - p
  p_post_b_s   <- length(keep_base_s)
  p_post_c_s   <- length(keep_covid_s)

  Z_post_comb_s <- cbind(s_Z[, keep_base_s, drop = FALSE],
                         s_dcov * s_Z[, keep_covid_s, drop = FALSE])

  fit_sC <- sparse_block_irls(s_sid, s_dcov, Z_post_comb_s, s_y,
                              use_covid_cov = FALSE)
  cat(sprintf(" [%.1fs]\n", (proc.time() - t0)["elapsed"]))

  fit_sC_se <- list(alpha = fit_sC$alpha, delta = fit_sC$delta,
                    gamma_b = fit_sC$gamma_b, gamma_c = numeric(0))
  se_sC <- compute_block_se(s_sid, s_dcov, Z_post_comb_s, fit_sC_se,
                            use_covid_cov = FALSE, ridge = 0)
  eb_alpha_sC <- eb_shrink(fit_sC$alpha, se_sC$se_alpha)
  eb_delta_sC <- eb_shrink(fit_sC$delta, se_sC$se_delta)

  rm(Z_post_comb_s)

  # --- Sector mean covariate profile ---
  sec_mean_c <- colMeans(s_Z)   # centered scale
  sec_mean_r <- sec_mean_c + col_means  # raw scale

  # --- Clamping: reduce GCSE if max p_base > 0.85 ---
  alpha_max_s    <- max(school_dt$alpha_j[s_orig_ids])
  non_gcse_shift <- params$eta["SES"]      * sec_mean_r[2] +
                    params$eta["Minority"] * sec_mean_r[3] +
                    params$eta["Gender"]   * sec_mean_r[4]
  max_p0_s <- plogis(alpha_max_s + params$eta["GCSE_std"] * sec_mean_r[1] +
                     non_gcse_shift)

  if (max_p0_s > P_BASE_CAP_SEP) {
    clamp_fn_s <- function(z) {
      plogis(alpha_max_s + params$eta["GCSE_std"] * z + non_gcse_shift) -
        P_BASE_CAP_SEP
    }
    gcse_clamp_s <- uniroot(clamp_fn_s, c(-5, sec_mean_r[1]), tol = 1e-8)$root
    cat(sprintf("    Clamping GCSE from %.3f to %.3f (max p0: %.3f -> 0.850)\n",
                sec_mean_r[1], gcse_clamp_s, max_p0_s))
    sec_mean_r[1] <- gcse_clamp_s
    sec_mean_r[5] <- gcse_clamp_s^2
    sec_mean_r[6] <- gcse_clamp_s * sec_mean_r[2]
    sec_mean_r[7] <- gcse_clamp_s * sec_mean_r[3]
    sec_mean_r[8] <- gcse_clamp_s * sec_mean_r[4]
    sec_mean_c <- sec_mean_r - col_means
  }

  # --- Reference shifts ---
  # MLE (sector-specific)
  mle_ref_base  <- drop(sec_mean_c %*% fit_sA$gamma_b)
  mle_ref_covid <- drop(sec_mean_c %*% (fit_sA$gamma_b + fit_sA$gamma_c))

  # Post-LASSO (sector-specific)
  post_ref_base  <- drop(sec_mean_c[keep_base_s] %*%
                         fit_sC$gamma_b[1:p_post_b_s])
  post_ref_covid <- post_ref_base +
    drop(sec_mean_c[keep_covid_s] %*%
         fit_sC$gamma_b[(p_post_b_s + 1):(p_post_b_s + p_post_c_s)])

  # Truth: only 4 true covariates + school-specific COVID biases
  true_ref_base <- params$eta["GCSE_std"] * sec_mean_r[1] +
                   params$eta["SES"]      * sec_mean_r[2] +
                   params$eta["Minority"] * sec_mean_r[3] +
                   params$eta["Gender"]   * sec_mean_r[4]

  true_ref_covid_common <-
    (params$eta["GCSE_std"] + params$delta_eta["GCSE_std"]) * sec_mean_r[1] +
    (params$eta["SES"]      + params$delta_eta["SES"])      * sec_mean_r[2] +
    (params$eta["Minority"] + params$delta_eta["Minority"]) * sec_mean_r[3] +
    (params$eta["Gender"]   + params$delta_eta["Gender"])   * sec_mean_r[4]
  true_covid_bias <- school_dt$beta_ses_j[s_orig_ids] * sec_mean_r[2] +
                     school_dt$beta_min_j[s_orig_ids] * sec_mean_r[3]

  # --- Build sector result table ---
  s_alpha <- school_dt$alpha_j[s_orig_ids]
  s_delta <- school_dt$delta_alpha_j[s_orig_ids]

  dt_s <- data.table(
    school_id   = s_orig_ids,
    school_type = factor(stype, levels = c("State", "Academy", "Independent")),
    true_alpha  = s_alpha,
    true_delta  = s_delta,
    # Truth
    sep_true_p0  = plogis(s_alpha + true_ref_base),
    sep_true_p1  = plogis(s_alpha + s_delta + true_ref_covid_common +
                          true_covid_bias),
    # MLE
    sep_mle_p0   = plogis(eb_alpha_sA + mle_ref_base),
    sep_mle_p1   = plogis(eb_alpha_sA + eb_delta_sA + mle_ref_covid),
    # Post-LASSO
    sep_post_p0  = plogis(eb_alpha_sC + post_ref_base),
    sep_post_p1  = plogis(eb_alpha_sC + eb_delta_sC + post_ref_covid)
  )

  sector_model_results[[stype]] <- dt_s
  gc(verbose = FALSE)
}

# --- Combine sector results ---
sep_results <- rbindlist(sector_model_results)

sep_results[, sep_true_raw := sep_true_p1 - sep_true_p0]
sep_results[, sep_true_rrr := sep_true_raw / (1 - sep_true_p0)]
sep_results[, sep_mle_raw  := sep_mle_p1  - sep_mle_p0]
sep_results[, sep_mle_rrr  := sep_mle_raw  / (1 - sep_mle_p0)]
sep_results[, sep_post_raw := sep_post_p1 - sep_post_p0]
sep_results[, sep_post_rrr := sep_post_raw / (1 - sep_post_p0)]

# Merge into main results table
sep_cols <- c("school_id", "sep_true_p0", "sep_true_p1", "sep_true_raw",
              "sep_true_rrr", "sep_mle_p0", "sep_mle_p1", "sep_mle_raw",
              "sep_mle_rrr", "sep_post_p0", "sep_post_p1", "sep_post_raw",
              "sep_post_rrr")
results <- merge(results, sep_results[, ..sep_cols], by = "school_id")

# Bin assignments for merged results (ensure they exist)
results[, alpha_pctl := frank(true_alpha) / .N]
results[, bin := cut(alpha_pctl, breaks = seq(0, 1, 0.05),
                     labels = FALSE, include.lowest = TRUE)]
results[, bin_center := (bin - 0.5) / 20]

# --- Summary ---
cat("\n\nSector-specific regression RRR by school type:\n")
sep_summ <- results[, .(
  True_RRR = round(mean(sep_true_rrr), 4),
  MLE_RRR  = round(mean(sep_mle_rrr), 4),
  Post_RRR = round(mean(sep_post_rrr), 4),
  max_p0   = round(max(sep_true_p0), 3),
  min_1mp0 = round(min(1 - sep_true_p0), 3)
), by = school_type]
print(sep_summ)

cat("\nSector-specific regression denominator bias:\n")
for (stype in c("State", "Academy", "Independent")) {
  mask <- results$school_type == stype
  db_mle  <- mean((1 - results$sep_mle_p0[mask])  - (1 - results$sep_true_p0[mask]))
  db_post <- mean((1 - results$sep_post_p0[mask]) - (1 - results$sep_true_p0[mask]))
  cat(sprintf("  %s: MLE %.5f, Post-LASSO %.5f\n", stype, db_mle, db_post))
}

cat("\nCorrelation with truth (sector-specific models):\n")
cat(sprintf("  Raw increase:  MLE %.3f | Post %.3f\n",
            cor(results$sep_mle_raw, results$sep_true_raw),
            cor(results$sep_post_raw, results$sep_true_raw)))
cat(sprintf("  RRR:           MLE %.3f | Post %.3f\n\n",
            cor(results$sep_mle_rrr, results$sep_true_rrr),
            cor(results$sep_post_rrr, results$sep_true_rrr)))


# ==============================================================================
# 7. PLOTS
# ==============================================================================
cat("[9] Generating plots...\n")

fig_dir <- file.path(Sys.getenv("HOME"), "A-Levels", "output", "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

model_colors <- c(True  = "#D55E00",   # vermillion
                  MLE   = "#009E73",   # green
                  LASSO = "#000000",   # black
                  Post  = "#0072B2")   # blue
model_labels <- c(True  = "True (DGP)",
                  MLE   = "Unpenalized MLE + EB",
                  LASSO = "Penalized MLE (LASSO) + EB",
                  Post  = "Post-LASSO + EB")
model_shapes <- c(True = 16, MLE = 17, LASSO = 15, Post = 18)

agg_with_se <- function(dt, value_col) {
  dt[, .(mean_val = mean(get(value_col), na.rm = TRUE),
         se_val   = sd(get(value_col), na.rm = TRUE) / sqrt(.N)),
     by = .(bin_center, school_type, Model)]
}

ref_label <- sprintf("Calibrated marginal student (GCSE z = %.2f, zero denom. bias)",
                     GCSE_OPT)


# --- FIGURE 1: Raw Probability Increase ---
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
plot_raw <- agg_with_se(long_raw, "prob_increase")

plt_prob <- ggplot(plot_raw, aes(x = bin_center, y = mean_val,
                                color = Model, shape = Model)) +
  geom_ribbon(aes(ymin = mean_val - 1.96 * se_val,
                  ymax = mean_val + 1.96 * se_val,
                  fill = Model), alpha = 0.12, colour = NA) +
  geom_line(aes(linetype = Model), linewidth = 0.9,
            position = position_dodge(width = 0.01)) +
  geom_point(size = 2, position = position_dodge(width = 0.01)) +
  facet_wrap(~school_type) +
  scale_color_manual(values = model_colors, labels = model_labels) +
  scale_fill_manual(values = model_colors, labels = model_labels) +
  scale_shape_manual(values = model_shapes, labels = model_labels) +
  scale_linetype_manual(
    values = c(True = "solid", MLE = "dashed", LASSO = "dotdash", Post = "solid"),
    labels = model_labels
  ) +
  labs(
    title    = "COVID Grade Inflation: Raw Probability Increase",
    subtitle = paste0("Inflation = P_covid - P_baseline  |  ", ref_label),
    x = "School Quality Percentile",
    y = expression(P[covid] - P[baseline])
  ) +
  guides(color = guide_legend(NULL), fill = guide_legend(NULL),
         shape = guide_legend(NULL), linetype = guide_legend(NULL)) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "bottom",
    strip.text      = element_text(face = "bold", size = 12),
    legend.text     = element_text(size = 10),
    plot.subtitle   = element_text(size = 10)
  )

out_prob <- file.path(fig_dir, "inflation_probability_increase.pdf")
ggsave(out_prob, plt_prob, width = 12, height = 6)
cat(sprintf("  Figure 1 saved: %s\n", out_prob))


# --- FIGURE 2: RRR (Calibrated Marginal Student) ---
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
plot_rrr <- agg_with_se(long_rrr, "rrr")

plt_rrr <- ggplot(plot_rrr, aes(x = bin_center, y = mean_val,
                                color = Model, shape = Model)) +
  geom_ribbon(aes(ymin = mean_val - 1.96 * se_val,
                  ymax = mean_val + 1.96 * se_val,
                  fill = Model), alpha = 0.12, colour = NA) +
  geom_line(aes(linetype = Model), linewidth = 0.9,
            position = position_dodge(width = 0.01)) +
  geom_point(size = 2, position = position_dodge(width = 0.01)) +
  facet_wrap(~school_type) +
  scale_color_manual(values = model_colors, labels = model_labels) +
  scale_fill_manual(values = model_colors, labels = model_labels) +
  scale_shape_manual(values = model_shapes, labels = model_labels) +
  scale_linetype_manual(
    values = c(True = "solid", MLE = "dashed", LASSO = "dotdash", Post = "solid"),
    labels = model_labels
  ) +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(
    title    = "Relative Risk Reduction (Calibrated Marginal Student)",
    subtitle = paste0("RRR = (P_covid - P_base) / (1 - P_base)  |  ", ref_label),
    x = "School Quality Percentile",
    y = expression(frac(P[covid] - P[baseline], 1 - P[baseline]))
  ) +
  guides(color = guide_legend(NULL), fill = guide_legend(NULL),
         shape = guide_legend(NULL), linetype = guide_legend(NULL)) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "bottom",
    strip.text      = element_text(face = "bold", size = 12),
    legend.text     = element_text(size = 10),
    plot.subtitle   = element_text(size = 10)
  )

out_rrr <- file.path(fig_dir, "RRR_calibrated_marginal.pdf")
ggsave(out_rrr, plt_rrr, width = 12, height = 6)
cat(sprintf("  Figure 2 saved: %s\n", out_rrr))


# --- FIGURE 3: RRR (Sector-Specific Reference Profiles) ---
long_sec <- melt(
  results,
  id.vars      = c("bin_center", "school_type"),
  measure.vars = c("sec_true_rrr", "sec_mle_rrr", "sec_lasso_rrr", "sec_post_rrr"),
  variable.name = "Model",
  value.name    = "rrr"
)
long_sec[, Model := factor(
  sub("^sec_", "", sub("_rrr$", "", Model)),
  levels = c("true", "mle", "lasso", "post"),
  labels = c("True", "MLE", "LASSO", "Post")
)]
plot_sec <- agg_with_se(long_sec, "rrr")

plt_sec <- ggplot(plot_sec, aes(x = bin_center, y = mean_val,
                                color = Model, shape = Model)) +
  geom_ribbon(aes(ymin = mean_val - 1.96 * se_val,
                  ymax = mean_val + 1.96 * se_val,
                  fill = Model), alpha = 0.12, colour = NA) +
  geom_line(aes(linetype = Model), linewidth = 0.9,
            position = position_dodge(width = 0.01)) +
  geom_point(size = 2, position = position_dodge(width = 0.01)) +
  facet_wrap(~school_type) +
  scale_color_manual(values = model_colors, labels = model_labels) +
  scale_fill_manual(values = model_colors, labels = model_labels) +
  scale_shape_manual(values = model_shapes, labels = model_labels) +
  scale_linetype_manual(
    values = c(True = "solid", MLE = "dashed", LASSO = "dotdash", Post = "solid"),
    labels = model_labels
  ) +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(
    title    = "Relative Risk Reduction (Sector-Specific Reference Profiles)",
    subtitle = "RRR = (P_covid - P_base) / (1 - P_base)  |  Each sector uses its own mean student profile",
    x = "School Quality Percentile",
    y = expression(frac(P[covid] - P[baseline], 1 - P[baseline]))
  ) +
  guides(color = guide_legend(NULL), fill = guide_legend(NULL),
         shape = guide_legend(NULL), linetype = guide_legend(NULL)) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "bottom",
    strip.text      = element_text(face = "bold", size = 12),
    legend.text     = element_text(size = 10),
    plot.subtitle   = element_text(size = 10)
  )

out_sec <- file.path(fig_dir, "RRR_sector_specific.pdf")
ggsave(out_sec, plt_sec, width = 12, height = 6)
cat(sprintf("  Figure 3 saved: %s\n", out_sec))


# --- FIGURE 4: RRR (Platt-Scaled Calibration + Bin Profiles) ---
long_cal <- melt(
  results,
  id.vars      = c("bin_center", "school_type"),
  measure.vars = c("cal_true_rrr", "cal_mle_rrr", "cal_post_rrr"),
  variable.name = "Model",
  value.name    = "rrr"
)
long_cal[, Model := factor(
  sub("^cal_", "", sub("_rrr$", "", Model)),
  levels = c("true", "mle", "post"),
  labels = c("True", "MLE", "Post")
)]
plot_cal <- agg_with_se(long_cal, "rrr")

cal_colors <- model_colors[c("True", "MLE", "Post")]
cal_labels <- model_labels[c("True", "MLE", "Post")]
cal_shapes <- model_shapes[c("True", "MLE", "Post")]

k_label <- sprintf("k: State=%.3f, Acad=%.3f, Indep=%.3f",
                    platt_A[["State"]][2], platt_A[["Academy"]][2],
                    platt_A[["Independent"]][2])

plt_cal <- ggplot(plot_cal, aes(x = bin_center, y = mean_val,
                                color = Model, shape = Model)) +
  geom_ribbon(aes(ymin = mean_val - 1.96 * se_val,
                  ymax = mean_val + 1.96 * se_val,
                  fill = Model), alpha = 0.12, colour = NA) +
  geom_line(aes(linetype = Model), linewidth = 0.9,
            position = position_dodge(width = 0.01)) +
  geom_point(size = 2, position = position_dodge(width = 0.01)) +
  facet_wrap(~school_type) +
  scale_color_manual(values = cal_colors, labels = cal_labels) +
  scale_fill_manual(values = cal_colors, labels = cal_labels) +
  scale_shape_manual(values = cal_shapes, labels = cal_labels) +
  scale_linetype_manual(
    values = c(True = "solid", MLE = "dashed", Post = "solid"),
    labels = cal_labels
  ) +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(
    title    = "RRR with Platt-Scaled Calibration (Bin-Specific Profiles)",
    subtitle = paste0("logit(P) = delta_s + k_s * Z_raw  |  ", k_label),
    x = "School Quality Percentile",
    y = expression(frac(P[covid] - P[baseline], 1 - P[baseline]))
  ) +
  guides(color = guide_legend(NULL), fill = guide_legend(NULL),
         shape = guide_legend(NULL), linetype = guide_legend(NULL)) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "bottom",
    strip.text      = element_text(face = "bold", size = 12),
    legend.text     = element_text(size = 10),
    plot.subtitle   = element_text(size = 10)
  )

out_cal <- file.path(fig_dir, "RRR_scaled_calibration.pdf")
ggsave(out_cal, plt_cal, width = 12, height = 6)
cat(sprintf("  Figure 4 saved: %s\n", out_cal))


# --- FIGURE 5: RRR (Sector-Specific Regressions) ---
long_sep <- melt(
  results,
  id.vars      = c("bin_center", "school_type"),
  measure.vars = c("sep_true_rrr", "sep_mle_rrr", "sep_post_rrr"),
  variable.name = "Model",
  value.name    = "rrr"
)
long_sep[, Model := factor(
  sub("^sep_", "", sub("_rrr$", "", Model)),
  levels = c("true", "mle", "post"),
  labels = c("True", "MLE", "Post")
)]
plot_sep <- agg_with_se(long_sep, "rrr")

sep_colors <- model_colors[c("True", "MLE", "Post")]
sep_labels <- model_labels[c("True", "MLE", "Post")]
sep_shapes <- model_shapes[c("True", "MLE", "Post")]

plt_sep <- ggplot(plot_sep, aes(x = bin_center, y = mean_val,
                                color = Model, shape = Model)) +
  geom_ribbon(aes(ymin = mean_val - 1.96 * se_val,
                  ymax = mean_val + 1.96 * se_val,
                  fill = Model), alpha = 0.12, colour = NA) +
  geom_line(aes(linetype = Model), linewidth = 0.9,
            position = position_dodge(width = 0.01)) +
  geom_point(size = 2, position = position_dodge(width = 0.01)) +
  facet_wrap(~school_type) +
  scale_color_manual(values = sep_colors, labels = sep_labels) +
  scale_fill_manual(values = sep_colors, labels = sep_labels) +
  scale_shape_manual(values = sep_shapes, labels = sep_labels) +
  scale_linetype_manual(
    values = c(True = "solid", MLE = "dashed", Post = "solid"),
    labels = sep_labels
  ) +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(
    title    = "RRR with Sector-Specific Regressions",
    subtitle = "Separate models fitted per sector (State, Academy, Independent) to eliminate weighting bias",
    x = "School Quality Percentile",
    y = expression(frac(P[covid] - P[baseline], 1 - P[baseline]))
  ) +
  guides(color = guide_legend(NULL), fill = guide_legend(NULL),
         shape = guide_legend(NULL), linetype = guide_legend(NULL)) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "bottom",
    strip.text      = element_text(face = "bold", size = 12),
    legend.text     = element_text(size = 10),
    plot.subtitle   = element_text(size = 10)
  )

out_sep <- file.path(fig_dir, "RRR_sector_regressions.pdf")
ggsave(out_sep, plt_sep, width = 12, height = 6)
cat(sprintf("  Figure 5 saved: %s\n", out_sep))


# --- FIGURE 6: Probability Increase (Sector-Specific Regressions) ---
long_sep_raw <- melt(
  results,
  id.vars      = c("bin_center", "school_type"),
  measure.vars = c("sep_true_raw", "sep_mle_raw", "sep_post_raw"),
  variable.name = "Model",
  value.name    = "prob_increase"
)
long_sep_raw[, Model := factor(
  sub("^sep_", "", sub("_raw$", "", Model)),
  levels = c("true", "mle", "post"),
  labels = c("True", "MLE", "Post")
)]
plot_sep_raw <- agg_with_se(long_sep_raw, "prob_increase")

plt_sep_raw <- ggplot(plot_sep_raw, aes(x = bin_center, y = mean_val,
                                        color = Model, shape = Model)) +
  geom_ribbon(aes(ymin = mean_val - 1.96 * se_val,
                  ymax = mean_val + 1.96 * se_val,
                  fill = Model), alpha = 0.12, colour = NA) +
  geom_line(aes(linetype = Model), linewidth = 0.9,
            position = position_dodge(width = 0.01)) +
  geom_point(size = 2, position = position_dodge(width = 0.01)) +
  facet_wrap(~school_type) +
  scale_color_manual(values = sep_colors, labels = sep_labels) +
  scale_fill_manual(values = sep_colors, labels = sep_labels) +
  scale_shape_manual(values = sep_shapes, labels = sep_labels) +
  scale_linetype_manual(
    values = c(True = "solid", MLE = "dashed", Post = "solid"),
    labels = sep_labels
  ) +
  labs(
    title    = "COVID Grade Inflation: Raw Probability Increase (Sector-Specific Regressions)",
    subtitle = "Inflation = P_covid - P_baseline  |  Separate models per sector",
    x = "School Quality Percentile",
    y = expression(P[covid] - P[baseline])
  ) +
  guides(color = guide_legend(NULL), fill = guide_legend(NULL),
         shape = guide_legend(NULL), linetype = guide_legend(NULL)) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "bottom",
    strip.text      = element_text(face = "bold", size = 12),
    legend.text     = element_text(size = 10),
    plot.subtitle   = element_text(size = 10)
  )

out_sep_raw <- file.path(fig_dir, "inflation_sector_regressions.pdf")
ggsave(out_sep_raw, plt_sep_raw, width = 12, height = 6)
cat(sprintf("  Figure 6 saved: %s\n", out_sep_raw))


# --- Save results ---
res_dir <- file.path(Sys.getenv("HOME"), "A-Levels", "output", "results")
dir.create(res_dir, recursive = TRUE, showWarnings = FALSE)
saveRDS(results, file.path(res_dir, "single_sample_results.rds"))
saveRDS(school_dt, file.path(res_dir, "school_structure.rds"))

cat("\nDone!\n")
