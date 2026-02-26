# ==============================================================================
# run_single.R — Single-sample diagnostic (Type-Specific Slopes)
# ==============================================================================
#   Estimator A: Global model (common slopes — misspecified)
#   Estimator B: Split by type (separate models per school type)
#   Estimator C: Proposed fix (LASSO on Demo × Type interactions)
#
# Produces 6 diagnostic figures comparing all three estimators to truth.
# ==============================================================================

.libPaths(c(file.path(Sys.getenv("HOME"), "A-Levels", "R_libs"), .libPaths()))

base_dir <- file.path(Sys.getenv("HOME"), "A-Levels", "school_type_fix", "Post_LASSO")
source(file.path(base_dir, "R", "01_dgp.R"))
library(Matrix)
library(glmnet)
library(ggplot2)
library(data.table)

# Source block IRLS solver and helpers from estimator
source(file.path(base_dir, "R", "02_estimator.R"))


# ==============================================================================
# 1. GENERATE DATA
# ==============================================================================
cat("========================================\n")
cat("  Single-Sample Diagnostic\n")
cat("  Type-Specific Demographic Slopes\n")
cat("========================================\n\n")

cat("[1] Generating data...\n")
params      <- default_params()
params$J    <- 300L
params$T_hist <- 4L   # fewer years to reduce memory
params$size_range <- list(
  State       = c(50L, 150L),
  Academy     = c(40L, 120L),
  Independent = c(25L, 80L)
)
params$n_noise <- 10L

school_dt <- generate_school_structure(params)
verify_dgp(school_dt, params)

t0    <- proc.time()
panel <- generate_panel(school_dt, params, seed = 1L)
cat(sprintf("    Panel: %s rows x %d cols [%.1fs]\n",
            format(nrow(panel), big.mark = ","), ncol(panel),
            (proc.time() - t0)["elapsed"]))

# --- Extract estimation inputs ---
J         <- params$J
cov_names <- get_covariate_names(params)
p         <- length(cov_names)
sid       <- panel$school_id
d_covid   <- panel$D_covid
y         <- panel$Y

Z_base <- as.matrix(panel[, ..cov_names])
Z_base <- scale(Z_base, center = TRUE, scale = FALSE)
col_means <- attr(Z_base, "scaled:center")

rm(panel)
gc(verbose = FALSE)
cat(sprintf("    J = %d schools, p = %d covariates\n\n", J, p))


# ==============================================================================
# 2. ESTIMATOR A: Global Model (common slopes — misspecified)
# ==============================================================================
cat("[2] ESTIMATOR A: Global model (common slopes)\n")
t0    <- proc.time()
fit_A <- sparse_block_irls(sid, d_covid, Z_base, y, use_covid_cov = TRUE)
time_A <- (proc.time() - t0)["elapsed"]

se_A <- compute_block_se(sid, d_covid, Z_base, fit_A,
                         use_covid_cov = TRUE, ridge = 0)
eb_alpha_A <- numeric(J)
eb_delta_A <- numeric(J)
for (stype in c("State", "Academy", "Independent")) {
  idx <- which(school_dt$school_type == stype)
  eb_alpha_A[idx] <- eb_shrink(fit_A$alpha[idx], se_A$se_alpha[idx], tau2_floor = 0.02)
  eb_delta_A[idx] <- eb_shrink(fit_A$delta[idx], se_A$se_delta[idx], tau2_floor = 0.02)
}
cat(sprintf("    Time: %.1fs\n\n", time_A))


# ==============================================================================
# 3. ESTIMATOR B: Split by Type
# ==============================================================================
cat("[3] ESTIMATOR B: Split by type\n")
t0 <- proc.time()
student_type <- school_dt$school_type[sid]

eb_alpha_B <- numeric(J)
eb_delta_B <- numeric(J)
fit_B_list <- list()

for (stype in c("State", "Academy", "Independent")) {
  cat(sprintf("    %s: ", stype))

  s_mask     <- which(student_type == stype)
  s_orig_ids <- sort(unique(sid[s_mask]))
  J_s        <- length(s_orig_ids)
  id_map     <- match(sid[s_mask], s_orig_ids)

  s_sid  <- id_map
  s_dcov <- d_covid[s_mask]
  s_y    <- y[s_mask]
  s_Z    <- Z_base[s_mask, , drop = FALSE]

  cat(sprintf("N=%s, J=%d  ", format(length(s_y), big.mark = ","), J_s))
  fit_s <- sparse_block_irls(s_sid, s_dcov, s_Z, s_y, use_covid_cov = TRUE)
  se_s  <- compute_block_se(s_sid, s_dcov, s_Z, fit_s,
                            use_covid_cov = TRUE, ridge = 0)
  eb_a <- eb_shrink(fit_s$alpha, se_s$se_alpha, tau2_floor = 0.02)
  eb_d <- eb_shrink(fit_s$delta, se_s$se_delta, tau2_floor = 0.02)

  eb_alpha_B[s_orig_ids] <- eb_a
  eb_delta_B[s_orig_ids] <- eb_d
  fit_B_list[[stype]] <- list(fit = fit_s, se = se_s,
                               eb_alpha = eb_a, eb_delta = eb_d,
                               school_ids = s_orig_ids)
}
time_B <- (proc.time() - t0)["elapsed"]
cat(sprintf("    Total time: %.1fs\n\n", time_B))


# ==============================================================================
# 4. ESTIMATOR C: LASSO on Demo × Type Interactions
# ==============================================================================
cat("[4] ESTIMATOR C: LASSO on Demo x Type interactions\n")
t0 <- proc.time()

N <- length(y)
is_acad  <- as.numeric(student_type == "Academy")
is_indep <- as.numeric(student_type == "Independent")

# Demo × Type columns (centered scale)
demo_acad  <- Z_base[, 2:4, drop = FALSE] * is_acad
demo_indep <- Z_base[, 2:4, drop = FALSE] * is_indep

# Build glmnet design matrix
school_sp <- sparseMatrix(i = seq_len(N), j = sid, x = 1, dims = c(N, J))
covid_idx <- which(d_covid == 1L)
covid_school_sp <- sparseMatrix(i = covid_idx, j = sid[covid_idx],
                                x = 1, dims = c(N, J))

n_core <- 4L
n_type <- 6L
n_rest <- p - n_core

Z_core       <- Z_base[, 1:4, drop = FALSE]
Z_core_covid <- d_covid * Z_core
Z_rest       <- Z_base[, 5:p, drop = FALSE]
Z_rest_covid <- d_covid * Z_rest
Z_type_int   <- cbind(demo_acad, demo_indep)

X_C <- cbind(
  school_sp, covid_school_sp,
  as(Z_core, "dgCMatrix"),
  as(Z_core_covid, "dgCMatrix"),
  as(Z_type_int, "dgCMatrix"),
  as(Z_rest, "dgCMatrix"),
  as(Z_rest_covid, "dgCMatrix")
)

rm(school_sp, covid_school_sp, Z_core, Z_core_covid,
   Z_rest, Z_rest_covid, Z_type_int, demo_acad, demo_indep)
gc(verbose = FALSE)

pf_C <- c(
  rep(0, 2L * J),     # school FEs + COVID × school FEs
  rep(0, n_core),      # core base (GCSE, SES, MIN, GEN)
  rep(0, n_core),      # COVID × core
  rep(0, n_type),      # Demo × Type interactions (unpenalized — structural)
  rep(1, n_rest),      # poly + noise base
  rep(1, n_rest)       # COVID × poly/noise
)

cat(sprintf("    X: %s x %d, penalized: %d/%d cov cols\n",
            format(N, big.mark = ","), ncol(X_C),
            sum(pf_C > 0), length(pf_C) - 2L * J))

# LASSO fit (AIC selection — less conservative than BIC for large N)
cat("    Fitting glmnet (AIC selection)...\n")
glm_fit_C <- glmnet(
  x = X_C, y = y, family = "binomial",
  alpha = 1, penalty.factor = pf_C, intercept = FALSE,
  standardize = FALSE, thresh = 1e-7, maxit = 1e5
)

dev_C <- deviance(glm_fit_C)
df_C  <- glm_fit_C$df
aic_C <- dev_C + 2 * df_C   # AIC instead of BIC
best_C     <- which.min(aic_C)
lambda_aic <- glm_fit_C$lambda[best_C]

cf_C    <- coef(glm_fit_C, s = lambda_aic)
coefs_C <- as.numeric(cf_C)[-1]
cat(sprintf("    lambda(AIC) = %.6f, df = %d\n", lambda_aic, df_C[best_C]))

# Extract type interaction coefficients
idx_start <- 2L * J + 1L
gamma_type_C <- coefs_C[(idx_start + 2*n_core):(idx_start + 2*n_core + n_type - 1)]
type_names <- c("SES_Acad", "MIN_Acad", "GEN_Acad",
                "SES_Indep", "MIN_Indep", "GEN_Indep")
cat("    LASSO type interaction coefficients:\n")
for (i in seq_along(type_names)) {
  cat(sprintf("      %s: %.4f%s\n", type_names[i], gamma_type_C[i],
              if (gamma_type_C[i] != 0) " *" else ""))
}

# All covariate coefficients
all_cov_coefs <- coefs_C[(2L * J + 1):length(coefs_C)]
n_total_cov   <- length(all_cov_coefs)
selected_cov  <- which(all_cov_coefs != 0)
mandatory     <- 1:(2 * n_core + n_type)
keep_cov_idx  <- sort(unique(c(selected_cov, mandatory)))

cat(sprintf("    LASSO selected %d / %d covariate columns\n",
            length(selected_cov), n_total_cov))
cat(sprintf("    Post-LASSO keeping %d columns (incl. mandatory)\n",
            length(keep_cov_idx)))

# Post-LASSO: refit using exact block IRLS solver (not approximate ridge)
# Determine which rest columns (5:p in Z_base) LASSO selected
rest_base_start  <- 2L * n_core + n_type + 1L
rest_base_end    <- 2L * n_core + n_type + n_rest
rest_covid_start <- rest_base_end + 1L
rest_covid_end   <- n_total_cov

sel_rest_base  <- intersect(selected_cov, rest_base_start:rest_base_end) - rest_base_start + 1L
sel_rest_covid <- intersect(selected_cov, rest_covid_start:rest_covid_end) - rest_covid_start + 1L
sel_rest_local <- sort(unique(c(sel_rest_base, sel_rest_covid)))

cat(sprintf("    LASSO selected %d / %d rest columns\n",
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

rm(X_C, glm_fit_C, cf_C, coefs_C, type_int_mat)
gc(verbose = FALSE)

cat(sprintf("    Post-LASSO refit via block IRLS: %d covariate columns\n", p_post))
fit_C_post <- sparse_block_irls(sid, d_covid, Z_post, y, use_covid_cov = TRUE)

# SE and EB
se_C <- compute_block_se(sid, d_covid, Z_post, fit_C_post,
                         use_covid_cov = TRUE, ridge = 0)
alpha_C    <- fit_C_post$alpha
delta_C    <- fit_C_post$delta

# Type-specific EB shrinkage: shrink within each school type to avoid
# cross-type contamination (Independent alphas >> grand mean)
eb_alpha_C <- numeric(J)
eb_delta_C <- numeric(J)
for (stype in c("State", "Academy", "Independent")) {
  idx <- which(school_dt$school_type == stype)
  eb_alpha_C[idx] <- eb_shrink(alpha_C[idx], se_C$se_alpha[idx], tau2_floor = 0.02)
  eb_delta_C[idx] <- eb_shrink(delta_C[idx], se_C$se_delta[idx], tau2_floor = 0.02)
}

# Extract type interaction coefficients from post-LASSO
gamma_type_post <- fit_C_post$gamma_b[5:10]

time_C <- (proc.time() - t0)["elapsed"]
cat(sprintf("    Time: %.1fs\n\n", time_C))


# ==============================================================================
# 5. RESULTS TABLE
# ==============================================================================
cat("========================================\n")
cat("  RESULTS\n")
cat("========================================\n\n")

results <- data.table(
  school_id   = 1:J,
  school_type = school_dt$school_type,
  true_alpha  = school_dt$alpha_j,
  true_delta  = school_dt$delta_alpha_j,
  # Estimator A
  A_alpha = eb_alpha_A, A_delta = eb_delta_A,
  # Estimator B
  B_alpha = eb_alpha_B, B_delta = eb_delta_B,
  # Estimator C
  C_alpha = eb_alpha_C, C_delta = eb_delta_C
)


# ==============================================================================
# 6. GLOBAL REFERENCE STUDENT + TYPE-SPECIFIC SLOPES => RRR
# ==============================================================================
# Fixed student (global mean) for all school types. Composition differences
# are removed; only the LASSO-estimated type-specific slopes vary.
cat("[5] Computing RRR with global reference student...\n")

stypes <- as.character(school_dt$school_type)

# Global mean student (raw and centered scales)
ref_raw      <- col_means                     # overall sample mean
ref_centered <- rep(0, p)                     # zero on centered scale

cat(sprintf("  Global reference student (raw): GCSE=%.3f, SES=%.3f, MIN=%.3f, GEN=%.3f\n",
            ref_raw[1], ref_raw[2], ref_raw[3], ref_raw[4]))

# Clamping: use worst-case (Independent) slopes for safety
P_BASE_CAP <- 0.85
alpha_max_all <- max(school_dt$alpha_j)
worst_et <- params$eta_type[["Independent"]]
non_gcse_shift <- worst_et["SES"]      * ref_raw[2] +
                  worst_et["Minority"] * ref_raw[3] +
                  worst_et["Gender"]   * ref_raw[4]
max_p <- plogis(alpha_max_all + params$eta_GCSE * ref_raw[1] + non_gcse_shift)

cat(sprintf("  Max p_base (unclamped): %.3f", max_p))
if (max_p > P_BASE_CAP) {
  clamp_fn <- function(z) {
    plogis(alpha_max_all + params$eta_GCSE * z + non_gcse_shift) - P_BASE_CAP
  }
  gcse_clamp <- uniroot(clamp_fn, interval = c(-5, ref_raw[1]), tol = 1e-8)$root
  cat(sprintf(" -> clamping GCSE from %.3f to %.3f", ref_raw[1], gcse_clamp))

  ref_raw[1] <- gcse_clamp
  ref_raw[5] <- gcse_clamp^2
  ref_raw[6] <- gcse_clamp * ref_raw[2]
  ref_raw[7] <- gcse_clamp * ref_raw[3]
  ref_raw[8] <- gcse_clamp * ref_raw[4]
  ref_centered <- ref_raw - col_means

  new_p <- plogis(alpha_max_all + params$eta_GCSE * gcse_clamp + non_gcse_shift)
  cat(sprintf(", new max: %.3f", new_p))
}
cat("\n")

# --- Compute reference shifts for each estimator ---
# Truth: type-specific slopes applied to FIXED student
r  <- ref_raw
rc <- ref_centered
true_base  <- numeric(J)
true_covid <- numeric(J)
for (stype in c("State", "Academy", "Independent")) {
  idx <- which(stypes == stype)
  et  <- params$eta_type[[stype]]

  true_base[idx] <- params$eta_GCSE * r[1] +
                    et["SES"]       * r[2] +
                    et["Minority"]  * r[3] +
                    et["Gender"]    * r[4]

  true_covid[idx] <- (params$eta_GCSE     + params$delta_eta["GCSE_std"]) * r[1] +
                     (et["SES"]           + params$delta_eta["SES"])      * r[2] +
                     (et["Minority"]      + params$delta_eta["Minority"]) * r[3] +
                     (et["Gender"]        + params$delta_eta["Gender"])   * r[4] +
                     school_dt$beta_ses_j[idx] * r[2] +
                     school_dt$beta_min_j[idx] * r[3]
}

# Estimator A (Global): common gamma, fixed student
A_base  <- rep(drop(rc %*% fit_A$gamma_b), J)
A_covid <- rep(drop(rc %*% (fit_A$gamma_b + fit_A$gamma_c)), J)

# Estimator B (Split): type-specific gamma, fixed student
B_base  <- numeric(J)
B_covid <- numeric(J)
for (stype in c("State", "Academy", "Independent")) {
  idx <- which(stypes == stype)
  f   <- fit_B_list[[stype]]$fit
  B_base[idx]  <- drop(rc %*% f$gamma_b)
  B_covid[idx] <- drop(rc %*% (f$gamma_b + f$gamma_c))
}

# Estimator C (Fix): fixed student, type interactions from LASSO
C_base  <- numeric(J)
C_covid <- numeric(J)
for (stype in c("State", "Academy", "Independent")) {
  idx <- which(stypes == stype)

  # Build reference profile in Z_post coordinates using FIXED student
  # but activating type interaction columns for this school type
  rc_post <- numeric(p_post)
  rc_post[1:4] <- rc[1:4]
  if (stype == "Academy") {
    rc_post[5:7] <- rc[2:4]       # Demo x Acad interactions
  } else if (stype == "Independent") {
    rc_post[8:10] <- rc[2:4]      # Demo x Indep interactions
  }
  if (length(sel_rest_zcols) > 0) {
    rc_post[11:p_post] <- rc[sel_rest_zcols]
  }

  C_base[idx]  <- drop(rc_post %*% fit_C_post$gamma_b)
  C_covid[idx] <- drop(rc_post %*% (fit_C_post$gamma_b + fit_C_post$gamma_c))
}

# --- Probabilities and RRR ---
results[, true_p0  := plogis(true_alpha + true_base)]
results[, true_p1  := plogis(true_alpha + true_delta + true_covid)]
results[, true_raw := true_p1 - true_p0]
results[, true_rrr := true_raw / (1 - true_p0)]

results[, A_p0  := plogis(A_alpha + A_base)]
results[, A_p1  := plogis(A_alpha + A_delta + A_covid)]
results[, A_raw := A_p1 - A_p0]
results[, A_rrr := A_raw / (1 - A_p0)]

results[, B_p0  := plogis(B_alpha + B_base)]
results[, B_p1  := plogis(B_alpha + B_delta + B_covid)]
results[, B_raw := B_p1 - B_p0]
results[, B_rrr := B_raw / (1 - B_p0)]

results[, C_p0  := plogis(C_alpha + C_base)]
results[, C_p1  := plogis(C_alpha + C_delta + C_covid)]
results[, C_raw := C_p1 - C_p0]
results[, C_rrr := C_raw / (1 - C_p0)]

# Binning
results[, alpha_pctl := frank(true_alpha) / .N]
results[, bin := cut(alpha_pctl, breaks = seq(0, 1, 0.05),
                     labels = FALSE, include.lowest = TRUE)]
results[, bin_center := (bin - 0.5) / 20]

# --- Summary ---
cat("\n\nMean inflation by school type:\n")
summ <- results[, .(
  True_RRR   = round(mean(true_rrr), 4),
  Global_RRR = round(mean(A_rrr), 4),
  Split_RRR  = round(mean(B_rrr), 4),
  Fix_RRR    = round(mean(C_rrr), 4)
), by = school_type]
print(summ)

cat("\nCorrelation with truth:\n")
cat(sprintf("  Raw: Global %.3f | Split %.3f | Fix %.3f\n",
            cor(results$A_raw, results$true_raw),
            cor(results$B_raw, results$true_raw),
            cor(results$C_raw, results$true_raw)))
cat(sprintf("  RRR: Global %.3f | Split %.3f | Fix %.3f\n\n",
            cor(results$A_rrr, results$true_rrr),
            cor(results$B_rrr, results$true_rrr),
            cor(results$C_rrr, results$true_rrr)))

cat("Denominator bias:\n")
for (stype in c("State", "Academy", "Independent")) {
  mask <- results$school_type == stype
  db_A <- mean((1 - results$A_p0[mask]) - (1 - results$true_p0[mask]))
  db_B <- mean((1 - results$B_p0[mask]) - (1 - results$true_p0[mask]))
  db_C <- mean((1 - results$C_p0[mask]) - (1 - results$true_p0[mask]))
  cat(sprintf("  %s: Global %.5f | Split %.5f | Fix %.5f\n",
              stype, db_A, db_B, db_C))
}

cat("\nGCSE slope estimates:\n")
cat(sprintf("  True: %.3f\n", params$eta_GCSE))
cat(sprintf("  Global (A): %.3f\n", fit_A$gamma_b[1]))
for (stype in c("State", "Academy", "Independent")) {
  cat(sprintf("  Split (B, %s): %.3f\n", stype,
              fit_B_list[[stype]]$fit$gamma_b[1]))
}
cat(sprintf("  Fix (C): %.3f\n", fit_C_post$gamma_b[1]))

cat("\nType interaction estimates vs truth:\n")
state_slopes <- params$eta_type[["State"]]
for (tp in c("Academy", "Independent")) {
  true_devs <- params$eta_type[[tp]] - state_slopes
  cat(sprintf("  %s deviations:\n", tp))
  if (tp == "Academy") {
    est <- gamma_type_post[1:3]
  } else {
    est <- gamma_type_post[4:6]
  }
  cat(sprintf("    SES:      true=%+.3f, est=%+.3f\n", true_devs["SES"], est[1]))
  cat(sprintf("    Minority: true=%+.3f, est=%+.3f\n", true_devs["Minority"], est[2]))
  cat(sprintf("    Gender:   true=%+.3f, est=%+.3f\n", true_devs["Gender"], est[3]))
}


# ==============================================================================
# 7. PLOTS
# ==============================================================================
cat("\n[6] Generating plots...\n")

fig_dir <- file.path(Sys.getenv("HOME"), "A-Levels", "school_type_fix",
                     "output", "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

model_colors <- c(True   = "#D55E00",   # vermillion
                  Global = "#009E73",    # green
                  Split  = "#CC79A7",    # pink
                  Fix    = "#0072B2")    # blue
model_labels <- c(True   = "True (DGP)",
                  Global = "A: Global (misspecified)",
                  Split  = "B: Split by Type",
                  Fix    = "C: LASSO Demo x Type (Fix)")
model_shapes <- c(True = 16, Global = 17, Split = 15, Fix = 18)

agg_with_se <- function(dt, value_col) {
  dt[, .(mean_val = mean(get(value_col), na.rm = TRUE),
         se_val   = sd(get(value_col), na.rm = TRUE) / sqrt(.N)),
     by = .(bin_center, school_type, Model)]
}

make_plot <- function(long_dt, value_col, title, subtitle, ylab,
                      use_percent = FALSE, colors = model_colors,
                      labels = model_labels, shapes = model_shapes) {
  plot_dt <- agg_with_se(long_dt, value_col)
  p <- ggplot(plot_dt, aes(x = bin_center, y = mean_val,
                            color = Model, shape = Model)) +
    geom_ribbon(aes(ymin = mean_val - 1.96 * se_val,
                    ymax = mean_val + 1.96 * se_val,
                    fill = Model), alpha = 0.12, colour = NA) +
    geom_line(aes(linetype = Model), linewidth = 0.9,
              position = position_dodge(width = 0.01)) +
    geom_point(size = 2, position = position_dodge(width = 0.01)) +
    facet_wrap(~school_type) +
    scale_color_manual(values = colors, labels = labels) +
    scale_fill_manual(values = colors, labels = labels) +
    scale_shape_manual(values = shapes, labels = labels) +
    scale_linetype_manual(
      values = setNames(c("solid", "dashed", "dotdash", "solid"),
                        names(colors)),
      labels = labels
    ) +
    labs(title = title, subtitle = subtitle,
         x = "School Quality Percentile", y = ylab) +
    guides(color = guide_legend(NULL), fill = guide_legend(NULL),
           shape = guide_legend(NULL), linetype = guide_legend(NULL)) +
    theme_minimal(base_size = 12) +
    theme(
      legend.position = "bottom",
      strip.text      = element_text(face = "bold", size = 12),
      legend.text     = element_text(size = 10),
      plot.subtitle   = element_text(size = 10)
    )
  if (use_percent) p <- p + scale_y_continuous(labels = scales::percent_format())
  p
}


# --- FIGURE 1: Raw Probability Increase ---
long_raw <- melt(
  results,
  id.vars      = c("bin_center", "school_type"),
  measure.vars = c("true_raw", "A_raw", "B_raw", "C_raw"),
  variable.name = "Model",
  value.name    = "prob_increase"
)
long_raw[, Model := factor(
  sub("_raw$", "", Model),
  levels = c("true", "A", "B", "C"),
  labels = c("True", "Global", "Split", "Fix")
)]

plt1 <- make_plot(
  long_raw, "prob_increase",
  "COVID Grade Inflation: Raw Probability Increase",
  "Inflation = P_covid - P_baseline  |  Sector-specific reference profiles",
  expression(P[covid] - P[baseline])
)
out1 <- file.path(fig_dir, "inflation_probability_increase.pdf")
ggsave(out1, plt1, width = 12, height = 6)
cat(sprintf("  Figure 1 saved: %s\n", out1))


# --- FIGURE 2: RRR ---
long_rrr <- melt(
  results,
  id.vars      = c("bin_center", "school_type"),
  measure.vars = c("true_rrr", "A_rrr", "B_rrr", "C_rrr"),
  variable.name = "Model",
  value.name    = "rrr"
)
long_rrr[, Model := factor(
  sub("_rrr$", "", Model),
  levels = c("true", "A", "B", "C"),
  labels = c("True", "Global", "Split", "Fix")
)]

plt2 <- make_plot(
  long_rrr, "rrr",
  "Relative Risk Reduction (Sector-Specific Reference Profiles)",
  "RRR = (P_covid - P_base) / (1 - P_base)  |  Each sector uses its own mean student profile",
  expression(frac(P[covid] - P[baseline], 1 - P[baseline])),
  use_percent = TRUE
)
out2 <- file.path(fig_dir, "RRR_sector_specific.pdf")
ggsave(out2, plt2, width = 12, height = 6)
cat(sprintf("  Figure 2 saved: %s\n", out2))


# --- FIGURE 3: Baseline Probabilities (p0) ---
long_p0 <- melt(
  results,
  id.vars      = c("bin_center", "school_type"),
  measure.vars = c("true_p0", "A_p0", "B_p0", "C_p0"),
  variable.name = "Model",
  value.name    = "p0"
)
long_p0[, Model := factor(
  sub("_p0$", "", Model),
  levels = c("true", "A", "B", "C"),
  labels = c("True", "Global", "Split", "Fix")
)]

plt3 <- make_plot(
  long_p0, "p0",
  "Baseline Probability P(Y=1 | no COVID)",
  "Shows denominator (1 - p0) accuracy — key driver of RRR bias",
  expression(P[baseline]),
  use_percent = TRUE
)
out3 <- file.path(fig_dir, "baseline_probability.pdf")
ggsave(out3, plt3, width = 12, height = 6)
cat(sprintf("  Figure 3 saved: %s\n", out3))


# --- FIGURE 4: COVID Probabilities (p1) ---
long_p1 <- melt(
  results,
  id.vars      = c("bin_center", "school_type"),
  measure.vars = c("true_p1", "A_p1", "B_p1", "C_p1"),
  variable.name = "Model",
  value.name    = "p1"
)
long_p1[, Model := factor(
  sub("_p1$", "", Model),
  levels = c("true", "A", "B", "C"),
  labels = c("True", "Global", "Split", "Fix")
)]

plt4 <- make_plot(
  long_p1, "p1",
  "COVID Probability P(Y=1 | COVID)",
  "Should be similar across models if alpha_hat + delta_hat is well estimated",
  expression(P[covid]),
  use_percent = TRUE
)
out4 <- file.path(fig_dir, "covid_probability.pdf")
ggsave(out4, plt4, width = 12, height = 6)
cat(sprintf("  Figure 4 saved: %s\n", out4))


# --- FIGURE 5: Alpha Recovery ---
long_alpha <- melt(
  results,
  id.vars      = c("bin_center", "school_type"),
  measure.vars = c("true_alpha", "A_alpha", "B_alpha", "C_alpha"),
  variable.name = "Model",
  value.name    = "alpha"
)
long_alpha[, Model := factor(
  sub("_alpha$", "", Model),
  levels = c("true", "A", "B", "C"),
  labels = c("True", "Global", "Split", "Fix")
)]

plt5 <- make_plot(
  long_alpha, "alpha",
  "School Quality Recovery (alpha_j)",
  "EB-shrunk estimates vs truth — Global absorbs type-specific slopes into alpha",
  expression(alpha[j])
)
out5 <- file.path(fig_dir, "alpha_recovery.pdf")
ggsave(out5, plt5, width = 12, height = 6)
cat(sprintf("  Figure 5 saved: %s\n", out5))


# --- FIGURE 6: Delta Recovery ---
long_delta <- melt(
  results,
  id.vars      = c("bin_center", "school_type"),
  measure.vars = c("true_delta", "A_delta", "B_delta", "C_delta"),
  variable.name = "Model",
  value.name    = "delta"
)
long_delta[, Model := factor(
  sub("_delta$", "", Model),
  levels = c("true", "A", "B", "C"),
  labels = c("True", "Global", "Split", "Fix")
)]

plt6 <- make_plot(
  long_delta, "delta",
  "COVID Inflation Recovery (delta_j)",
  "EB-shrunk estimates vs truth",
  expression(delta[j])
)
out6 <- file.path(fig_dir, "delta_recovery.pdf")
ggsave(out6, plt6, width = 12, height = 6)
cat(sprintf("  Figure 6 saved: %s\n", out6))


# --- Save results ---
res_dir <- file.path(Sys.getenv("HOME"), "A-Levels", "school_type_fix",
                     "output", "results")
dir.create(res_dir, recursive = TRUE, showWarnings = FALSE)
saveRDS(results, file.path(res_dir, "single_sample_results.rds"))
saveRDS(school_dt, file.path(res_dir, "school_structure.rds"))

cat("\nDone!\n")
