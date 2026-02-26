# ==============================================================================
# 03_simulation_runner.R — Monte Carlo Runner
# ==============================================================================
# Runs estimators on replicated panels from the type-specific DGP.
#
# With --split-only: Only Split-by-Type (B) — avoids extrapolation bias.
# Without:          Global (A) + Split (B) + LASSO-Fix (C).
#
# Reports:
#   1. Per-rep RRR results (saved as rep_XXX.rds)
#   2. Monte Carlo distribution of delta hyperparameters (mu_delta, tau2_delta)
#      vs DGP truth — overall and by school type
# ==============================================================================

.libPaths(c(file.path(Sys.getenv("HOME"), "A-Levels", "R_libs"), .libPaths()))
library(parallel)
library(data.table)

# Source components
base_dir <- file.path(Sys.getenv("HOME"), "A-Levels", "school_type_fix", "Post_LASSO")
source(file.path(base_dir, "R", "01_dgp.R"))
source(file.path(base_dir, "R", "02_estimator.R"))

params  <- default_params()
OUT_DIR <- file.path(Sys.getenv("HOME"), "A-Levels", "school_type_fix",
                     "output", "results")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# --- CONFIGURATION (overridable via --reps N --cores N --split-only) ---
args_raw <- commandArgs(trailingOnly = TRUE)
parse_arg <- function(flag, default) {
  i <- match(flag, args_raw)
  if (!is.na(i) && i < length(args_raw)) as.integer(args_raw[i + 1L]) else default
}
parse_flag <- function(flag) {
  flag %in% args_raw
}
N_REPS     <- parse_arg("--reps",  50L)
N_CORES    <- parse_arg("--cores",  1L)
SPLIT_ONLY <- parse_flag("--split-only")

cat(sprintf("Starting Monte Carlo Simulation (Type-Specific Slopes)\n"))
cat(sprintf("Cores: %d | Total Reps: %d | Split-only (B): %s\n",
            N_CORES, N_REPS, if (SPLIT_ONLY) "YES" else "NO"))

# 1. Report any existing rep files (checkpoint resume — do NOT delete them)
old_reps <- list.files(OUT_DIR, pattern = "rep_.*\\.rds", full.names = TRUE)
if (length(old_reps) > 0) {
  cat(sprintf("Found %d existing rep file(s) — will skip those reps (checkpoint resume).\n",
              length(old_reps)))
}

# 2. Generate/Save school structure
school_dt <- generate_school_structure(params, seed = 42L)
saveRDS(school_dt, file.path(OUT_DIR, "school_structure.rds"))
verify_dgp(school_dt, params)

# 3. Compute TRUE hyperparameters from the DGP
true_mu_delta    <- mean(school_dt$delta_alpha_j)
true_tau2_delta  <- var(school_dt$delta_alpha_j)
true_hp <- data.table(group = "Overall",
                      mu_delta = true_mu_delta,
                      tau2_delta = true_tau2_delta)
for (stype in c("State", "Academy", "Independent")) {
  idx <- which(school_dt$school_type == stype)
  true_hp <- rbind(true_hp, data.table(
    group      = stype,
    mu_delta   = mean(school_dt$delta_alpha_j[idx]),
    tau2_delta = var(school_dt$delta_alpha_j[idx])
  ))
}

cat("\nTrue delta hyperparameters (from DGP):\n")
print(true_hp)
cat("\n")

# 4. Worker Function
run_one <- function(r) {
  rep_file <- file.path(OUT_DIR, sprintf("rep_%03d.rds", r))

  if (file.exists(rep_file)) {
    saved <- tryCatch(readRDS(rep_file), error = function(e) NULL)
    if (is.list(saved) && !is.null(saved$hyperparams)) {
      cat(sprintf("  Rep %d: loaded from checkpoint.\n", r))
      return(saved)
    }
  }

  suppressMessages(library(data.table))
  suppressMessages(library(glmnet))
  suppressMessages(library(Matrix))

  cat(sprintf("\n--- Rep %d ---\n", r))

  panel <- generate_panel(school_dt, params, seed = r)
  out <- tryCatch({
    estimate_all(panel, school_dt, params, split_only = SPLIT_ONLY)
  }, error = function(e) {
    cat(sprintf("  ERROR in rep %d: %s\n", r, e$message))
    return(NULL)
  })

  if (is.list(out) && !is.null(out$results)) {
    out$results[, rep_id := r]
    saveRDS(out, rep_file)
    cat(sprintf("  Rep %d saved.\n", r))
  }

  rm(panel)
  gc(verbose = FALSE)
  return(out)
}

# 5. Execute
t0 <- proc.time()
all_results <- mclapply(1:N_REPS, run_one, mc.cores = N_CORES,
                        mc.preschedule = FALSE)
elapsed <- (proc.time() - t0)["elapsed"]
cat(sprintf("\nSimulation complete! Total time: %.1f minutes\n", elapsed / 60))


# ==============================================================================
# 6. MONTE CARLO REPORT: Delta Hyperparameter Distributions
# ==============================================================================
cat("\n")
cat("====================================================================\n")
cat("  MONTE CARLO REPORT: Delta Hyperparameter Distributions\n")
if (SPLIT_ONLY) cat("  (Split-by-Type estimation only)\n")
cat("  MAE = mean |est - true|, RMSE = sqrt(mean (est-true)^2) over schools\n")
cat("====================================================================\n\n")

# Collect hyperparameters across replications
# If a rep is missing from all_results (e.g. mclapply fork issue), fall back to disk
hp_list <- list()
for (r in seq_len(N_REPS)) {
  res_r <- all_results[[r]]
  if (!is.list(res_r) || is.null(res_r$hyperparams)) {
    rep_file <- file.path(OUT_DIR, sprintf("rep_%03d.rds", r))
    if (file.exists(rep_file)) {
      res_r <- tryCatch(readRDS(rep_file), error = function(e) NULL)
    }
  }
  if (is.list(res_r) && !is.null(res_r$hyperparams)) {
    hp_r <- copy(res_r$hyperparams)
    hp_r[, rep_id := r]
    hp_list[[length(hp_list) + 1L]] <- hp_r
  }
}
hp_all <- rbindlist(hp_list)
n_valid <- length(hp_list)

cat(sprintf("Valid replications: %d / %d\n\n", n_valid, N_REPS))

# --- Report by group and estimator ---
report_ests <- if (SPLIT_ONLY) "B_Split" else c("A_Global", "B_Split", "C_Fix")
for (grp in c("Overall", "State", "Academy", "Independent")) {
  cat(sprintf("--- %s ---\n", grp))

  # True values
  true_row <- true_hp[group == grp]
  cat(sprintf("  True:  mu_delta = %.4f,  tau2_delta = %.4f\n",
              true_row$mu_delta, true_row$tau2_delta))

  for (est in report_ests) {
    sub <- hp_all[group == grp & estimator == est]
    if (nrow(sub) == 0) next

    mu_mean   <- mean(sub$mu_delta)
    mu_sd     <- sd(sub$mu_delta)
    mu_bias   <- mu_mean - true_row$mu_delta
    tau2_mean <- mean(sub$tau2_delta)
    tau2_sd   <- sd(sub$tau2_delta)
    tau2_bias <- tau2_mean - true_row$tau2_delta
    cat(sprintf("  %s:\n", est))
    cat(sprintf("    mu_delta:   mean=%.4f  sd=%.4f  bias=%+.4f\n",
                mu_mean, mu_sd, mu_bias))
    cat(sprintf("    tau2_delta: mean=%.4f  sd=%.4f  bias=%+.4f\n",
                tau2_mean, tau2_sd, tau2_bias))
    if ("mae_delta" %in% names(sub)) {
      mae_mean  <- mean(sub$mae_delta)
      mae_sd    <- sd(sub$mae_delta)
      rmse_mean <- mean(sub$rmse_delta)
      rmse_sd   <- sd(sub$rmse_delta)
      cat(sprintf("    MAE (school):  mean=%.4f  sd=%.4f  (mean |est-true|)\n",
                  mae_mean, mae_sd))
      cat(sprintf("    RMSE (school): mean=%.4f  sd=%.4f  (sqrt mean (est-true)^2)\n",
                  rmse_mean, rmse_sd))
    }
  }
  cat("\n")
}

# --- Save hyperparameter results ---
saveRDS(hp_all, file.path(OUT_DIR, "mc_hyperparams.rds"))
saveRDS(true_hp, file.path(OUT_DIR, "true_hyperparams.rds"))
cat(sprintf("Hyperparameter results saved to %s\n", OUT_DIR))

cat("\nDone!\n")
