# ==============================================================================
# 03_simulation_runner.R — Monte Carlo Runner (5-Rep)
# ==============================================================================
# Runs Global / Split-by-Type / LASSO-Fix estimators on replicated panels
# from the type-specific demographic slopes DGP.
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

# --- CONFIGURATION ---
N_REPS  <- 5
N_CORES <- 2

cat(sprintf("Starting Monte Carlo Simulation (Type-Specific Slopes)\n"))
cat(sprintf("Cores: %d | Total Reps: %d\n", N_CORES, N_REPS))

# 1. Clean old results
old_reps <- list.files(OUT_DIR, pattern = "rep_.*\\.rds", full.names = TRUE)
if (length(old_reps) > 0) file.remove(old_reps)

# 2. Generate/Save school structure
school_dt <- generate_school_structure(params, seed = 42L)
saveRDS(school_dt, file.path(OUT_DIR, "school_structure.rds"))
verify_dgp(school_dt, params)

# 3. Worker Function
run_one <- function(r) {
  suppressMessages(library(data.table))
  suppressMessages(library(glmnet))
  suppressMessages(library(Matrix))

  cat(sprintf("\n--- Rep %d ---\n", r))

  panel <- generate_panel(school_dt, params, seed = r)
  res <- tryCatch({
    estimate_all(panel, school_dt, params)
  }, error = function(e) {
    cat(sprintf("  ERROR in rep %d: %s\n", r, e$message))
    return(NULL)
  })

  if (!is.null(res)) {
    res[, rep_id := r]
    saveRDS(res, file.path(OUT_DIR, sprintf("rep_%03d.rds", r)))
    cat(sprintf("  Rep %d saved.\n", r))
  }

  rm(panel, res)
  gc(verbose = FALSE)
  return(r)
}

# 4. Execute
t0 <- proc.time()
mclapply(1:N_REPS, run_one, mc.cores = N_CORES, mc.preschedule = FALSE)
elapsed <- (proc.time() - t0)["elapsed"]

cat(sprintf("\nDone! Total time: %.1f minutes\n", elapsed / 60))
