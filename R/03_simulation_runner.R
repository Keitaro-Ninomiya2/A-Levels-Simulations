# ==============================================================================
# 03_simulation_runner.R — Ultra-Fast 5-Rep Version
# ==============================================================================
.libPaths(c(file.path(Sys.getenv("HOME"), "A-Levels", "R_libs"), .libPaths()))
library(parallel)
library(data.table)

# Source components
source(file.path(Sys.getenv("HOME"), "A-Levels", "R", "01_dgp.R"))
source(file.path(Sys.getenv("HOME"), "A-Levels", "R", "02_estimator.R"))

params  <- default_params()
OUT_DIR <- file.path(Sys.getenv("HOME"), "A-Levels", "output", "results")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# --- CONFIGURATION ---
N_REPS  <- 5
N_CORES <- 5  # One core per rep for maximum speed

cat(sprintf("Starting Monte Carlo Simulation\nCores: %d | Total Reps: %d\n", N_CORES, N_REPS))

# 1. Clean old results to avoid confusion
old_reps <- list.files(OUT_DIR, pattern = "rep_.*\\.rds", full.names = TRUE)
if(length(old_reps) > 0) file.remove(old_reps)

# 2. Generate/Save school structure
school_dt <- generate_school_structure(params, seed = 42L)
saveRDS(school_dt, file.path(OUT_DIR, "school_structure.rds"))

# 3. Worker Function
run_one <- function(r) {
  suppressMessages(library(data.table))
  suppressMessages(library(glmnet))
  
  panel <- generate_panel(school_dt, params, seed = r)
  res <- tryCatch({
    estimate_all(panel, school_dt, params)
  }, error = function(e) return(NULL))
  
  if(!is.null(res)) {
    res[, rep_id := r]
    saveRDS(res, file.path(OUT_DIR, sprintf("rep_%03d.rds", r)))
  }
  
  rm(panel, res)
  gc()
  return(r)
}

# 4. Execute (mc.preschedule = FALSE to handle stragglers)
t0 <- proc.time()
mclapply(1:N_REPS, run_one, mc.cores = N_CORES, mc.preschedule = FALSE)
elapsed <- (proc.time() - t0)["elapsed"]

cat(sprintf("\nDone! Total time: %.1f minutes\n", elapsed / 60))
