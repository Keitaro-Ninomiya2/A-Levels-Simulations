# ==============================================================================
# 04_analysis.R — Robust Plotter for Individual Rep Files
# ==============================================================================
.libPaths(c(file.path(Sys.getenv("HOME"), "A-Levels", "R_libs"), .libPaths()))
library(data.table)
library(ggplot2)

# --- 1. Load Paths ---
results_dir <- file.path(Sys.getenv("HOME"), "A-Levels", "output", "results")
fig_dir     <- file.path(Sys.getenv("HOME"), "A-Levels", "output", "figures")
if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)

# --- 2. Load Simulation Results ---
# Find all the rep_XXX.rds files we just generated
sim_files <- list.files(results_dir, pattern = "rep_.*\\.rds", full.names = TRUE)
if (length(sim_files) == 0) stop("No result files (rep_XXX.rds) found in output/results/")

cat(sprintf("Found %d replication files. Loading and merging...\n", length(sim_files)))
res_dt <- rbindlist(lapply(sim_files, readRDS), fill = TRUE)

# --- 3. Merge Truth from School Structure ---
struct_file <- file.path(results_dir, "school_structure.rds")
if (!file.exists(struct_file)) stop("Critical: school_structure.rds not found!")
school_dt <- readRDS(struct_file)

# Recalculate alpha_percentile if missing or just to be safe
res_dt <- merge(res_dt[, .(school_id, unpen_est_delta, pen_est_delta, post_est_delta, rep_id)], 
                school_dt[, .(school_id, school_type, true_alpha = alpha_j, true_delta = delta_alpha_j)],
                by = "school_id")

# --- 4. Binning ---
res_dt[, alpha_percentile := frank(true_alpha, ties.method = "average") / .N, by = rep_id]
res_dt[, bin := cut(alpha_percentile, breaks = seq(0, 1, 0.05), labels = FALSE, include.lowest = TRUE)]
res_dt[, bin_center := (bin - 0.5) / 20]

# --- 5. Aggregation ---
# Melt to long format for ggplot
long_data <- melt(res_dt, 
                  id.vars = c("bin_center", "school_type"),
                  measure.vars = c("true_delta", "unpen_est_delta", "pen_est_delta", "post_est_delta"),
                  variable.name = "Model",
                  value.name = "Value")

# Average across replications
plot_data <- long_data[, .(
  Mean = mean(Value, na.rm = TRUE),
  SE   = sd(Value, na.rm = TRUE) / sqrt(.N)
), by = .(bin_center, school_type, Model)]

plot_data[, Lower := Mean - 1.96 * SE]
plot_data[, Upper := Mean + 1.96 * SE]

# --- 6. Plotting ---
p <- ggplot(plot_data, aes(x = bin_center, y = Mean, color = Model, group = Model)) +
  geom_errorbar(aes(ymin = Lower, ymax = Upper), width = 0.02, position = position_dodge(0.02), alpha = 0.4) +
  geom_line(linewidth = 1, position = position_dodge(0.02)) +
  geom_point(size = 1.5, position = position_dodge(0.02)) +
  facet_wrap(~school_type) +
  scale_color_manual(values = c(
    "true_delta"      = "#D55E00",   # Red
    "unpen_est_delta" = "grey70",    # Grey
    "pen_est_delta"   = "black",     # Black
    "post_est_delta"  = "#0072B2"    # Blue
  ), labels = c("True", "Unpenalized", "Penalized (LASSO)", "Post-LASSO")) +
  labs(title = "Post-LASSO Simulation Results (5 Reps)",
       subtitle = "Blue line should correctly track Red (Truth) better than Black (LASSO).",
       y = "Estimated Inflation (Log-Odds)",
       x = "School Quality Percentile") +
  theme_minimal() + theme(legend.position = "bottom")

# --- 7. Save ---
out_path <- file.path(fig_dir, "post_lasso_results.pdf")
ggsave(out_path, p, width = 10, height = 6)
cat("Success! Final plot saved to:", out_path, "\n")
