# ==============================================================================
# 04_analysis.R — Plotting for Type-Specific Slopes Simulation
# ==============================================================================
# Loads MC replication files, computes binned means, and produces
# comparison plots: True vs Global vs Split vs Fix, faceted by school type.
# ==============================================================================

.libPaths(c(file.path(Sys.getenv("HOME"), "A-Levels", "R_libs"), .libPaths()))
library(data.table)
library(ggplot2)

# --- 1. Load Paths ---
results_dir <- file.path(Sys.getenv("HOME"), "A-Levels", "school_type_fix",
                         "output", "results")
fig_dir     <- file.path(Sys.getenv("HOME"), "A-Levels", "school_type_fix",
                         "output", "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

# --- 2. Load Simulation Results ---
sim_files <- list.files(results_dir, pattern = "rep_.*\\.rds", full.names = TRUE)
if (length(sim_files) == 0)
  stop("No result files (rep_XXX.rds) found in ", results_dir)

cat(sprintf("Found %d replication files. Loading and merging...\n", length(sim_files)))
res_dt <- rbindlist(lapply(sim_files, readRDS), fill = TRUE)

# --- 3. Verify columns ---
expected_cols <- c("school_id", "school_type", "bin_center",
                   "true_raw", "A_raw", "B_raw", "C_raw",
                   "true_rrr", "A_rrr", "B_rrr", "C_rrr")
missing <- setdiff(expected_cols, names(res_dt))
if (length(missing) > 0)
  stop("Missing columns: ", paste(missing, collapse = ", "))

# --- 4. Plotting Setup ---
model_colors <- c(True   = "#D55E00",   # vermillion
                  Global = "#009E73",    # green
                  Split  = "#CC79A7",    # pink
                  Fix    = "#0072B2")    # blue
model_labels <- c(True   = "True (DGP)",
                  Global = "A: Global (common slopes)",
                  Split  = "B: Split by Type",
                  Fix    = "C: LASSO Demo x Type (Fix)")
model_shapes <- c(True = 16, Global = 17, Split = 15, Fix = 18)

agg_with_se <- function(dt, value_col) {
  dt[, .(mean_val = mean(get(value_col), na.rm = TRUE),
         se_val   = sd(get(value_col), na.rm = TRUE) / sqrt(.N)),
     by = .(bin_center, school_type, Model)]
}

make_plot <- function(long_dt, value_col, title, subtitle, ylab,
                      use_percent = FALSE) {
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
    scale_color_manual(values = model_colors, labels = model_labels) +
    scale_fill_manual(values = model_colors, labels = model_labels) +
    scale_shape_manual(values = model_shapes, labels = model_labels) +
    scale_linetype_manual(
      values = c(True = "solid", Global = "dashed",
                 Split = "dotdash", Fix = "solid"),
      labels = model_labels
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


# --- 5. FIGURE 1: Raw Probability Increase ---
long_raw <- melt(
  res_dt,
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

plt_raw <- make_plot(
  long_raw, "prob_increase",
  title    = "COVID Grade Inflation: Raw Probability Increase",
  subtitle = "Inflation = P_covid - P_baseline  |  Sector-specific reference profiles",
  ylab     = expression(P[covid] - P[baseline])
)

out_raw <- file.path(fig_dir, "inflation_probability_increase.pdf")
ggsave(out_raw, plt_raw, width = 12, height = 6)
cat(sprintf("  Figure 1 saved: %s\n", out_raw))


# --- 6. FIGURE 2: RRR ---
long_rrr <- melt(
  res_dt,
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

plt_rrr <- make_plot(
  long_rrr, "rrr",
  title    = "Relative Risk Reduction (Sector-Specific Reference Profiles)",
  subtitle = "RRR = (P_covid - P_base) / (1 - P_base)  |  Each sector uses its own mean student profile",
  ylab     = expression(frac(P[covid] - P[baseline], 1 - P[baseline])),
  use_percent = TRUE
)

out_rrr <- file.path(fig_dir, "RRR_sector_specific.pdf")
ggsave(out_rrr, plt_rrr, width = 12, height = 6)
cat(sprintf("  Figure 2 saved: %s\n", out_rrr))


# --- 7. Summary Statistics ---
cat("\n\nMean RRR by school type (averaged across replications):\n")
summ <- res_dt[, .(
  True_RRR   = round(mean(true_rrr, na.rm = TRUE), 4),
  Global_RRR = round(mean(A_rrr, na.rm = TRUE), 4),
  Split_RRR  = round(mean(B_rrr, na.rm = TRUE), 4),
  Fix_RRR    = round(mean(C_rrr, na.rm = TRUE), 4)
), by = school_type]
print(summ)

cat("\nCorrelation with truth:\n")
cat(sprintf("  Raw: Global %.3f | Split %.3f | Fix %.3f\n",
            cor(res_dt$A_raw, res_dt$true_raw, use = "complete.obs"),
            cor(res_dt$B_raw, res_dt$true_raw, use = "complete.obs"),
            cor(res_dt$C_raw, res_dt$true_raw, use = "complete.obs")))
cat(sprintf("  RRR: Global %.3f | Split %.3f | Fix %.3f\n",
            cor(res_dt$A_rrr, res_dt$true_rrr, use = "complete.obs"),
            cor(res_dt$B_rrr, res_dt$true_rrr, use = "complete.obs"),
            cor(res_dt$C_rrr, res_dt$true_rrr, use = "complete.obs")))

cat("\nDenominator bias by type:\n")
for (stype in c("State", "Academy", "Independent")) {
  mask <- res_dt$school_type == stype
  db_A <- mean((1 - res_dt$A_p0[mask]) - (1 - res_dt$true_p0[mask]), na.rm = TRUE)
  db_B <- mean((1 - res_dt$B_p0[mask]) - (1 - res_dt$true_p0[mask]), na.rm = TRUE)
  db_C <- mean((1 - res_dt$C_p0[mask]) - (1 - res_dt$true_p0[mask]), na.rm = TRUE)
  cat(sprintf("  %s: Global %.5f | Split %.5f | Fix %.5f\n",
              stype, db_A, db_B, db_C))
}

cat("\nDone!\n")
