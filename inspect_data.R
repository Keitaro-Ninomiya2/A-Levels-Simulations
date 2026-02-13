# inspect_data.R
library(data.table)

# Define paths based on your tree structure
results_dir <- "output/results"
structure_file <- file.path(results_dir, "school_structure.rds")
# Find all simulation result files
sim_files <- list.files(results_dir, pattern = "sim_results_.*\\.rds", full.names = TRUE)

# --- 1. Inspect the "Truth" (School Structure) ---
if (file.exists(structure_file)) {
  cat("\n==============================================\n")
  cat("=== SCHOOL STRUCTURE (Ground Truth) ===\n")
  cat("==============================================\n")
  
  schools <- readRDS(structure_file)
  
  # Print structure
  cat("Dimensions:", dim(schools), "\n")
  print(str(schools))
  
  cat("\n--- Head of Data ---\n")
  print(head(schools))
  
  cat("\n--- School Type Counts ---\n")
  if("school_type" %in% names(schools)) {
    print(table(schools$school_type))
  }
} else {
  cat("Warning: school_structure.rds not found.\n")
}

# --- 2. Inspect Simulation Results ---
if (length(sim_files) > 0) {
  cat("\n==============================================\n")
  cat("=== SIMULATION RESULTS (Found", length(sim_files), "files) ===\n")
  cat("==============================================\n")
  
  # Read the first file to see what columns we have
  first_res <- readRDS(sim_files[1])
  
  cat("File inspected:", sim_files[1], "\n")
  cat("Dimensions:", dim(first_res), "\n")
  
  cat("\n--- Column Names ---\n")
  print(colnames(first_res))
  
  cat("\n--- Head of Data ---\n")
  print(head(first_res))
  
  # Check for NAs in key columns
  cat("\n--- Summary of Estimates ---\n")
  print(summary(first_res))
  
} else {
  cat("Warning: No sim_results files found in", results_dir, "\n")
}
