# Perceptoner statistical validation using R

project_dir <- "D:/Esoft/Sem 03/Advanced Programming for Data analytics/Perceptoner_Project"

processed_dir <- file.path(project_dir, "data", "processed")
output_dir <- file.path(project_dir, "outputs", "r_analysis")

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

sales_file <- file.path(processed_dir, "cleaned_sales.csv")
telemetry_file <- file.path(processed_dir, "cleaned_telemetry.csv")

if (!file.exists(sales_file) || !file.exists(telemetry_file)) {
  stop("Processed files were not found. Run data_cleaning.py first.")
}

library(dplyr)
library(ggplot2)

sales <- read.csv(sales_file, check.names = FALSE)
telemetry <- read.csv(telemetry_file, check.names = FALSE)

# Statistical summary of flight conditions by flight status
flight_status_summary <- telemetry %>%
  group_by(`Flight Status`) %>%
  summarise(
    flight_count = n(),
    average_battery = round(mean(`Battery Remaining (%)`, na.rm = TRUE), 2),
    average_wind_speed = round(mean(`Wind Speed (m/s)`, na.rm = TRUE), 2),
    average_payload_ratio = round(mean(payload_ratio, na.rm = TRUE), 2),
    .groups = "drop"
  )

write.csv(
  flight_status_summary,
  file.path(output_dir, "r_statistics_summary.csv"),
  row.names = FALSE
)

# Battery comparison chart by flight status
battery_plot <- ggplot(
  telemetry,
  aes(
    x = `Flight Status`,
    y = `Battery Remaining (%)`,
    fill = `Flight Status`
  )
) +
  geom_boxplot() +
  labs(
    title = "Battery Remaining by Flight Status",
    x = "Flight Status",
    y = "Battery Remaining (%)"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

ggsave(
  filename = file.path(output_dir, "battery_by_flight_status.png"),
  plot = battery_plot,
  width = 9,
  height = 5,
  dpi = 300
)

# Operational-risk distribution chart
risk_plot <- ggplot(
  telemetry,
  aes(x = risk_level, fill = risk_level)
) +
  geom_bar() +
  labs(
    title = "Operational Risk Level Distribution",
    x = "Risk Level",
    y = "Number of Flights"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

ggsave(
  filename = file.path(output_dir, "risk_level_distribution.png"),
  plot = risk_plot,
  width = 9,
  height = 5,
  dpi = 300
)

cat("\nR statistical validation completed successfully.\n")
cat("Output folder:", output_dir, "\n")