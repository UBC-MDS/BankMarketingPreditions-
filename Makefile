#Makefile


# Variables
RAW_DATA=data/bank-additional-full.csv
CLEANED_DATA=data/cleaned_bank_data.csv
EDA_PLOTS=results/data_info.csv \
          results/decision_tree_high_res.png \
          results/eda_age_dist.png \
          results/eda_campaign_dist.png \
          results/eda_correlation.png \
          results/eda_duration_dist.png \
          results/eda_previous_dist.png \
          results/eda_scatterplot.png \
          results/eda_target_dist.png
REPORT=reports/bank_marketing_analysis.pdf
ENV_NAME=bankenv

# Targets
all: $(REPORT)

# Data download
$(RAW_DATA):
	python src/01_download.py --directory data/ --filename bank-additional-full.csv

# Data cleaning
$(CLEANED_DATA): $(RAW_DATA)
	python src/02_clean_data.py --input_path $(RAW_DATA) --output_path $(CLEANED_DATA)

# Exploratory analysis
$(EDA_PLOTS): $(CLEANED_DATA)
	python src/03_explory_analysis.py --cleaned_data_path $(CLEANED_DATA) --output_prefix results/eda

# Report generation
$(REPORT): $(EDA_PLOTS)
	quarto render reports/bank_marketing_analysis.qmd --to pdf

# Set up environment
environment:
	conda env create -f environment.yml

# Clean intermediate and final files
clean:
	rm -f $(CLEANED_DATA) $(EDA_PLOTS) $(REPORT) reports/bank_marketing_analysis_files/*.png

# Run tests
test:
	pytest test/

