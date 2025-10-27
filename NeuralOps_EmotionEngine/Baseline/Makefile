#======================================================
# Makefile for Baseline Fault Injection on Darcy Flow
#------------------------------------------------------

# Variables
PYTHON := python

# --- PHONY Targets ---
.PHONY: all baseline

# --- Main Targets ---

# Run baseline
baseline:
	@echo "--- Running Baseline Fault Injection on Darcy Flow ---"
	PYTHONWARNINGS=ignore $(PYTHON) baseline_DarcyFI.py | tee logs/baselineDarcyFI.log
