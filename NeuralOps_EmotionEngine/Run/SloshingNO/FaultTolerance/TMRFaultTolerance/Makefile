#======================================================
# Makefile for TMR Fault Tolerance 
#------------------------------------------------------

# Variables
PYTHON := python

# --- PHONY Targets ---
.PHONY: all baseline

# --- Main Targets ---

# Run TMR Analysis 
tmr_analysis:
	@echo "--- Running TMR Analysis ---"
	PYTHONWARNINGS=ignore $(PYTHON) TMRAnalysis.py | tee logs/TMRAnalysis.log

# Run TMR Implementation
tmr_implementation:
	@echo "--- Running TMR Implementation ---"
	PYTHONWARNINGS=ignore $(PYTHON) TMRFaultTolerance.py | tee logs/TMRFaultTolerance.log

