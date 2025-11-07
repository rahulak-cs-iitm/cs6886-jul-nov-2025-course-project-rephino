#======================================================
# Makefile for Isolated SEU FI Campaign 
#------------------------------------------------------

# Variables
PYTHON := python

# --- PHONY Targets ---
.PHONY: all baseline

# --- Main Targets ---

# Run IsolatedSEU 
isolatedSEU:
	@echo "--- Running Isolated SEU Fault Injection Campaign ---"
	PYTHONWARNINGS=ignore $(PYTHON) IsolatedSEULayerWise.py | tee logs/IsolatedSEU.log
