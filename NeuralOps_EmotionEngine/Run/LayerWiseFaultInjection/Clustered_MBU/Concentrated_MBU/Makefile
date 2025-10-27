#===========================================================
# Makefile for Clustered SEU FI Campaign (Concentrated MBU)
#-----------------------------------------------------------

# Variables
PYTHON := python

# --- PHONY Targets ---
.PHONY: all baseline

# --- Main Targets ---

# Run IsolatedSEU 
concentratedMBU:
	@echo "--- Running Concentrated MBU Fault Injection Campaign ---"
	PYTHONWARNINGS=ignore $(PYTHON) ConcentratedMBULayerWise.py | tee logs/ConcentratedMBU.log
