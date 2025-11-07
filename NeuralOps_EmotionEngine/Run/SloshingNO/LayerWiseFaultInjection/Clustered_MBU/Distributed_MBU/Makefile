#===========================================================
# Makefile for Clustered SEU FI Campaign (Distributed MBU)
#-----------------------------------------------------------

# Variables
PYTHON := python

# --- PHONY Targets ---
.PHONY: all baseline

# --- Main Targets ---

# Run IsolatedSEU 
distributedMBU:
	@echo "--- Running Distributed MBU Fault Injection Campaign ---"
	PYTHONWARNINGS=ignore $(PYTHON) DistributedMBULayerWise.py | tee logs/DistributedMBU.log
