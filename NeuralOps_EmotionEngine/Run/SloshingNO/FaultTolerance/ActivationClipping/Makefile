#======================================================
# Makefile for Activation Clipping
#------------------------------------------------------

# Variables
PYTHON := python

# --- PHONY Targets ---
.PHONY: all baseline

# --- Main Targets ---

# Run Collect Max Activation
collectMaxActivation:
	@echo "--- Collecting Max Activation Value ---"
	PYTHONWARNINGS=ignore $(PYTHON) CollectMaxActivation.py | tee logs/CollectMaxActivation.log

# Run Activation Clipping 
activationClipping:
	@echo "--- Running Activation Clipping ---"
	PYTHONWARNINGS=ignore $(PYTHON) ActivationClipping.py | tee logs/ActivationClipping.log
