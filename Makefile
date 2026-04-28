.PHONY: data train eval infer all clean help

# Default target
help:
	@echo "xLAM Llama 3.1 qLoRA Fine-tuning"
	@echo ""
	@echo "Available targets:"
	@echo "  make data       - Download and format xLAM dataset"
	@echo "  make train      - Train LoRA adapter"
	@echo "  make eval       - Evaluate base vs LoRA model"
	@echo "  make infer      - Run single-query inference (use QUERY=...)"
	@echo "  make all        - Run data + train + eval"
	@echo "  make clean      - Remove data/ and outputs/ directories"
	@echo ""
	@echo "Example:"
	@echo "  make infer QUERY='get the weather in Tokyo'"

# Data preparation
data:
	@echo "Preparing xLAM dataset..."
	python -m src.data_prep

# Training
train:
	@echo "Starting qLoRA training..."
	python -m src.train

# Evaluation
eval:
	@echo "Evaluating models..."
	python -m src.eval

# Inference (requires QUERY variable)
infer:
ifndef QUERY
	@echo "Error: QUERY variable not set"
	@echo "Usage: make infer QUERY='your query here'"
	@exit 1
endif
	@echo "Running inference..."
	python -m src.inference --query "$(QUERY)"

# Run full pipeline
all: data train eval

# Clean generated files
clean:
	@echo "Cleaning data/ and outputs/ directories..."
	rm -rf data/ outputs/
	@echo "Clean complete"
