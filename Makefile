.PHONY: install install-dev clean train demo generate generate-scienceqa generate-small eval test lint help validate-data verify

PYTHON = python
PIP = pip

# Default target
help:
	@echo "V-CoT: Visual Chain-of-Thought"
	@echo ""
	@echo "Available targets:"
	@echo "  install      - Install production dependencies"
	@echo "  install-dev  - Install with development dependencies"
	@echo "  generate     - Generate canonical VisCOT training data"
	@echo "  generate-scienceqa - Generate ScienceQA data using GPT-4o"
	@echo "  train        - Train the model"
	@echo "  eval         - Evaluate model on validation set"
	@echo "  demo         - Launch Gradio demo"
	@echo "  demo-base    - Launch demo with base model (no training required)"
	@echo "  validate-data - Validate processed JSONL files"
	@echo "  test         - Run syntax checks"
	@echo "  clean        - Remove generated files"
	@echo ""
	@echo "Quick start:"
	@echo "  1. make install"
	@echo "  2. export OPENAI_API_KEY='sk-...'"
	@echo "  3. make generate"
	@echo "  4. make train"
	@echo "  5. make demo"

install:
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	# Install Unsloth with specific CUDA support
	$(PIP) install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
	$(PIP) install -r requirements.txt
	@echo ""
	@echo "Installation complete! Run 'make help' for available commands."

install-dev: install
	$(PIP) install pytest black ruff

# Data generation
generate:
	@echo "Generating training data..."
	$(PYTHON) scripts/generate_data.py \
		--source viscot \
		--output_dir data/processed \
		--max_samples 2000

generate-scienceqa:
	@echo "Generating ScienceQA training data..."
	@if [ -z "$$OPENAI_API_KEY" ]; then \
		echo "ERROR: OPENAI_API_KEY not set"; \
		echo "Run: export OPENAI_API_KEY='sk-...'"; \
		exit 1; \
	fi
	$(PYTHON) scripts/generate_data.py \
		--source scienceqa \
		--output_dir data/processed \
		--max_samples 2000 \
		--save_images

generate-small:
	@echo "Generating small dataset (100 samples) for testing..."
	$(PYTHON) scripts/generate_data.py \
		--source viscot \
		--output_dir data/processed \
		--max_samples 100

# Training
train:
	@echo "Starting training..."
	$(PYTHON) scripts/train.py --config configs/default.yaml

train-debug:
	@echo "Starting debug training run (10 steps)..."
	$(PYTHON) scripts/train.py --config configs/default.yaml \
		--mode text_only_debug \
		--training.max_steps 10 \
		--training.logging_steps 1

train-test:
	@echo "Starting test training with sample data..."
	$(PYTHON) scripts/train.py --config configs/test.yaml --mode text_only_debug

verify:
	@echo "Verifying V-CoT setup..."
	$(PYTHON) scripts/verify_setup.py

validate-data:
	@echo "Validating processed datasets..."
	@if [ -f data/processed/train.jsonl ]; then $(PYTHON) scripts/validate_data.py --input data/processed/train.jsonl; else echo "Skipping train.jsonl (not found)"; fi
	@if [ -f data/processed/val.jsonl ]; then $(PYTHON) scripts/validate_data.py --input data/processed/val.jsonl; else echo "Skipping val.jsonl (not found)"; fi

# Evaluation
eval:
	@echo "Running evaluation..."
	$(PYTHON) scripts/inference.py \
		--model_path outputs/checkpoints \
		--eval_jsonl data/processed/val.jsonl \
		--output_json results/evaluation.json

# Demo
demo:
	@echo "Launching V-CoT demo..."
	$(PYTHON) app/app.py --model_path outputs/checkpoints

demo-base:
	@echo "Launching demo with base model..."
	$(PYTHON) app/app.py --model_path unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit

demo-share:
	@echo "Launching demo with public link..."
	$(PYTHON) app/app.py --model_path outputs/checkpoints --share

# Testing
test:
	@echo "Running syntax checks..."
	$(PYTHON) -m py_compile src/*.py scripts/*.py app/*.py
	@echo "All syntax checks passed!"

lint:
	ruff check src/ scripts/ app/
	black --check src/ scripts/ app/

format:
	black src/ scripts/ app/

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache/ .ruff_cache/

clean-all: clean
	rm -rf logs/* outputs/* data/processed/*
	@echo "Cleaned all generated files"
