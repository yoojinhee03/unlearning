.PHONY: dashboard install run demo train unlearn evaluate experiment lint test

install:
	uv pip install -r requirements.txt

run:
	uv run streamlit run dashboard/app.py

demo:
	DEMO_MODE=true uv run streamlit run dashboard/app.py

dashboard: run

train:
	python -m src.train

unlearn:
	@test -n "$(INDICES)" || (echo "Usage: make unlearn INDICES='0 1 2'"; exit 1)
	python main.py unlearn --forget-indices $(INDICES)

evaluate:
	python main.py evaluate

experiment:
	python experiments/run_experiment.py \
		--num_shards 5 \
		--num_slices 4 \
		--unlearn_indices 0 1 2 3 4

lint:
	uv run ruff check src/ dashboard/
	uv run ruff format --check src/ dashboard/

test:
	uv run pytest tests/ -v