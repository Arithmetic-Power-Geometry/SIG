.PHONY: all test run
all: test run
test:
	python -m pytest -q
run:
	python run_all.py
