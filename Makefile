PYTHON ?= python

.PHONY: test smoke-data validate sft-smoke api demo

test:
	$(PYTHON) -m pytest -q

smoke-data:
	$(PYTHON) scripts/make_smoke_data.py

validate:
	rs-vlm-data validate --inputs data/processed/sft_train.jsonl data/processed/sft_val.jsonl data/processed/sft_test.jsonl --require-images

sft-smoke:
	SMOKE=1 bash scripts/train_sft.sh

api:
	rs-vlm-api

demo:
	rs-vlm-demo

