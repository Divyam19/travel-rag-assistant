PY := .venv/bin/python
CRON_MARK := \# travelrag-ingest
CRON_LINE := 0 */4 * * * cd $(CURDIR) && $(PY) -m travelrag.ingest >> logs/ingest.log 2>&1 $(CRON_MARK)

.PHONY: setup migrate doctor ingest stats chat test api web web-install cron-install cron-remove

setup:
	python3.12 -m venv .venv
	$(PY) -m pip install -q -e .

migrate:
	$(PY) -m travelrag.migrate

# Snapshot of the live public schema -> db/schema.sql (read-only reference; migrations stay the source of truth).
# pg_dump must be >= the server version (Supabase runs 17): brew install postgresql@17
PG_DUMP ?= $(firstword $(wildcard /opt/homebrew/opt/postgresql@17/bin/pg_dump) pg_dump)
schema:
	$(PG_DUMP) --schema-only --no-owner --no-privileges --schema=public \
		"$$($(PY) -c 'from travelrag.config import get_settings; print(get_settings().supabase_db_url)')" \
		-f db/schema.sql
	@echo "Wrote db/schema.sql"

doctor:
	$(PY) -m travelrag.doctor

ingest:
	$(PY) -m travelrag.ingest

stats:
	$(PY) -m travelrag.stats

# Re-scrape every 4 hours. Idempotent, so runs missed while the laptop sleeps are harmless.
cron-install:
	mkdir -p logs
	(crontab -l 2>/dev/null | grep -v 'travelrag-ingest'; echo '$(CRON_LINE)') | crontab -
	@echo "Installed:"; crontab -l | grep travelrag-ingest

cron-remove:
	(crontab -l 2>/dev/null | grep -v 'travelrag-ingest') | crontab -

chat:
	$(PY) -m travelrag.chat

test:
	$(PY) -m pytest -q

# Backend on :8000, then the UI on :5173 (in a second terminal). `make web-install` once first.
api:
	$(PY) -m uvicorn travelrag.api:app --port 8000

web-install:
	cd frontend && npm install

web:
	cd frontend && npm run dev
