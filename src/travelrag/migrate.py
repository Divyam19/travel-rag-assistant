"""Apply db/migrations/*.sql in order. Usage: python -m travelrag.migrate"""

from .config import ROOT
from .db import connect

MIGRATIONS_DIR = ROOT / "db" / "migrations"


def main() -> None:
    files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    with connect(with_vectors=False) as conn:
        conn.execute(
            "create table if not exists public.schema_migrations ("
            " version text primary key, applied_at timestamptz not null default now())"
        )
        conn.execute("alter table public.schema_migrations enable row level security")
        conn.commit()
        applied = {r[0] for r in conn.execute("select version from public.schema_migrations")}
        pending = [f for f in files if f.name not in applied]
        if not pending:
            print("Nothing to apply; database is up to date.")
            return
        for f in pending:
            print(f"Applying {f.name} ...")
            with conn.transaction():
                conn.execute(f.read_text())
                conn.execute("insert into public.schema_migrations (version) values (%s)", (f.name,))
        print(f"Applied {len(pending)} migration(s).")


if __name__ == "__main__":
    main()
