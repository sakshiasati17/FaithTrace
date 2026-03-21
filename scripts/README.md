# Scripts

Utility scripts for data generation, corpus management, and migration.

| Script | Purpose |
|---|---|
| `generate_corpus.py` | Generate a synthetic enterprise corpus with versioned documents |
| `generate_eval_set.py` | Generate evaluation question sets from a corpus |
| `seed_db.py` | Seed the database with sample experiments for development |
| `export_results.py` | Export experiment results to CSV or JSON |

Run scripts from the repo root:

```bash
python scripts/generate_corpus.py --output corpus/
python scripts/seed_db.py
```
