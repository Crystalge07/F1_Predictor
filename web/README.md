# The dashboard

React + TypeScript (Vite). It reads only the static JSON files exported by
`scripts/export_dashboard.py`; it never talks to the database or a server.
Deployed to GitHub Pages by the hourly pipeline (`.github/workflows/tick.yml`)
whenever the data or the code changed.

```sh
cd web
npm ci                      # exact versions from package-lock.json
npm test                    # front-end tests (data from src/test/data)
npm run build               # type-check + build into dist/

# Run it locally with your real data:
python ../scripts/export_dashboard.py ../data/f1.sqlite public   # -> public/data (gitignored)
npm run dev
```

`src/test/data` is the export of the simulated test world
(`python scripts/export_test_data.py` from the repo root). A Python test checks
it still matches what the export writes, so the app is always tested against
the real file format. Re-run the script after changing the export.
