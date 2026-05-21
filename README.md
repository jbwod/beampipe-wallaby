<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/4414e79f-7431-4999-b2ef-28cf9f0b254e">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/648d6a14-e1ee-4297-aa36-ff58f130e5d8">
   <img src="" />
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/2f989202-a13f-4928-b897-5aa595a5fb54">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/1f545ee5-2ef3-4a50-adbf-df96e2acba27">
   <img src="" />
</picture>

> **`beampipe-wallaby`** is the [WALLABY hi-res](https://github.com/ICRAR/wallaby-hires) project module for [`beampipe-core`](../README.md). It registers as **`wallaby_hires`** and plugs survey-specific discovery, metadata preparation, and DALiuGE manifest shaping into the core orchestration runtime - CASDA visibility discovery, VizieR HIPASS kinematics, async staging, and Slurm-remote execution on [Pawsey Setonix](https://pawsey.org.au/systems/setonix/).

## `What it does`

> - **`CASDA-driven discovery`**: queries `ivoa.obscore` for the latest WALLABY / ASKAP Pilot visibility per source, and fetches per-SBID evaluation files from CASDA.

> - **`Catalog enrichment`**: resolves RA, Dec, and systemic velocity from the VizieR HIPASS catalog (`VIII/73/hicat`) for pipeline inputs.

> - **`Manifest shaping`**: builds per-source manifests with `sbids[]`, staged visibility URLs, checksum URLs, and evaluation-file links for DALiuGE translation and deployment.

> - **`Automation policy`**: ships default discovery and execution automation settings (batch sizes, Slurm-remote deployment profile, concurrent run limits) consumed by beampipe-core schedulers and Restate workflows.

> - **`Graph reference`**: points execution at the WALLABY hi-res DALiuGE graph used on Setonix (overridable via `GRAPH_PATH` / `GRAPH_GITHUB_URL` in the module).

## `Module contract`

The package exposes a standard **`beampipe.projects`** entry point. beampipe-core loads `wallaby_hires.module` and calls:

| Hook | Role |
|------|------|
| `discover` | TAP queries via `casda` and `vizier` adapters |
| `prepare_metadata` | Normalises discovery rows into ledger-ready metadata |
| `manifest` | Produces execution manifests from staged archive URLs |

Required archive `core` adapters: **`casda`**, **`vizier`**.

## `Install`

This module is a **separate Python package**. beampipe-core does not ship it; it must be installed in the same environment.

tbd 

### Register the entry point
```toml
[project.entry-points."beampipe.projects"]
wallaby_hires = "wallaby_hires.module"
```

After install, verify discovery from the core tree:

```bash
cd ..
python -m app.core.projects.test_load
```

You should see `wallaby_hires` in the loaded module list.

### Configure beampipe-core

1. Run the core setup and supply **CASDA / OPAL** credentials (or set in root `.env`):

   ```bash
   CASDA_USERNAME=...
   CASDA_PASSWORD=...
   ```

2. Register sources in beampipe-core with **`project_module: wallaby_hires`** using the `HIPASS` identifier.

3. Optional: set **`CASDA_STAGE_BY_SBID=true`** in `.env` when multiple SBIDs share visibility basenames.

## `Defaults shipped by the module`

Key constants in `src/wallaby_hires/module.py` (override in code or via core deployment profiles):

- **`PROJECT_NAME`**: `wallaby_hires`
- **`GRAPH_GITHUB_URL`**: WALLABY hi-res Setonix graph (raw GitHub URL)
- **`WORKFLOW_DISCOVERY_AUTOMATION`**: enabled; CASDA archive; batch and stale-window tuning
- **`WORKFLOW_EXECUTION_AUTOMATION`**: enabled; `deployment_profile_name: slurm-remote`; execution poll limits for remote Slurm jobs

Import the [beampipe palette](https://github.com/jbwod/wallaby-hires-beampipe) into [EAGLE](https://eagle.icrar.org/) when editing DALiuGE graphs that consume beampipe-generated manifests.

## `Development`

```bash
cd beampipe-wallaby
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Run core with this module
cd ..
uv sync --extra wallaby
make dev
```

Run core tests to confirm load:

```bash
cd ..
pytest tests/test_casda_parse_job_results.py tests/test_discovery_metadata.py -q
```

## `Related`

- [beampipe-core](https://github.com/jbwod/beampipe-core) - orchestration runtime, setup wizard, Compose layouts
- [wallaby-hires](https://github.com/jbwod/wallaby-hires-beampipe) - science pipeline and DALiuGE graphs
- [DALiuGE](https://daliuge.icrar.org/) - workflow management
- [CASDA](https://casda.csiro.au/) - archive access

## Contributing

Follow the contributing guidelines in the beampipe-core repository when opening changes that affect both packages.