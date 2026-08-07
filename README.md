# csv-eda-mcp

An HTTP MCP server for exploring CSV files **without letting PII values reach the model**.
PII columns are masked *before* any pandas code runs, so it does not matter what code the
calling LLM writes — the raw values are already gone from the DataFrame.

Built for small local models (llama.cpp / Ollama, 4B–12B) as much as for cloud ones:
the schema tools are cheap, the EDA tool takes whatever pandas the model produces, and
the guardrail is enforced server-side rather than by prompting.

## Tools

| Tool | What it does |
|---|---|
| `list_files` | CSV files visible to the server (data dir + registered sources) |
| `csv_schema` | Header, column dtypes, null %, row/column counts |
| `detect_pii` | PII columns from column names (~24 ordered regex patterns) |
| `set_pii_columns` | Persist the definitive PII list for a file |
| `classify_pii_local` | Classify PII with the local LLM and save the result |
| `run_eda` | Run pandas code with the PII guardrail; returns stdout + last expression |
| `visualize` | Same, rendered as a styled HTML table page; returns URL + markdown preview |
| `export_table` | Write a view to CSV or Excel (still masked) and return its URL |
| `llm_status` | GPU memory and which local models llama-swap has loaded |
| `llm_stop` | Unload every loaded model, freeing VRAM immediately |

`run_eda` and `visualize` also take `joins` — up to four other files merged in first, so
analysis can span files (see [Joins](#joins)).

### How PII is decided

Precedence, highest first:

1. `pii_columns` passed explicitly to `run_eda` / `visualize` (a cloud model can pass what
   it deduced from `csv_schema` alone — no values needed)
2. The list saved for that file, by the webui, `set_pii_columns`, or `classify_pii_local`
3. Name-based regex patterns

`classify_pii_local` sends only column names, dtypes and aggregate stats (null %, distinct
count) to `LOCAL_LLM_URL` — never a row. It catches semantic cases the name patterns miss,
where a column reads as innocuous but holds personal or health information.

### Guardrails in `run_eda` / `visualize`

- PII columns replaced with `***` (see `REDACTION_MASK`) **before** the code executes
- AST rejects imports, `open`/`eval`/`exec`, dunder access, and `pd.read_*` / `to_*` I/O
- Restricted builtins; no filesystem or network from user code
- Paths confined to the data dir and registered sources
- Output clipped to `MAX_OUTPUT_CHARS` (default 6000)

Redaction is a masking guardrail, not anonymisation: it stops values from surfacing in
tool output. It does not defend against a caller who is authorised to pass `redact=False`.

## Joins

A join can span up to four files. Each file contributes its **own** PII list, and masking
happens once, after the merges — so a masked column still works as a join key: it matches
on the real values and is masked in the result. Filtering on that key through generated
code still matches nothing, because the frame is masked before any caller code runs.

Three ways in:

- **Viewer** — the Join builder: pick a file, the column on each side, and the type
  (left / inner / right / outer). Joins are remembered per base file.
- **Ask box** — a question mentioning join/merge/combine is planned by the local LLM,
  validated against real columns, then applied; the Join panel and its storage adopt the
  spec, so the two never disagree.
- **MCP** — pass `joins` to `run_eda` or `visualize`.

## Web UI

Served at `/` — the place to curate PII per file and to read the data:

- **Files** — every known CSV with column count, PII count, auto/manual badge and detected
  types; folder filter, name search, sortable columns, pagination, copy-path button
- **Data viewer** — page through a file with the header and row numbers pinned, horizontal
  scroll for wide files, type-aware sorting from the header (numbers numerically, dates
  chronologically, blanks last), a column picker, and a Hide PII toggle. The **Ask** box
  turns a question into pandas (written by the local LLM, run under the same guardrail)
  and filters the view in place — grouping and HAVING-style conditions included. The
  generated code is shown, and nothing is saved
- **File detail** — tick/untick PII columns (auto-detection pre-checked), save or reset
- **Sources** — registered sources plus a server-side filesystem browser

Per file, the browser remembers the last 10 questions (each with the join it was asked
against), the chosen columns and the joins. Reports written by `visualize` are served at
`/reports/<name>.html`: paginated client-side, magnitude bars on numeric columns,
light/dark, redaction noted in the footer.

## Exports and GPU

The viewer's **Export CSV / Excel** buttons and the `export_table` tool write the *whole*
current view — joins, question, sort and chosen columns — not just the visible page.
The frame is the same masked one the guardrail produces, so a masked column exports as
`***`. Files land in `data/exports/` and are served at `/exports/<name>`.

`llm_status` and `llm_stop` exist because VRAM is finite: llama-swap unloads a model on
its own idle TTL, and `llm_stop` is the "give it back now" button. It unloads the whole
llama-swap instance, so it stops models other clients loaded too — the result names them.
`start.sh` gives the container GPU visibility (`runtime: nvidia`, driver capability
`utility`, i.e. nvidia-smi only — no CUDA) when the host has the nvidia runtime; without
it `llm_status` still reports what is loaded, just with `gpus: null`.

## Setup

```sh
cp .env.example .env      # then edit it
./start.sh                # generates docker-compose.override.yml from .env, builds, starts
./update-volumes.sh       # re-points the mounts only — no rebuild
```

`start.sh` exists because compose cannot expand a variable-length volume list from one
env var: it turns each `CSV_SOURCES` entry into a read-only `/mnt/<name>` mount. Mounts are
browsable in the UI; a file appears in the Files list once you register it (a whole mount
is one click under Sources). Both scripts share that generation logic via `volumes.sh`.

Reach for `update-volumes.sh` after editing `CSV_SOURCES`, or when a mounted host dir was
replaced rather than edited in place (re-clone, atomic `mv`) — a bind mount pins the source
inode at container start, so a swapped-out dir keeps serving the old contents until the
container is recreated. Edits *inside* a mounted dir need neither script; bind mounts are live.

### Environment

| Variable | Purpose |
|---|---|
| `PORT` | Host port for the MCP endpoint and web UI (default 41733) |
| `CSV_SOURCES` | `/host/path[:name],…` — each mounted read-only at `/mnt/<name>` |
| `REDACTION_MASK` | What PII values become (default `***`) |
| `PUBLIC_BASE_URL` | Base URL for report links; empty gives relative paths |
| `LOCAL_LLM_URL` / `LOCAL_LLM_MODEL` | OpenAI-compatible endpoint for PII classification, the Ask box and join planning |
| `LLAMA_SWAP_URL` | llama-swap control API for `llm_status` / `llm_stop`; derived from `LOCAL_LLM_URL` when unset |
| `TS_AUTHKEY` | Tailscale auth key for the optional `ts-csv-analyst` sidecar |

Config lives in `data/config.json` (registered sources + per-file PII lists); reports in
`data/reports/`. Both are gitignored.

## Connecting a client

```sh
claude mcp add --transport http csv-analyst http://localhost:41733/mcp
```

Browser-based clients (llama.cpp web UI, other tailnet apps) need HTTPS — an `https://`
page cannot fetch `http://localhost`. The compose file ships a Tailscale sidecar that
serves the container at `https://csv-analyst.<your-tailnet>.ts.net`; set `TS_AUTHKEY` and
use that URL with `/mcp`. CORS is open on the MCP endpoint, and the SDK's DNS-rebinding
Host allowlist is disabled so non-localhost hostnames are accepted — access control is the
tailnet itself, so do not expose this server publicly.

## Notes

- Streamable HTTP, stateless, JSON responses — no session handshake, friendly to
  lightweight clients
- Endpoints: `/mcp`, `/` (UI), `/reports/<name>`, `/exports/<name>`, `/api/{files,preview,query,join,export,browse,columns,pii,sources}`
- Row counts come from CSV records, not lines: a quoted field may contain newlines, and a
  line count overreports
- `data/sample.csv` is a small demo file with obvious PII columns

## License

MIT — see [LICENSE](LICENSE).
