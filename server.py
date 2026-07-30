"""csv-analyst — HTTP MCP server for CSV schema inspection, PII detection, and guarded pandas EDA.

Config (registered sources + per-file PII overrides) lives in DATA_DIR/config.json,
editable through the bundled webui served at "/".
"""

import ast
import csv
import html
import io
import json
import os
import re
import shutil
import subprocess
import contextlib
import warnings
from datetime import datetime
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
from mcp.server.mcpserver import MCPServer
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response

DATA_DIR = Path(os.environ.get("DATA_DIR", ".")).resolve()
CONFIG_FILE = DATA_DIR / "config.json"
REPORTS_DIR = DATA_DIR / "reports"
EXPORTS_DIR = DATA_DIR / "exports"
# base URL reports are reachable at (e.g. the tailnet hostname); relative when unset
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
TEMPLATES = Path(__file__).parent / "templates"
BROWSE_ROOTS = [
    Path(p).resolve()
    for p in os.environ.get("BROWSE_ROOTS", f"{DATA_DIR}:/mnt").split(":")
    if p and Path(p).is_dir()
]
PORT = int(os.environ.get("PORT", "41733"))
MAX_OUTPUT_CHARS = int(os.environ.get("MAX_OUTPUT_CHARS", "6000"))
MAX_LISTED_FILES = 500
ROW_SORT_KEY = "__row__"  # the viewer's row-number gutter
REDACTION_MASK = os.environ.get("REDACTION_MASK", "***")
LOCAL_LLM_URL = os.environ.get("LOCAL_LLM_URL", "").rstrip("/")
LOCAL_LLM_MODEL = os.environ.get("LOCAL_LLM_MODEL", "local")
# llama-swap's control API sits above its per-model /upstream/<model>/v1 proxy
LLAMA_SWAP_URL = (
    os.environ.get("LLAMA_SWAP_URL") or LOCAL_LLM_URL.split("/upstream/")[0]
).rstrip("/").removesuffix("/v1")

mcp = MCPServer("csv-analyst")

# column-name pattern -> pii type (first match wins; heuristic defaults, override via webui)
PII_PATTERNS = [
    (r"e[-_ ]?mail", "email"),
    (r"pass(word|wd)|pwd|secret|token|api[-_ ]?key", "credential"),
    (r"ssn|social[-_ ]?sec|nir|insee", "national_id"),
    (r"passport|driver[-_ ]?licen|licen[cs]e[-_ ]?(no|num)", "government_id"),
    (r"iban|swift|bic|routing|account[-_ ]?(no|num)|acct", "bank_account"),
    (r"(credit|debit)[-_ ]?card|cvv|cvc", "payment_card"),
    (r"phone|mobile|cell|fax|msisdn|(^|[-_ ])tel(ephone)?([-_ ]|$)", "phone"),
    (r"user[-_ ]?(name|id)|user$|login|(^|[-_ ])uid([-_ ]|$)|submitter|sender|creator", "identifier"),
    (r"(human|person|patient|customer|employee|victim|member|subject)[-_ ]?(id|serial|no|num)", "identifier"),
    (r"(^|[-_ ])(sur|first|last|full|middle|family|given|maiden|nick|popular|other|owner|victim|patient|customer|employee)[-_ ]?names?([-_ ]|$)", "person_name"),
    (r"^names?$|^nom$|prenom", "person_name"),
    (r"birth|naissance|(^|[-_ ])dob([-_ ]|$)|born", "date_of_birth"),
    (r"(^|[-_ ])age([-_ ]|$)|age[-_ ]?(group|years)", "age"),
    (r"(^|[-_ ])(sex|gender)([-_ ]|$)|ethnic|race|religion|nationality", "sensitive_attribute"),
    (r"address|addr([-_ ]|$)|street|(^|[-_ ])rue([-_ ]|$)|residence", "address"),
    (r"zip|postal|postcode", "postal_code"),
    (r"location|city|town|ville|village|ward|district|region|county|province|municipal", "location"),
    (r"lat(itude)?$|^lng$|long(itude)?$|(^|[-_ ])(geo|gps)|coord", "geolocation"),
    (r"(^|[-_ ])ip[-_ ]?(addr|address)?$", "ip_address"),
    (r"(^|[-_ ])mac([-_ ]|$)|imei|device[-_ ]?id|serial[-_ ]?(no|num)", "device_id"),
    (r"salary|income|wage|revenu", "financial"),
    (r"diagnos|medical|health|disease|condition|treatment|medication|symptom|death", "health"),
    (r"comment|remark|narrative|notes?$", "free_text"),
    (r"names?$", "person_name"),
]


def _config() -> dict:
    if CONFIG_FILE.is_file():
        cfg = json.loads(CONFIG_FILE.read_text())
    else:
        cfg = {}
    cfg.setdefault("sources", [])
    cfg.setdefault("pii", {})
    return cfg


def _save_config(cfg: dict) -> None:
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


def _mounted_sources() -> list[Path]:
    mnt = Path("/mnt")
    if not mnt.is_dir():
        return []
    return sorted(d for d in mnt.iterdir() if d.is_dir())


def _roots(cfg: dict | None = None) -> list[Path]:
    """Where files may be read from. Mounts are browsable but NOT included: listing
    a whole mounted repo buries the handful of files someone actually registered."""
    cfg = cfg or _config()
    return [DATA_DIR] + [Path(s) for s in cfg["sources"]]


def _resolve(path: str) -> Path:
    p = Path(path)
    p = (p if p.is_absolute() else DATA_DIR / p).resolve()
    for root in _roots():
        root = root.resolve()
        if p == root or p.is_relative_to(root):
            break
    else:
        raise ValueError(
            f"'{path}' is not under a registered source; register its directory in the webui first"
        )
    if not p.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    return p


def _known_files() -> list[str]:
    seen: set[str] = set()
    for root in _roots():
        root = root.resolve()
        if root.is_file() and root.suffix.lower() == ".csv":
            seen.add(str(root))
        elif root.is_dir():
            for f in root.rglob("*.csv"):
                if EXPORTS_DIR in f.parents:  # our own output, not source data
                    continue
                seen.add(str(f))
                if len(seen) >= MAX_LISTED_FILES:
                    return sorted(seen)
    return sorted(seen)


def _sniff(p: Path) -> str:
    with open(p, newline="", errors="replace") as f:
        sample = f.read(65536)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _load(p: Path, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(p, sep=_sniff(p), nrows=nrows, on_bad_lines="skip")


def _header(p: Path) -> list[str]:
    with open(p, newline="", errors="replace") as f:
        return next(csv.reader(f, delimiter=_sniff(p)))


_ROW_COUNTS: dict[tuple, int] = {}


def _record_count(p: Path) -> int:
    """Data rows in the file. Counts CSV records, not lines: quoted fields may
    contain newlines, so a line count overreports (and paged reads then run off
    the end of the data). Cached per file revision — the scan is a full pass."""
    st = p.stat()
    key = (str(p), st.st_mtime_ns, st.st_size)
    if key not in _ROW_COUNTS:
        with open(p, newline="", errors="replace") as f:
            n = sum(1 for _ in csv.reader(f, delimiter=_sniff(p)))
        if len(_ROW_COUNTS) > 64:
            _ROW_COUNTS.clear()
        _ROW_COUNTS[key] = max(n - 1, 0)
    return _ROW_COUNTS[key]


def _classify_columns(columns: list[str]) -> list[dict]:
    found = []
    for col in columns:
        norm = col.strip().lower()
        for pattern, pii_type in PII_PATTERNS:
            if re.search(pattern, norm):
                found.append({"column": col, "pii_type": pii_type})
                break
    return found


def _pii_override(p: Path) -> list[str] | None:
    return _config()["pii"].get(str(p))


@mcp.tool()
def list_files() -> dict:
    """List all CSV files known to the server (data directory + registered sources)."""
    return {"files": _known_files()}


@mcp.tool()
def csv_schema(path: str, sample_rows: int = 500) -> dict:
    """Get the schema of a CSV file: header, column names and inferred types, number of columns and rows.

    Args:
        path: CSV file path (absolute, or relative to the server data directory).
        sample_rows: rows sampled for type inference (default 500).
    """
    p = _resolve(path)
    delimiter = _sniff(p)
    df = _load(p, nrows=sample_rows)
    return {
        "file": str(p),
        "delimiter": delimiter,
        "n_columns": len(df.columns),
        "n_rows": _record_count(p),
        "columns": [
            {
                "name": c,
                "dtype": str(df[c].dtype),
                "null_pct_sample": round(float(df[c].isna().mean() * 100), 1),
            }
            for c in df.columns
        ],
    }


@mcp.tool()
def detect_pii(path: str = "", columns: list[str] | None = None) -> dict:
    """Deduce which columns likely contain PII.

    Provide either a CSV path OR an explicit list of column names. A user-defined
    PII list saved for the file (via webui or set_pii_columns) takes precedence
    over name-based auto-detection.

    Args:
        path: CSV file path (header row is read).
        columns: column names to classify instead of reading a file.
    """
    source = "auto"
    if columns is None:
        if not path:
            raise ValueError("provide 'path' or 'columns'")
        p = _resolve(path)
        columns = _header(p)
        override = _pii_override(p)
        if override is not None:
            return {
                "pii_columns": [{"column": c, "pii_type": "user_defined"} for c in override],
                "n_pii": len(override),
                "n_columns": len(columns),
                "source": "manual",
                "note": "User-defined PII list. Never print raw values from pii_columns.",
            }
    pii = _classify_columns(columns)
    return {
        "pii_columns": pii,
        "n_pii": len(pii),
        "n_columns": len(columns),
        "source": source,
        "note": "Name-based heuristic detection. Never print raw values from pii_columns.",
    }


@mcp.tool()
def set_pii_columns(path: str, columns: list[str]) -> dict:
    """Persist the definitive PII column list for a CSV file (overrides auto-detection).

    Args:
        path: CSV file path.
        columns: exact column names to treat as PII (empty list = no PII).
    """
    p = _resolve(path)
    header = set(_header(p))
    unknown = [c for c in columns if c not in header]
    if unknown:
        raise ValueError(f"columns not in file header: {unknown}")
    cfg = _config()
    cfg["pii"][str(p)] = columns
    _save_config(cfg)
    return {"file": str(p), "pii_columns": columns, "saved": True}


# --- LLM-based PII classification ---

PII_TYPES = sorted({t for _, t in PII_PATTERNS})

PII_PROMPT = (
    "You classify CSV columns for PII (personally identifiable information).\n"
    "Return ONLY a JSON array of objects "
    '[{"column": "<exact column name>", "pii_type": "<type>"}] listing the columns that contain PII. '
    f"Use pii_type values from: {', '.join(PII_TYPES)}.\n"
    "Include direct identifiers and quasi-identifiers: person names, contact details, personal IDs, "
    "device IDs, locations below country level, birth dates, ages, sensitive attributes (sex, "
    "ethnicity, religion, health), and free-text fields that may embed PII. "
    "Exclude purely operational columns: statuses, categories, measurements, amounts, timestamps of "
    "records, and facility/organization fields.\n\n"
)


def _llm_chat(prompt: str, max_tokens: int = 1500) -> tuple[str, str, str]:
    """Call the local model. Returns (content, reasoning, finish_reason).

    Thinking is switched off where the backend supports it: these models spend
    their whole budget reasoning and then emit nothing, and a filter expression
    does not need a chain of thought. Backends that reject the flag are retried
    without it, and the reasoning text is kept as a salvage path.
    """
    import httpx

    body = {
        "model": LOCAL_LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    url = f"{LOCAL_LLM_URL}/chat/completions"
    try:
        r = httpx.post(url, json={**body, "chat_template_kwargs": {"enable_thinking": False}}, timeout=300)
        if r.status_code >= 400:
            r = httpx.post(url, json=body, timeout=300)
    except httpx.HTTPError:
        r = httpx.post(url, json=body, timeout=300)
    r.raise_for_status()

    choice = r.json()["choices"][0]
    message = choice.get("message", {})
    return (
        (message.get("content") or "").strip(),
        (message.get("reasoning_content") or message.get("reasoning") or "").strip(),
        choice.get("finish_reason", ""),
    )


def _extract_code(text: str) -> str:
    """Pull pandas out of a small model's reply: fenced block, else the df line."""
    fence = re.search(r"```(?:python|py)?\s*(.+?)```", text, re.S)
    body = fence.group(1) if fence else text
    lines = [ln.rstrip() for ln in body.strip().splitlines() if ln.strip()]
    if not lines:
        return ""
    if fence:
        return "\n".join(lines)
    # unfenced: keep the code-looking lines in order. A grouped answer ends on a
    # line like "g[g['n'] > 20]" that never mentions df, so track what earlier
    # lines assigned and treat references to those as code too.
    code: list[str] = []
    assigned: set[str] = set()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("#", "-", ">", "*")):
            continue
        target = re.match(r"^(\w+)\s*=[^=]", stripped)
        words = set(re.findall(r"\w+", stripped))
        if "df" in words or target or (assigned & words):
            if target:
                assigned.add(target.group(1))
            code.append(stripped)
    return "\n".join(code)


def _extract_json_array(text: str) -> list:
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON array in LLM output: {text[:300]}")
    return json.loads(text[start:end + 1])


def _save_pii_result(p: Path, items: list) -> list[dict]:
    header = set(_header(p))
    kept = [
        {"column": it["column"], "pii_type": str(it.get("pii_type", "unknown"))}
        for it in items
        if isinstance(it, dict) and it.get("column") in header
    ]
    cfg = _config()
    cfg["pii"][str(p)] = [it["column"] for it in kept]
    _save_config(cfg)
    return kept


@mcp.tool()
def classify_pii_local(path: str) -> dict:
    """Classify PII columns with the local LLM (llama.cpp) and save the result for the file.

    Sends column names, dtypes and shape statistics (null %, distinct count) to the
    OpenAI-compatible endpoint at LOCAL_LLM_URL. Raw values never leave the machine.
    The saved list overrides pattern-based auto-detection (like set_pii_columns).

    Args:
        path: CSV file path.
    """
    import httpx

    if not LOCAL_LLM_URL:
        raise ValueError("LOCAL_LLM_URL is not configured (set it in .env)")
    p = _resolve(path)
    df = _load(p, nrows=500)
    lines = [
        f"- {c} (dtype={df[c].dtype}, null_pct={df[c].isna().mean() * 100:.0f}, distinct={df[c].nunique()})"
        for c in df.columns
    ]
    content, reasoning, _ = _llm_chat(PII_PROMPT + "Columns:\n" + "\n".join(lines), max_tokens=2048)
    items = _extract_json_array(content or reasoning)
    kept = _save_pii_result(p, items)
    return {
        "file": str(p),
        "pii_columns": kept,
        "n_pii": len(kept),
        "source": "local_llm",
        "model": LOCAL_LLM_MODEL,
        "saved": True,
    }


# --- guarded exec for run_eda ---

BLOCKED_NAMES = {
    "open", "exec", "eval", "compile", "input", "breakpoint", "globals", "locals",
    "vars", "getattr", "setattr", "delattr", "memoryview", "__import__",
}
BLOCKED_ATTRS = {
    "read_csv", "read_table", "read_fwf", "read_json", "read_excel", "read_parquet",
    "read_pickle", "read_sql", "read_html", "read_clipboard", "to_csv", "to_json",
    "to_excel", "to_parquet", "to_pickle", "to_sql", "to_clipboard", "eval", "query",
}
SAFE_BUILTINS = {
    n: __builtins__[n] if isinstance(__builtins__, dict) else getattr(__builtins__, n)
    for n in (
        "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float",
        "format", "frozenset", "int", "isinstance", "len", "list", "map", "max", "min",
        "print", "range", "repr", "reversed", "round", "set", "sorted", "str", "sum",
        "tuple", "type", "zip",
    )
    if (n in __builtins__ if isinstance(__builtins__, dict) else hasattr(__builtins__, n))
}


def _check_code(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("imports are not allowed; 'pd', 'np' and 'df' are preloaded")
        if isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            raise ValueError(f"'{node.id}' is not allowed")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                raise ValueError("dunder attribute access is not allowed")
            if node.attr in BLOCKED_ATTRS:
                raise ValueError(f"'.{node.attr}()' is not allowed (file/SQL I/O is disabled)")


MAX_JOINS = 4
JOIN_HOWS = ("left", "inner", "right", "outer")
MAX_JOIN_ROWS = 50_000


def _pii_columns_of(p: Path, columns: list[str]) -> list[str]:
    """The PII column names for a file, from its saved list or name detection."""
    override = _pii_override(p)
    targets = override if override is not None else [
        m["column"] for m in _classify_columns(columns)
    ]
    return [c for c in targets if c in columns]


def _build_join(base: Path, joins: list[dict]) -> tuple[pd.DataFrame, list[str], list[dict], bool]:
    """Join the base file with up to MAX_JOINS others.

    Masking happens once, after the merges: joining on a PII column is a normal
    thing to want (link records by patient id, customer id, name) and the keys
    have to hold their real values for the merge to match anything. The values
    never leave the server — the masked columns, keys included, are replaced
    before this frame reaches display or caller code.
    """
    if len(joins) > MAX_JOINS:
        raise ValueError(f"at most {MAX_JOINS} joins at a time")

    frame = _load(base)
    pii = set(_pii_columns_of(base, list(frame.columns)))
    applied: list[dict] = []
    truncated = False

    for spec in joins:
        right_path = _resolve(str(spec.get("file", "")))
        right = _load(right_path)
        right_pii = _pii_columns_of(right_path, list(right.columns))
        left_on = str(spec.get("left_on", ""))
        right_on = str(spec.get("right_on", ""))
        how = str(spec.get("how", "left")).lower()

        if how not in JOIN_HOWS:
            raise ValueError(f"join type must be one of {', '.join(JOIN_HOWS)}")
        if left_on not in frame.columns:
            raise ValueError(f"'{left_on}' is not a column of the current table")
        if right_on not in right.columns:
            raise ValueError(f"'{right_on}' is not a column of {right_path.name}")

        suffix = "_" + (re.sub(r"[^a-z0-9]+", "", right_path.stem.lower())[:12] or "r")
        frame = frame.merge(right, how=how, left_on=left_on, right_on=right_on, suffixes=("", suffix))
        pii |= {
            c for c in frame.columns
            if c in right_pii or (c.endswith(suffix) and c[: -len(suffix)] in right_pii)
        }
        applied.append({"file": str(right_path), "name": right_path.name, "left_on": left_on,
                        "right_on": right_on, "how": how, "rows": len(frame)})

        if len(frame) > MAX_JOIN_ROWS:
            frame = frame.head(MAX_JOIN_ROWS)
            truncated = True

    redacted = sorted(pii & set(frame.columns))
    for col in redacted:
        frame[col] = frame[col].where(frame[col].isna(), REDACTION_MASK)
    return frame, redacted, applied, truncated


def _clip(s: str) -> str:
    return s if len(s) <= MAX_OUTPUT_CHARS else s[:MAX_OUTPUT_CHARS] + "\n...[truncated]"


def _exec_eda(
    p: Path,
    code: str,
    pii_columns: list[str] | None,
    redact: bool,
    frame: pd.DataFrame | None = None,
    frame_redacted: list[str] | None = None,
) -> tuple[str, object, list[str]]:
    """Redact PII, then run caller code. Returns (stdout, last-expression value, redacted columns).

    A prebuilt frame (a join, say) can be passed in already masked, in which case
    it is used as `df` untouched.
    """
    if frame is not None:
        return _run_code(code, frame, frame_redacted or [])

    df = _load(p)

    redacted: list[str] = []
    if redact:
        if pii_columns is not None:
            targets = pii_columns
        else:
            override = _pii_override(p)
            targets = override if override is not None else [
                m["column"] for m in _classify_columns(list(df.columns))
            ]
        for col in targets:
            if col in df.columns:
                df[col] = df[col].where(df[col].isna(), REDACTION_MASK)
                redacted.append(col)

    return _run_code(code, df, redacted)


def _run_code(code: str, df: pd.DataFrame, redacted: list[str]) -> tuple[str, object, list[str]]:
    """Guarded exec of caller pandas against an already-masked frame."""
    tree = ast.parse(code, mode="exec")
    _check_code(tree)

    # notebook-style: last bare expression becomes the result
    last_expr = None
    show_name = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last_expr = ast.Expression(tree.body.pop().value)
    elif (
        tree.body
        and isinstance(tree.body[-1], ast.Assign)
        and len(tree.body[-1].targets) == 1
        and isinstance(tree.body[-1].targets[0], ast.Name)
    ):
        # code that ends on an assignment: show what it just built rather than
        # nothing — a model asked for a grouped table often stops one line short
        show_name = tree.body[-1].targets[0].id

    env = {"__builtins__": SAFE_BUILTINS, "df": df, "pd": pd, "np": np}
    stdout = io.StringIO()
    result = None
    with contextlib.redirect_stdout(stdout):
        exec(compile(tree, "<eda>", "exec"), env)
        if last_expr is not None:
            result = eval(compile(last_expr, "<eda>", "eval"), env)
        elif show_name:
            result = env.get(show_name)
    return stdout.getvalue(), result, redacted


@mcp.tool()
def run_eda(
    path: str,
    code: str,
    pii_columns: list[str] | None = None,
    redact: bool = True,
    joins: list[dict] | None = None,
) -> dict:
    """Run pandas EDA code against a CSV file with a PII guardrail.

    The file is preloaded as DataFrame `df` (pandas as `pd`, numpy as `np`).
    PII columns are replaced with "***" BEFORE the code runs, so their raw
    values can never appear in the output. Never attempt to print PII values.
    The value of the last expression is returned like in a notebook cell.

    Args:
        path: CSV file path.
        code: pandas code operating on `df` (no imports, no file I/O).
        pii_columns: explicit PII column names to redact; when omitted, the
            user-defined list saved for the file is used, else name-based
            auto-detection.
        redact: set False only if the caller is authorized to see raw values.
        joins: up to 4 other files to join in first, each
            {"file": path, "left_on": col, "right_on": col,
             "how": "left"|"inner"|"right"|"outer"}. `df` is then the joined
            table and every file's own PII columns are masked. A masked column
            works as a join key: it matches on real values and stays masked.
    """
    p = _resolve(path)
    frame = frame_pii = None
    if joins:
        frame, frame_pii, _, _ = _build_join(p, joins)
    stdout, result, redacted = _exec_eda(
        p, code, pii_columns, redact, frame=frame, frame_redacted=frame_pii
    )
    return {
        "stdout": _clip(stdout),
        "result": _clip("" if result is None else str(result)),
        "redacted_columns": redacted,
    }


# --- table rendering for visualize ---


def _to_frame(result: object) -> tuple[pd.DataFrame, int]:
    """Normalize an EDA result to a DataFrame; returns (frame, number of leading index columns)."""
    if isinstance(result, pd.DataFrame):
        df = result
    elif isinstance(result, pd.Series):
        df = result.to_frame(name=result.name if result.name is not None else "value")
    elif isinstance(result, pd.Index):
        return pd.DataFrame({result.name or "value": list(result)}), 0
    else:
        return pd.DataFrame({"value": [result]}), 0

    # a named or non-positional index carries meaning (groupby keys) — promote it to columns
    names = list(df.index.names)
    if any(n is not None for n in names) or not isinstance(df.index, pd.RangeIndex):
        return df.reset_index(), df.index.nlevels
    return df, 0


def _fmt(v: object) -> str:
    # .unique() and friends put arrays in cells; show the values, not the repr
    if isinstance(v, (list, tuple, set, np.ndarray)) or isinstance(v, pd.api.extensions.ExtensionArray):
        return ", ".join(_fmt(x) for x in list(v)[:20]) or "—"
    if isinstance(v, float):
        if not np.isfinite(v):
            return "—" if np.isnan(v) else str(v)
        if v.is_integer() and abs(v) < 1e15:
            return f"{v:,.0f}"
        return f"{v:,.6g}"
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return f"{v:,}"
    if v is None or v is pd.NaT:
        return "—"
    try:
        if pd.isna(v):
            return "—"
    except (TypeError, ValueError):
        pass
    return str(v)


def _bar_scale(s: pd.Series) -> float | None:
    """Max value to scale magnitude bars against, or None when bars would be noise."""
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return None
    vals = s.dropna()
    if len(vals) < 2 or vals.nunique() < 2 or (vals < 0).any():
        return None
    top = float(vals.max())
    # near-constant columns (years, ids) would render as uniformly full bars
    if top <= 0 or (top - float(vals.min())) / top < 0.1:
        return None
    return top


def _render_table(
    df: pd.DataFrame, n_index_cols: int, redacted: list[str], bar_columns: list[str] | None = None
) -> str:
    numeric = {c: pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]) for c in df.columns}
    scales = {
        c: _bar_scale(df[c]) if (bar_columns is None or c in bar_columns) else None
        for c in df.columns
    }

    head = []
    for i, c in enumerate(df.columns):
        tag = '<span class="tag">redacted</span>' if c in redacted else ""
        cls = "num" if numeric[c] and i >= n_index_cols else ""
        head.append(f'<th class="{cls}">{html.escape(str(c))}{tag}</th>')

    rows = []
    for _, row in df.iterrows():
        cells = []
        for i, c in enumerate(df.columns):
            v = row[c]
            text = _fmt(v)
            safe = html.escape(text)
            if i < n_index_cols:
                cells.append(f'<td class="idx" title="{safe}">{safe}</td>')
            elif numeric[c]:
                scale = scales[c]
                pct = min(abs(float(v)) / scale * 100, 100) if scale and pd.notna(v) else 0
                if pct > 0:
                    inner = f'<div class="cell"><span>{safe}</span><div class="bar" style="width:{pct:.1f}%"></div></div>'
                else:
                    inner = safe if text != "—" else '<span class="nan">—</span>'
                cells.append(f'<td class="num">{inner}</td>')
            else:
                body = safe if text != "—" else '<span class="nan">—</span>'
                cells.append(f'<td title="{safe}">{body}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")

    return (
        "<table><thead><tr>" + "".join(head) + "</tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )


MARKDOWN_MAX_ROWS = 100  # keeps a preview from blowing a small model's context
# chat clients render markdown tables in a fixed-width bubble with no horizontal
# scroll, so a wide preview spills out of the container — keep it narrow and let
# the report URL carry the full width
MARKDOWN_MAX_COLS = 6


def _markdown_table(df: pd.DataFrame, page: int, page_size: int, max_cols: int = MARKDOWN_MAX_COLS) -> tuple[str, int]:
    """Render one page of the frame as markdown; returns (markdown, page count)."""
    window = max(1, min(page_size, MARKDOWN_MAX_ROWS))
    n_pages = max(1, -(-len(df) // window))
    page = min(max(1, page), n_pages)
    start = (page - 1) * window
    chunk = df.iloc[start:start + window]

    cols = list(df.columns[:max_cols])
    lines = [
        "| " + " | ".join(str(c) for c in cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in chunk.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c]).replace("|", "\\|") for c in cols) + " |")

    notes = []
    if n_pages > 1:
        notes.append(
            f"rows {start + 1}–{start + len(chunk)} of {len(df)} (page {page} of {n_pages}) — "
            f"call visualize again with page={page + 1 if page < n_pages else 1} for the next page"
        )
    if len(df.columns) > max_cols:
        notes.append(f"showing {max_cols} of {len(df.columns)} columns — open the report URL for all of them")
    if notes:
        lines.append(f"\n_{'; '.join(notes)}_")
    return "\n".join(lines), n_pages


@mcp.tool()
def visualize(
    path: str,
    code: str = "df",
    title: str = "",
    pii_columns: list[str] | None = None,
    redact: bool = True,
    max_rows: int = 1000,
    bar_columns: list[str] | None = None,
    page: int = 1,
    page_size: int = 25,
    preview_columns: int = MARKDOWN_MAX_COLS,
    joins: list[dict] | None = None,
) -> dict:
    """Render the result of pandas EDA code as a styled HTML table page and return its URL.

    Same execution and PII guardrail as run_eda: PII columns are replaced with
    "***" before the code runs. The last expression must produce a
    DataFrame or Series (e.g. a groupby aggregation, value_counts or describe) —
    that becomes the table. Numeric columns get magnitude bars; groupby keys are
    shown as leading columns.

    Returns the report URL plus a markdown preview for inline display. The report
    page paginates all rendered rows client-side (prev/next, rows-per-page); the
    markdown preview holds one page, so call again with page=2, 3, ... to walk
    through the rest inline.

    Args:
        path: CSV file path.
        code: pandas code whose last expression is the table to render. Defaults
            to "df" — the whole file, which is what "show me the table" wants.
        title: heading for the report (defaults to the file name).
        pii_columns: explicit PII column names to redact (see run_eda).
        redact: set False only if the caller is authorized to see raw values.
        max_rows: rows carried into the report and paged through (default 1000).
        bar_columns: columns that get magnitude bars. Omit to auto-pick every
            varying non-negative numeric column; pass an explicit list to keep
            bars off columns where length is misleading (ranks, years, ids), or
            an empty list for a plain table.
        page: which page the markdown preview shows and the report opens on.
        page_size: rows per page (default 25; the markdown preview caps at 100).
        preview_columns: columns in the markdown preview (default 6). Chat clients
            render markdown tables without horizontal scroll, so a wide preview
            overflows its container — the report URL is where the full width goes.
        joins: up to 4 other files to join in first, each
            {"file": path, "left_on": col, "right_on": col,
             "how": "left"|"inner"|"right"|"outer"}. `df` is then the joined
            table, with every file's own PII columns masked.
    """
    p = _resolve(path)
    frame = frame_pii = None
    if joins:
        frame, frame_pii, _, _ = _build_join(p, joins)
    stdout, result, redacted = _exec_eda(
        p, code, pii_columns, redact, frame=frame, frame_redacted=frame_pii
    )
    if result is None:
        raise ValueError(
            "nothing to visualize: the last statement must be an expression producing a "
            "DataFrame or Series (a trailing print() returns nothing)"
        )

    frame, n_index_cols = _to_frame(result)
    n_total = len(frame)
    shown = frame.head(max_rows)
    heading = title or p.name
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    footer = [
        f"{len(redacted)} PII column(s) redacted before execution: {', '.join(redacted)}"
        if redacted else "No PII columns redacted for this file."
    ]
    if n_total > len(shown):
        footer.append(f"Showing the first {len(shown)} of {n_total} rows.")
    footer.append("Generated by csv-analyst — values in redacted columns never leave the server.")

    markdown, n_pages = _markdown_table(shown, page, page_size, max(1, preview_columns))

    doc = (TEMPLATES / "report.html").read_text()
    doc = doc.replace("__TITLE__", html.escape(heading))
    doc = doc.replace(
        "__SUBTITLE__",
        f"<code>{html.escape(str(p))}</code> &middot; {n_total} rows &times; "
        f"{len(frame.columns)} columns &middot; {stamp}",
    )
    doc = doc.replace("__TABLE__", _render_table(shown, n_index_cols, redacted, bar_columns))
    doc = doc.replace("__FOOTER__", "".join(f"<p>{html.escape(line)}</p>" for line in footer))
    doc = doc.replace("__PAGE__", str(max(1, page)))
    doc = doc.replace("__PAGE_SIZE__", str(max(1, page_size)))

    slug = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")[:40] or "report"
    name = f"{slug}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.html"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / name).write_text(doc)

    return {
        "url": f"{PUBLIC_BASE_URL}/reports/{name}" if PUBLIC_BASE_URL else f"/reports/{name}",
        "markdown": markdown,
        "n_rows": n_total,
        "n_columns": len(frame.columns),
        "page": min(max(1, page), n_pages),
        "page_size": page_size,
        "n_pages": n_pages,
        "redacted_columns": redacted,
        "stdout": _clip(stdout),
    }


@mcp.tool()
def export_table(
    path: str,
    code: str = "df",
    joins: list[dict] | None = None,
    format: str = "csv",
    columns: list[str] | None = None,
    sort: str = "",
    dir: str = "asc",
) -> dict:
    """Write the result of an EDA view to a CSV or Excel file and return its URL.

    Same guardrail as run_eda: PII columns are masked before the code runs, so the
    exported file carries "***" wherever a column is masked — an export can never
    contain values the tools would not show. The file is written to the server's
    exports directory and served over the same host as the reports; treat that as
    a shared location rather than a private one.

    Args:
        path: CSV file path.
        code: pandas code whose last expression is the table to export (default
            "df" — the whole file).
        joins: other files to join in first (see run_eda).
        format: "csv" or "xlsx".
        columns: restrict the export to these columns, in this order.
        sort: column to sort by before exporting.
        dir: "asc" or "desc".
    """
    p = _resolve(path)
    frame, redacted, label = _view_frame(p, {
        "code": code, "joins": joins or [], "columns": columns or [], "sort": sort, "dir": dir,
    })
    truncated = len(frame) > MAX_EXPORT_ROWS
    frame = frame.head(MAX_EXPORT_ROWS)

    fmt = "xlsx" if str(format).lower() in ("xlsx", "excel") else "csv"
    name = f"{p.stem}-{label}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.{fmt}"
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    target = EXPORTS_DIR / name
    if fmt == "xlsx":
        frame.to_excel(target, index=False, sheet_name=p.stem[:28] or "data")
    else:
        frame.to_csv(target, index=False)

    return {
        "url": f"{PUBLIC_BASE_URL}/exports/{name}" if PUBLIC_BASE_URL else f"/exports/{name}",
        "format": fmt,
        "n_rows": len(frame),
        "n_columns": len(frame.columns),
        "columns": [str(c) for c in frame.columns],
        "redacted_columns": [c for c in redacted if c in frame.columns],
        "truncated": truncated,
    }


# --- local LLM / GPU ---

def _vram() -> list[dict] | None:
    """GPU memory per device, or None when the container cannot see a GPU."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    gpus = []
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 3 or not parts[1].isdigit():
            continue
        used, total = int(parts[1]), int(parts[2])
        gpus.append({
            "name": parts[0],
            "used_mib": used,
            "free_mib": total - used,
            "total_mib": total,
            "used_pct": round(100 * used / total, 1) if total else None,
        })
    return gpus or None


def _swap_running() -> list[dict]:
    import httpx

    if not LLAMA_SWAP_URL:
        raise ValueError("no llama-swap endpoint (set LLAMA_SWAP_URL, or LOCAL_LLM_URL in .env)")
    r = httpx.get(f"{LLAMA_SWAP_URL}/running", timeout=10)
    r.raise_for_status()
    return [
        {"model": m.get("model"), "state": m.get("state"), "idle_ttl_seconds": m.get("ttl")}
        for m in (r.json().get("running") or [])
    ]


@mcp.tool()
def llm_status() -> dict:
    """Report GPU memory and which local models are currently loaded.

    Worth checking before a long local-LLM step, or to decide whether to call llm_stop:
    VRAM is shared with everything else on the machine, so a model left loaded can be
    what stops the next one from fitting. "gpus" is null when the container has no GPU
    visibility — "loaded" still answers the question in that case.
    """
    gpus = _vram()
    loaded = _swap_running()
    return {
        "endpoint": LLAMA_SWAP_URL,
        "configured_model": LOCAL_LLM_MODEL,
        "gpus": gpus,
        "loaded": loaded,
        "n_loaded": len(loaded),
    }


@mcp.tool()
def llm_stop() -> dict:
    """Unload every model llama-swap is holding, freeing GPU VRAM immediately.

    llama-swap already unloads a model once its idle TTL expires; this is the "give the
    VRAM back now" button. It applies to the whole llama-swap instance, so it also stops
    models loaded by other clients — the return value names what was stopped.
    """
    import httpx

    stopped = _swap_running()
    httpx.get(f"{LLAMA_SWAP_URL}/unload", timeout=60).raise_for_status()
    return {
        "stopped": [m["model"] for m in stopped],
        "still_loaded": [m["model"] for m in _swap_running()],
        "gpus": _vram(),
    }


# --- webui ---

@mcp.custom_route("/exports/{name}", methods=["GET"])
async def export_file(request: Request):
    f = EXPORTS_DIR / Path(request.path_params["name"]).name
    if f.suffix not in (".csv", ".xlsx") or not f.is_file():
        return JSONResponse({"error": "no such export"}, status_code=404)
    media = ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
             if f.suffix == ".xlsx" else "text/csv; charset=utf-8")
    return Response(
        content=f.read_bytes(),
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{f.name}"'},
    )


@mcp.custom_route("/", methods=["GET"])
async def ui(_: Request):
    return HTMLResponse((TEMPLATES / "index.html").read_text())


@mcp.custom_route("/reports/{name}", methods=["GET"])
async def report(request: Request):
    f = REPORTS_DIR / Path(request.path_params["name"]).name
    if f.suffix != ".html" or not f.is_file():
        return HTMLResponse("<h1>404 — no such report</h1>", status_code=404)
    return HTMLResponse(f.read_text())


@mcp.custom_route("/api/files", methods=["GET"])
async def api_files(_: Request):
    cfg = _config()
    out = []
    for f in _known_files():
        try:
            columns = _header(Path(f))
            override = cfg["pii"].get(f)
            if override is not None:
                pii, source = [{"column": c, "pii_type": "user_defined"} for c in override], "manual"
            else:
                pii, source = _classify_columns(columns), "auto"
            out.append({
                "path": f,
                "n_columns": len(columns),
                "n_pii": len(pii),
                "source": source,
                "pii_columns": [m["column"] for m in pii],
                "pii_types": sorted({m["pii_type"] for m in pii}),
            })
        except Exception as e:
            out.append({"path": f, "error": str(e)})
    return JSONResponse({
        "files": out,
        "sources": cfg["sources"],
        "mounted": [str(d) for d in _mounted_sources()],
        "data_dir": str(DATA_DIR),
    })


@mcp.custom_route("/api/browse", methods=["GET"])
async def api_browse(request: Request):
    raw = request.query_params.get("path", "")
    if not raw:
        return JSONResponse({"roots": [str(r) for r in BROWSE_ROOTS]})
    p = Path(raw).resolve()
    if not any(p == r or p.is_relative_to(r) for r in BROWSE_ROOTS):
        return JSONResponse({"error": f"outside browse roots: {p}"}, status_code=400)
    if not p.is_dir():
        return JSONResponse({"error": f"not a directory: {p}"}, status_code=400)
    try:
        entries = sorted(p.iterdir())
    except PermissionError:
        return JSONResponse({"error": f"permission denied: {p}"}, status_code=400)
    at_root = any(p == r for r in BROWSE_ROOTS)
    return JSONResponse({
        "path": str(p),
        "parent": None if at_root else str(p.parent),
        "dirs": [e.name for e in entries if e.is_dir() and not e.name.startswith(".")],
        "csvs": [e.name for e in entries if e.is_file() and e.suffix.lower() == ".csv"],
    })



@mcp.custom_route("/api/sources", methods=["POST"])
async def api_sources(request: Request):
    body = await request.json()
    path = str(body.get("path", "")).strip()
    cfg = _config()
    if body.get("remove"):
        cfg["sources"] = [s for s in cfg["sources"] if s != path]
        _save_config(cfg)
        return JSONResponse({"ok": True})
    p = Path(path)
    if not p.is_absolute():
        return JSONResponse({"error": "path must be absolute"}, status_code=400)
    if not (p.is_dir() or (p.is_file() and p.suffix.lower() == ".csv")):
        return JSONResponse({"error": f"not a directory or .csv file visible to the server: {path}"}, status_code=400)
    if path not in cfg["sources"]:
        cfg["sources"].append(path)
        _save_config(cfg)
    return JSONResponse({"ok": True})


@mcp.custom_route("/api/columns", methods=["GET"])
async def api_columns(request: Request):
    try:
        p = _resolve(request.query_params.get("file", ""))
        columns = _header(p)
    except (ValueError, FileNotFoundError, StopIteration) as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    auto = _classify_columns(columns)
    override = _pii_override(p)
    selected = override if override is not None else [m["column"] for m in auto]
    return JSONResponse({
        "columns": columns,
        "auto": auto,
        "selected": selected,
        "source": "manual" if override is not None else "auto",
    })


@mcp.custom_route("/api/preview", methods=["GET"])
async def api_preview(request: Request):
    """One page of a CSV for the data viewer, with PII columns already masked."""
    q = request.query_params
    try:
        p = _resolve(q.get("file", ""))
        page = max(1, int(q.get("page", 1)))
        page_size = min(max(int(q.get("page_size", 50)), 1), 500)
    except (ValueError, FileNotFoundError) as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    sort_col = q.get("sort") or ""
    descending = q.get("dir", "asc") == "desc"
    # the row-number gutter sorts by position in the file; ascending is already
    # the natural order, so only descending needs to do anything
    by_row = sort_col == ROW_SORT_KEY

    delimiter = _sniff(p)
    n_rows = _record_count(p)
    n_pages = max(1, -(-n_rows // page_size))
    page = min(page, n_pages)
    start = (page - 1) * page_size

    # dtypes come from a sample: the viewer shows the file verbatim, the dtype is a hint
    sample = _load(p, nrows=200)
    kinds = {c: _column_kind(sample[c]) for c in sample.columns}

    if by_row and descending:
        # walk the same records, just take the window from the end
        end = max(n_rows - start, 0)
        begin = max(end - page_size, 0)
        with open(p, newline="", errors="replace") as f:
            reader = csv.reader(f, delimiter=delimiter)
            header = next(reader, [])
            window = list(islice(reader, begin, end))[::-1]
        row_numbers = list(range(end, begin, -1))
    elif sort_col and not by_row and sort_col in sample.columns:
        # sorting spans the whole file, so the window has to come after the sort
        full = pd.read_csv(p, sep=delimiter, dtype=str, keep_default_na=False,
                           on_bad_lines="skip")
        header = list(full.columns)
        order = _sort_key(full[sort_col], kinds.get(sort_col, "text")).sort_values(
            ascending=not descending, kind="stable", na_position="last"
        ).index[start:start + page_size]
        window = full.loc[order].values.tolist()
        row_numbers = [int(i) + 1 for i in order]
    else:
        # csv.reader walks records (quote-aware), so the window lands on real rows
        with open(p, newline="", errors="replace") as f:
            reader = csv.reader(f, delimiter=delimiter)
            header = next(reader, [])
            window = list(islice(reader, start, start + page_size))
        row_numbers = list(range(start + 1, start + len(window) + 1))
    override = _pii_override(p)
    pii = set(override if override is not None else
              [m["column"] for m in _classify_columns(header)])
    redacted = [c for c in header if c in pii]

    columns = [
        {
            "name": str(c),
            "dtype": str(sample[c].dtype) if c in sample.columns else "str",
            "pii": c in pii,
            "kind": kinds.get(c, "text"),
            "numeric": kinds.get(c) == "number",
        }
        for c in header
    ]
    rows = [
        [
            REDACTION_MASK if columns[i]["pii"] and (i < len(row) and row[i] != "")
            else (row[i] if i < len(row) and row[i] != "" else "—")
            for i in range(len(columns))
        ]
        for row in window
    ]
    return JSONResponse({
        "file": str(p),
        "columns": columns,
        "rows": rows,
        "row_numbers": row_numbers,
        "page": page,
        "page_size": page_size,
        "n_pages": n_pages,
        "n_rows": n_rows,
        "start": start,
        "sort": sort_col if (by_row or sort_col in sample.columns) else "",
        "dir": "desc" if descending else "asc",
        "redacted_columns": redacted,
        "pii_source": "manual" if override is not None else "auto",
    })


FILTER_PROMPT = (
    "You turn a question into pandas over a DataFrame named df, and the result is "
    "shown as a table.\n"
    "Rules:\n"
    "- Reply with code only: no explanation, no markdown fences, no imports.\n"
    "- Usually one line. If you need a couple of steps, the LAST line must be the "
    "expression to show.\n"
    "- The last expression must be a DataFrame or Series, never a plain number.\n"
    "- Do not use .query(), .eval(), file I/O, or any import.\n"
    "- Filtering: boolean masks like df[(df['a'] > 5) & (df['b'] == 'x')] — wrap each "
    "condition in parentheses.\n"
    "- Text matching: .str.contains('x', case=False, na=False).\n"
    "- Only some columns: df.loc[<mask>, ['A', 'B']].\n"
    "- Grouping with a condition on the aggregate (SQL HAVING) takes two lines, "
    "because .query() is not allowed:\n"
    "    g = df.groupby('col').agg(n=('a', 'size'), total=('b', 'sum')).reset_index()\n"
    "    g[g['n'] > 20]\n"
    "- NA / empty / missing / blank means a null value — use .isna() or .notna(). "
    "Compare against the text 'NA' only if the question says the literal text.\n"
    "- The table shows missing values as an em dash. It is a display placeholder, never a "
    "value: read \"col != —\" as col.notna() and \"col = —\" as col.isna(), and never "
    "compare a column to '—' or '-'.\n"
    "- Columns marked PII hold \"***\" instead of their real values: never filter on them.\n"
)


def _column_kind(s: pd.Series) -> str:
    """How a column should be sorted and labelled: number, date, bool or text."""
    if pd.api.types.is_bool_dtype(s):
        return "bool"
    if pd.api.types.is_numeric_dtype(s):
        return "number"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "date"
    values = s.dropna().astype(str).head(50)
    if len(values):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(values, errors="coerce", format="mixed")
        if parsed.notna().mean() > 0.8:
            return "date"
    return "text"


BLANKS = {"", "na", "n/a", "nan", "none", "null", "-", "—"}


def _sort_key(col: pd.Series, kind: str) -> pd.Series:
    """Sort text as text but numbers as numbers — the frame is read as strings to
    keep values verbatim, so '9' must not sort above '10'.

    Blanks become NaN so na_position keeps them out of the ordering entirely:
    read verbatim they are empty strings, which would otherwise sort ahead of
    every real value in ascending order instead of sitting at the end.
    """
    if kind == "number":
        return pd.to_numeric(col, errors="coerce")
    if kind == "date":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pd.to_datetime(col, errors="coerce", format="mixed")

    text = col.astype(str).str.strip()
    blank = text.str.lower().isin(BLANKS)
    if kind == "bool":
        return text.str.lower().map({"true": 1, "false": 0}).mask(blank)
    return text.str.lower().mask(blank)


JOIN_INTENT = re.compile(r"\b(join|joined|merge|combine|enrich|lookup|look up|together with)\b", re.I)

JOIN_PLAN_PROMPT = (
    "Decide which files to join to answer a question about a base table.\n"
    "Reply with JSON only, no explanation:\n"
    '{"joins": [{"file": "<exact path from the list>", "left_on": "<base column>", '
    '"right_on": "<column in that file>", "how": "left"}]}\n'
    "Rules:\n"
    "- Return {\"joins\": []} if the question can be answered from the base table alone.\n"
    "- At most 4 joins. how is one of left, inner, right, outer — prefer left.\n"
    "- left_on must exist in the base table and right_on in the joined file.\n"
    "- Join on the column the two tables share, usually the one with the same meaning.\n"
)


def _plan_join(question: str, base: Path, limit: int = 12) -> list[dict]:
    """Ask the local model which files to join for a question. Returns a validated
    spec (invalid files or columns are dropped, so a bad guess costs nothing)."""
    base_cols = _header(base)
    candidates = [f for f in _known_files() if f != str(base)][:limit]
    if not candidates:
        return []

    catalogue = "\n".join(
        f"- {f}\n    columns: {', '.join(_header(Path(f))[:30])}" for f in candidates
    )
    content, reasoning, _ = _llm_chat(
        f"{JOIN_PLAN_PROMPT}\nBase table: {base}\nBase columns: {', '.join(base_cols[:60])}\n\n"
        f"Other files:\n{catalogue}\n\nQuestion: {question}",
        max_tokens=800,
    )
    raw = content or reasoning
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return []
    try:
        plan = json.loads(raw[start:end + 1]).get("joins", [])
    except json.JSONDecodeError:
        return []

    valid: list[dict] = []
    for spec in plan[:MAX_JOINS]:
        try:
            right = _resolve(str(spec.get("file", "")))
        except (ValueError, FileNotFoundError):
            continue
        left_on, right_on = str(spec.get("left_on", "")), str(spec.get("right_on", ""))
        how = str(spec.get("how", "left")).lower()
        if left_on in base_cols and right_on in _header(right) and how in JOIN_HOWS:
            valid.append({"file": str(right), "left_on": left_on, "right_on": right_on, "how": how})
    return valid


def _display_rows(frame: pd.DataFrame, n_index_cols: int = 0) -> tuple[list[dict], list[list[str]]]:
    """Frame -> (column metadata, display strings) in the shape the viewer expects."""
    columns = [
        {
            "name": str(c),
            "dtype": str(frame[c].dtype),
            "pii": False,
            "kind": _column_kind(frame[c]),
            "numeric": bool(
                pd.api.types.is_numeric_dtype(frame[c]) and not pd.api.types.is_bool_dtype(frame[c])
            ),
        }
        for c in frame.columns
    ]
    rows = [[_fmt(v) for v in row] for row in frame.itertuples(index=False, name=None)]
    return columns, rows


def _paged_frame_response(
    frame: pd.DataFrame, p: Path, redacted: list[str], body: dict, extra: dict
) -> JSONResponse:
    """Page, sort and format a built frame the way the viewer expects."""
    page = max(1, int(body.get("page", 1)))
    page_size = min(max(int(body.get("page_size", 50)), 1), 500)
    sort_col = str(body.get("sort") or "")
    descending = body.get("dir", "asc") == "desc"

    try:
        row_numbers = [int(i) + 1 for i in frame.index]
    except (TypeError, ValueError):
        row_numbers = None

    if sort_col == ROW_SORT_KEY:
        order = frame.index.sort_values(ascending=not descending)
    elif sort_col in frame.columns:
        order = _sort_key(frame[sort_col], _column_kind(frame[sort_col])).sort_values(
            ascending=not descending, kind="stable", na_position="last"
        ).index
    else:
        order = None
    if order is not None:
        frame = frame.loc[order]
        if row_numbers is not None:
            row_numbers = [int(i) + 1 for i in order]
    frame = frame.reset_index(drop=True)

    n_rows = len(frame)
    n_pages = max(1, -(-n_rows // page_size))
    page = min(page, n_pages)
    start = (page - 1) * page_size
    columns, rows = _display_rows(frame.iloc[start:start + page_size])
    for col in columns:
        col["pii"] = col["name"] in redacted

    return JSONResponse({
        "file": str(p),
        "columns": columns,
        "rows": rows,
        "row_numbers": row_numbers[start:start + page_size] if row_numbers else None,
        "page": page,
        "page_size": page_size,
        "n_pages": n_pages,
        "n_rows": n_rows,
        "start": start,
        "sort": sort_col,
        "dir": "desc" if descending else "asc",
        "redacted_columns": redacted,
        "pii_source": "manual" if _pii_override(p) is not None else "auto",
        **extra,
    })


MAX_EXPORT_ROWS = 200_000


def _masked_file_frame(p: Path) -> tuple[pd.DataFrame, list[str]]:
    frame = _load(p)
    redacted = _pii_columns_of(p, list(frame.columns))
    for col in redacted:
        frame[col] = frame[col].where(frame[col].isna(), REDACTION_MASK)
    return frame, redacted


def _view_frame(p: Path, body: dict) -> tuple[pd.DataFrame, list[str], str]:
    """Rebuild what the viewer is currently showing — joins, question and sort —
    as a whole frame rather than a page. Masked exactly as on screen."""
    joins = body.get("joins") or []
    code = (body.get("code") or "").strip()
    if code == "df":  # the default "whole file" code is not a filter
        code = ""

    if joins:
        frame, redacted, _, _ = _build_join(p, joins)
        label = "joined"
    else:
        frame, redacted = _masked_file_frame(p)
        label = "table"

    if code:
        _, result, _ = _run_code(code, frame, redacted)
        if isinstance(result, pd.Series):
            result = result.to_frame(name=result.name if result.name is not None else "value")
        if not isinstance(result, pd.DataFrame):
            raise ValueError("the current view is not a table")
        frame = result
        if not isinstance(frame.index, pd.RangeIndex) and not pd.api.types.is_integer_dtype(frame.index):
            frame = frame.reset_index()
        label = "filtered"

    sort_col = str(body.get("sort") or "")
    if sort_col and sort_col != ROW_SORT_KEY and sort_col in frame.columns:
        frame = frame.loc[
            _sort_key(frame[sort_col], _column_kind(frame[sort_col])).sort_values(
                ascending=body.get("dir", "asc") != "desc", kind="stable", na_position="last"
            ).index
        ]
    elif sort_col == ROW_SORT_KEY:
        frame = frame.sort_index(ascending=body.get("dir", "asc") != "desc")

    # only the columns on screen, in the order the viewer shows them
    chosen = [c for c in (body.get("columns") or []) if c in frame.columns]
    if chosen:
        frame = frame[chosen]
    return frame.reset_index(drop=True), redacted, label


@mcp.custom_route("/api/export", methods=["POST"])
async def api_export(request: Request):
    """Download the current view as CSV or Excel, masked the same way it is shown."""
    body = await request.json()
    try:
        p = _resolve(str(body.get("file", "")))
        frame, redacted, label = _view_frame(p, body)
    except (ValueError, FileNotFoundError) as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=422)

    truncated = len(frame) > MAX_EXPORT_ROWS
    frame = frame.head(MAX_EXPORT_ROWS)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    fmt = "xlsx" if str(body.get("format", "csv")).lower() in ("xlsx", "excel") else "csv"
    name = f"{p.stem}-{label}-{stamp}.{fmt}"

    buffer = io.BytesIO()
    if fmt == "xlsx":
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            frame.to_excel(writer, index=False, sheet_name=p.stem[:28] or "data")
        media = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        buffer.write(frame.to_csv(index=False).encode("utf-8-sig"))  # BOM: Excel opens it cleanly
        media = "text/csv; charset=utf-8"

    return Response(
        content=buffer.getvalue(),
        media_type=media,
        headers={
            "Content-Disposition": f'attachment; filename="{name}"',
            "X-Export-Rows": str(len(frame)),
            "X-Export-Truncated": "1" if truncated else "0",
            "X-Export-Redacted": ", ".join(c for c in redacted if c in frame.columns),
            "Access-Control-Expose-Headers": "Content-Disposition, X-Export-Rows, X-Export-Truncated, X-Export-Redacted",
        },
    )


@mcp.custom_route("/api/join", methods=["POST"])
async def api_join(request: Request):
    """Join the current file with up to four others and page through the result.

    Each file contributes its own PII list, so a column masked in the file it
    came from stays masked here. Nothing is written to disk.
    """
    body = await request.json()
    try:
        p = _resolve(str(body.get("file", "")))
        frame, redacted, applied, truncated = _build_join(p, body.get("joins") or [])
    except (ValueError, FileNotFoundError) as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=422)

    return _paged_frame_response(frame, p, redacted, body, {
        "joins": applied,
        "truncated": truncated,
        "file_rows": _record_count(p),
    })


@mcp.custom_route("/api/query", methods=["POST"])
async def api_query(request: Request):
    """Filter a file from a natural-language question, or re-run generated code.

    The local LLM writes the pandas; it runs under the same guardrail as the MCP
    tools (PII masked before execution, AST checks). Nothing is written to disk —
    pass `code` back to page through the same result without re-asking the model.
    """
    import httpx

    body = await request.json()
    try:
        p = _resolve(str(body.get("file", "")))
        page = max(1, int(body.get("page", 1)))
        page_size = min(max(int(body.get("page_size", 50)), 1), 500)
    except (ValueError, FileNotFoundError) as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    code = (body.get("code") or "").strip()
    question = (body.get("question") or "").strip()

    # questions can run over a join, which is where the versatile part lives:
    # group/aggregate/HAVING-style conditions across the joined columns
    joins = body.get("joins") or []
    planned = False
    # "join X with Y" in the question sets the join up; the client adopts the
    # spec that comes back, so the join panel and its storage stay in step
    if question and not joins and LOCAL_LLM_URL and JOIN_INTENT.search(question):
        joins = _plan_join(question, p)
        planned = bool(joins)

    joined_frame = joined_pii = None
    joined_applied: list[dict] = []
    if joins:
        try:
            joined_frame, joined_pii, joined_applied, _ = _build_join(p, joins)
        except (ValueError, FileNotFoundError) as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    if not code:
        if not question:
            return JSONResponse({"error": "ask a question or pass code"}, status_code=400)
        if not LOCAL_LLM_URL:
            return JSONResponse(
                {"error": "natural-language queries need LOCAL_LLM_URL set in .env"},
                status_code=400,
            )
        if joined_frame is not None:
            sample, pii = joined_frame.head(200), set(joined_pii or [])
        else:
            sample = _load(p, nrows=200)
            override = _pii_override(p)
            pii = set(override if override is not None else
                      [m["column"] for m in _classify_columns(list(sample.columns))])
        cols = "\n".join(
            f"- {c} ({sample[c].dtype}){' [PII]' if c in pii else ''}" for c in sample.columns
        )
        # without this the model reads the other file itself when the question
        # says "join with X.csv" — the merge is already done by the time it runs
        already = (
            "The join is ALREADY DONE: df below is the joined table, including the "
            f"columns from {', '.join(Path(j['file']).name for j in joined_applied)}. "
            "Do not merge and do not read any file — just use df.\n"
            if joined_applied else ""
        )
        try:
            content, reasoning, finish = _llm_chat(
                f"{FILTER_PROMPT}{already}\nColumns:\n{cols}\n\nQuestion: {question}"
            )
        except httpx.HTTPError as e:
            return JSONResponse({"error": f"local LLM unreachable: {e}"}, status_code=502)

        # a reasoning model can spend the whole budget thinking and answer with
        # nothing — the expression is usually in the reasoning, so look there too
        code = _extract_code(content) or _extract_code(reasoning)
        if not code:
            detail = (
                "the model ran out of tokens while reasoning and never wrote the code"
                if finish == "length" else
                f"the model replied: {(content or reasoning or '(nothing)')[:200]}"
            )
            return JSONResponse(
                {"error": f"could not read pandas out of the model reply — {detail}"},
                status_code=422,
            )

    try:
        _, result, redacted = _exec_eda(
            p, code, None, True, frame=joined_frame, frame_redacted=joined_pii
        )
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}", "code": code}, status_code=422)
    if result is None:
        return JSONResponse({"error": "that code returned nothing to show", "code": code}, status_code=422)

    if isinstance(result, pd.DataFrame):
        frame = result
    elif isinstance(result, pd.Series):
        frame = result.to_frame(name=result.name if result.name is not None else "value")
    else:
        frame = pd.DataFrame({"value": [result]})

    # an integer index means row positions (a filter) — keep those for the gutter.
    # anything else is data, e.g. groupby keys, and has to become a column or the
    # answer loses the very thing it grouped by.
    if isinstance(frame.index, pd.MultiIndex) or not pd.api.types.is_integer_dtype(frame.index):
        frame = frame.reset_index()
        row_numbers = None
    else:
        try:
            row_numbers = [int(i) + 1 for i in frame.index]
        except (TypeError, ValueError):
            row_numbers = None

    sort_col = str(body.get("sort") or "")
    descending = body.get("dir", "asc") == "desc"
    if sort_col == ROW_SORT_KEY:
        order = frame.index.sort_values(ascending=not descending)
        frame = frame.loc[order]
        if row_numbers is not None:
            row_numbers = [int(i) + 1 for i in order]
    elif sort_col in frame.columns:
        order = _sort_key(frame[sort_col], _column_kind(frame[sort_col])).sort_values(
            ascending=not descending, kind="stable", na_position="last"
        ).index
        frame = frame.loc[order]
        if row_numbers is not None:
            row_numbers = [int(i) + 1 for i in order]
    frame = frame.reset_index(drop=True)

    n_rows = len(frame)
    n_pages = max(1, -(-n_rows // page_size))
    page = min(page, n_pages)
    start = (page - 1) * page_size
    columns, rows = _display_rows(frame.iloc[start:start + page_size])
    for col in columns:
        col["pii"] = col["name"] in redacted

    return JSONResponse({
        "file": str(p),
        "code": code,
        "question": question,
        "joins": joined_applied,
        "joins_planned": planned,
        "columns": columns,
        "rows": rows,
        "row_numbers": row_numbers[start:start + page_size] if row_numbers else None,
        "page": page,
        "page_size": page_size,
        "n_pages": n_pages,
        "n_rows": n_rows,
        "file_rows": _record_count(p),
        "start": start,
        "redacted_columns": redacted,
        "pii_source": "manual" if _pii_override(p) is not None else "auto",
    })


@mcp.custom_route("/api/pii", methods=["POST"])
async def api_pii(request: Request):
    body = await request.json()
    try:
        p = _resolve(str(body.get("file", "")))
    except (ValueError, FileNotFoundError) as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    cfg = _config()
    if body.get("reset"):
        cfg["pii"].pop(str(p), None)
    else:
        cfg["pii"][str(p)] = [str(c) for c in body.get("columns", [])]
    _save_config(cfg)
    return JSONResponse({"ok": True})


if __name__ == "__main__":
    import uvicorn
    from starlette.middleware.cors import CORSMiddleware
    from mcp.server.transport_security import TransportSecuritySettings

    # served on localhost, LAN and tailnet hostnames — the SDK's default
    # localhost-only Host allowlist would 421 tailnet requests
    app = mcp.streamable_http_app(
        stateless_http=True,
        json_response=True,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )
    # browser-based MCP clients (e.g. llama.cpp webui) need CORS on /mcp
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Mcp-Session-Id"],
    )
    uvicorn.run(app, host="0.0.0.0", port=PORT)
