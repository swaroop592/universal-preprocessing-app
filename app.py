
import io
import re
import json
import csv
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from rapidfuzz import process, fuzz
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Universal Data Preprocessing + Viz (MVP++)", layout="wide")

# ----------------------------
# Helpers
# ----------------------------
DEFAULT_MISSING_TOKENS = {
    "", "na", "n/a", "nan", "null", "none", "-", "--", "—", " ", "missing", "not available"
}

# "Metadata block" words seen in exports like Arable. We ONLY treat these as "bad header"
# when the row looks like key/value metadata (few non-empty cells), not a real wide header row.
HEADER_KEYWORDS_BAD = {
    "title", "date", "date range", "site", "latitude", "longitude", "number of records",
    "temp", "pressure", "speed", "precip", "soil moisture"
}

HEADER_HINT_TOKENS = {
    "time", "date", "utc", "timestamp", "site", "temp", "pressure", "rh", "wind",
    "moisture", "salinity", "sensor", "station"
}

def normalize_colname(c) -> str:
    c = "" if c is None else str(c)
    c = c.strip().lower()
    c = re.sub(r"[^\w\s]", " ", c)
    c = re.sub(r"\s+", "_", c).strip("_")
    return c if c else "unnamed"

def fix_duplicate_colnames(cols):
    seen = {}
    out = []
    for c in cols:
        base = c
        if base not in seen:
            seen[base] = 1
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}_{seen[base]}")
    return out

def is_id_like(name: str) -> bool:
    n = name.lower()
    return any(tok in n for tok in ["id", "code", "zip", "postal", "phone"])

def standardize_missing(df: pd.DataFrame, missing_tokens=None) -> pd.DataFrame:
    if missing_tokens is None:
        missing_tokens = DEFAULT_MISSING_TOKENS

    df2 = df.copy()
    df2 = df2.replace(r"^\s*$", np.nan, regex=True)

    for col in df2.columns:
        s = df2[col]
        if s.dtype == "object" or pd.api.types.is_string_dtype(s):
            s_norm = s.astype(str).str.strip()
            mask = s_norm.str.lower().isin(missing_tokens)
            df2.loc[mask, col] = np.nan
    return df2

def guess_column_types(df: pd.DataFrame, force_numeric=None, force_datetime=None, force_categorical=None):
    force_numeric = set(force_numeric or [])
    force_datetime = set(force_datetime or [])
    force_categorical = set(force_categorical or [])

    numeric_cols, datetime_cols, categorical_cols = [], [], []

    for col in df.columns:
        if col in force_categorical:
            categorical_cols.append(col)
            continue
        if col in force_numeric:
            numeric_cols.append(col)
            continue
        if col in force_datetime:
            datetime_cols.append(col)
            continue

        s = df[col]

        # universal safety: don't treat IDs/codes as numeric
        if is_id_like(col):
            categorical_cols.append(col)
            continue

        if pd.api.types.is_numeric_dtype(s):
            numeric_cols.append(col)
            continue

        # datetime guess
        if (s.dtype == "object" or pd.api.types.is_string_dtype(s)) and s.notna().mean() > 0.2:
            parsed = pd.to_datetime(s, errors="coerce")
            if parsed.notna().mean() > 0.6:
                datetime_cols.append(col)
                continue

        # numeric-ish strings
        if s.dtype == "object" or pd.api.types.is_string_dtype(s):
            s2 = pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")
            if s2.notna().mean() > 0.7:
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
        else:
            categorical_cols.append(col)

    datetime_cols = [c for c in datetime_cols if c not in numeric_cols]
    categorical_cols = [c for c in categorical_cols if c not in numeric_cols and c not in datetime_cols]
    return numeric_cols, datetime_cols, categorical_cols

def convert_types(df: pd.DataFrame, numeric_cols, datetime_cols):
    df2 = df.copy()
    for col in numeric_cols:
        if df2[col].dtype == "object" or pd.api.types.is_string_dtype(df2[col]):
            df2[col] = df2[col].astype(str).str.replace(",", "", regex=False)
        df2[col] = pd.to_numeric(df2[col], errors="coerce")

    for col in datetime_cols:
        df2[col] = pd.to_datetime(df2[col], errors="coerce")

    return df2

def high_missing_cols(df: pd.DataFrame, threshold: float):
    miss = df.isna().mean()
    flagged = miss[miss > threshold].sort_values(ascending=False)
    return flagged

def drop_cols(df: pd.DataFrame, cols_to_drop):
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    return df.drop(columns=cols_to_drop).copy()

def align_to_schema(df: pd.DataFrame, schema_cols: list[str], min_score: int = 80):
    src_cols = list(df.columns)
    mapping = {}
    used_src = set()

    for target in schema_cols:
        match = process.extractOne(target, src_cols, scorer=fuzz.WRatio)
        if match and match[1] >= min_score and match[0] not in used_src:
            mapping[match[0]] = target
            used_src.add(match[0])

    df2 = df.rename(columns=mapping).copy()

    for c in schema_cols:
        if c not in df2.columns:
            df2[c] = np.nan

    remaining = [c for c in df2.columns if c not in schema_cols]
    df2 = df2[schema_cols + remaining]

    unmatched_source = [c for c in src_cols if c not in mapping.keys()]
    unmatched_schema = [c for c in schema_cols if c not in df.columns and c not in mapping.values()]
    return df2, mapping, unmatched_schema, unmatched_source

def df_to_download_bytes(df: pd.DataFrame, fmt: str, export_missing_as: str):
    if fmt == "CSV":
        return (
            df.to_csv(index=False, na_rep=export_missing_as).encode("utf-8"),
            "text/csv",
            "cleaned_dataset.csv",
        )
    if fmt == "XLSX":
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="cleaned")
        return (
            bio.getvalue(),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "cleaned_dataset.xlsx",
        )
    if fmt == "PARQUET":
        bio = io.BytesIO()
        df.to_parquet(bio, index=False)
        return bio.getvalue(), "application/octet-stream", "cleaned_dataset.parquet"
    raise ValueError("Unknown format")

def download_plot(fig, filename="plot.png"):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    st.download_button("Download plot (PNG)", data=buf.getvalue(), file_name=filename, mime="image/png")

# ----------------------------
# Robust CSV/XLSX import
# ----------------------------
def header_row_score(row_vals: list[str]) -> float:
    vals = [("" if v is None else str(v)).strip() for v in row_vals]
    non_empty = [v for v in vals if v != "" and v.lower() != "nan"]

    if len(non_empty) < 2:
        return -1e9

    joined = " ".join(non_empty).lower()

    # IMPORTANT FIX:
    # Only penalize metadata keywords if the row looks like key/value metadata (few non-empty cells)
    if any(k in joined for k in HEADER_KEYWORDS_BAD) and len(non_empty) <= 4:
        return -1e6 + len(non_empty)

    def looks_like_colname(v):
        v2 = v.lower()
        if re.fullmatch(r"[-+]?\d+(\.\d+)?", v2 or ""):
            return False
        if re.search(r"[a-zA-Z_]", v2) is None:
            return False
        return True

    colname_like = sum(looks_like_colname(v) for v in non_empty)
    uniqueness = len(set(non_empty)) / max(1, len(non_empty))

    hint_hits = 0
    for v in non_empty:
        lv = v.lower()
        if any(tok in lv for tok in HEADER_HINT_TOKENS):
            hint_hits += 1

    score = 0.0
    score += 3.0 * colname_like
    score += 2.0 * hint_hits
    score += 5.0 * uniqueness
    score += 0.5 * len(non_empty)
    return score

def parse_quality(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return -1e9
    nrows, ncols = df.shape
    if ncols <= 1:
        return -1e8 + nrows
    empty_col_frac = (df.isna().all()).mean() if ncols else 1.0
    unnamed = sum(str(c).lower().startswith("unnamed") for c in df.columns)
    return (ncols * 3.0) + (min(nrows, 5000) * 0.01) - (empty_col_frac * 50.0) - (unnamed * 2.0)

def detect_delimiter_by_quality(text: str):
    """
    IMPORTANT FIX:
    Instead of scoring only a preview's shape, we score the *best parsed result quality*
    for each delimiter. This is much more reliable for exports that have a 2-column metadata
    section followed by a real wide table.
    """
    candidates = [",", "\t", ";", "|"]
    best_sep = ","
    best_score = -1e18

    for sep in candidates:
        try:
            preview = pd.read_csv(
                io.StringIO(text),
                sep=sep,
                engine="python",
                header=None,
                nrows=80,
                on_bad_lines="skip",
            )
        except Exception:
            continue

        # Try top header candidates quickly
        scores = []
        for r in range(min(len(preview), 40)):
            row_vals = preview.iloc[r].tolist()
            non_empty = sum((str(v).strip() not in ["", "nan"]) for v in row_vals)
            score = header_row_score(row_vals) + (non_empty * 2.0)
            scores.append((r, score))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        header_candidates = [r for r, _ in scores[:5]]

        # Attempt parses and take best quality
        best_local = -1e18
        for hdr in header_candidates:
            try:
                df_try = pd.read_csv(
                    io.StringIO(text),
                    sep=sep,
                    engine="python",
                    header=hdr,
                    skip_blank_lines=True,
                    on_bad_lines="skip",
                )
                best_local = max(best_local, parse_quality(df_try))
            except Exception:
                continue

        if best_local > best_score:
            best_score = best_local
            best_sep = sep

    return best_sep

def read_csv_robust(file, encoding="utf-8", sep=None, force_header_row=None):
    raw = file.getvalue()
    text = raw.decode(encoding, errors="replace")

    if sep is None:
        sep = detect_delimiter_by_quality(text)

    # Read preview safely
    preview = pd.read_csv(
        io.StringIO(text),
        sep=sep,
        engine="python",
        header=None,
        nrows=80,
        on_bad_lines="skip"
    )

    # Candidate header rows
    if force_header_row is not None:
        candidates = [max(0, int(force_header_row) - 1)]  # user gives 1-based
    else:
        scores = []
        for r in range(min(len(preview), 60)):
            row_vals = preview.iloc[r].tolist()
            non_empty = sum((str(v).strip() not in ["", "nan"]) for v in row_vals)
            score = header_row_score(row_vals) + (non_empty * 2.0)
            scores.append((r, score))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        candidates = [r for r, _ in scores[:8]]

    best_df = None
    best_meta = None
    best_score = -1e18

    for hdr in candidates:
        try:
            df_try = pd.read_csv(
                io.StringIO(text),
                sep=sep,
                engine="python",
                header=hdr,
                skip_blank_lines=True,
                on_bad_lines="skip",
            )
        except Exception:
            continue

        sc = parse_quality(df_try)
        if sc > best_score:
            best_score = sc
            best_df = df_try
            best_meta = {"sep": sep, "header_row": hdr + 1, "on_bad_lines": "skip"}

    if best_df is None:
        raise ValueError("Could not parse CSV even with robust import. Try changing encoding or delimiter.")

    return best_df, best_meta

def read_xlsx_robust(file, force_header_row=None):
    xls = pd.ExcelFile(file)
    sheet = xls.sheet_names[0]
    preview = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=80)

    if force_header_row is not None:
        candidates = [max(0, int(force_header_row) - 1)]
    else:
        scores = []
        for r in range(min(len(preview), 60)):
            row_vals = preview.iloc[r].tolist()
            scores.append((r, header_row_score(row_vals)))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        candidates = [r for r, _ in scores[:8]]

    best_df = None
    best_meta = None
    best_score = -1e18

    for hdr in candidates:
        try:
            df_try = pd.read_excel(xls, sheet_name=sheet, header=hdr)
        except Exception:
            continue
        sc = parse_quality(df_try)
        if sc > best_score:
            best_score = sc
            best_df = df_try
            best_meta = {"sheet": sheet, "header_row": hdr + 1}

    if best_df is None:
        raise ValueError("Could not parse XLSX. Try setting header row manually.")

    return best_df, best_meta

def read_uploaded_file(file, encoding="utf-8", sep=None, force_header_row=None):
    name = file.name.lower()
    if name.endswith(".csv"):
        return read_csv_robust(file, encoding=encoding, sep=sep, force_header_row=force_header_row)
    if name.endswith(".xlsx"):
        return read_xlsx_robust(file, force_header_row=force_header_row)
    raise ValueError("Unsupported file type (only CSV/XLSX supported)")

# ----------------------------
# UI
# ----------------------------
st.title("Universal Data Preprocessing + Visualization (MVP++)")
st.caption("Upload → combine → robust import → clean → (optional) schema/label → visualize → download.")

with st.sidebar:
    st.subheader("Session")
    if st.button("Reset session"):
        st.session_state.clear()
        st.rerun()

    st.subheader("Upload")
    uploaded_files = st.file_uploader(
        "Upload dataset(s) (CSV or XLSX)",
        type=["csv", "xlsx"],
        accept_multiple_files=True,
        key="file_uploader",
    )

    combine_mode = st.selectbox(
        "Combine mode",
        ["Single file (first)", "Append rows (stack files)", "Join on keys (merge files)"],
        index=0,
    )

    add_source_file = st.checkbox("Add _source_file column when appending", value=True)

    st.subheader("Advanced import")
    override_header = st.checkbox("Manually override header row", value=False)
    header_row_manual = st.number_input(
        "Header row (1-based)", min_value=1, value=1, step=1, disabled=not override_header
    )

    # IMPORTANT FIX: if override delimiter is enabled, do NOT offer AUTO
    override_delim = st.checkbox("Manually override delimiter (CSV only)", value=False)
    if override_delim:
        delim_choice = st.selectbox(
            "Delimiter (one character)",
            ["COMMA (,)", "TAB (\\t)", "SEMICOLON (;)", "PIPE (|)"],
            index=0,
        )
    else:
        delim_choice = "AUTO"

    def delim_from_choice(choice: str):
        if choice.startswith("COMMA"):
            return ","
        if choice.startswith("TAB"):
            return "\t"
        if choice.startswith("SEMICOLON"):
            return ";"
        if choice.startswith("PIPE"):
            return "|"
        return None

    encoding = st.selectbox("Encoding (CSV only)", ["utf-8", "latin-1"], index=0)

if not uploaded_files:
    st.info("Upload one or more CSV/XLSX files to begin.")
    st.stop()

force_header = int(header_row_manual) if override_header else None
sep = delim_from_choice(delim_choice) if override_delim else None

# Read all files robustly
dfs_raw = []
read_notes = []

for f in uploaded_files:
    df_tmp, meta = read_uploaded_file(
        f,
        encoding=encoding,
        sep=sep,
        force_header_row=force_header
    )
    dfs_raw.append((f.name, df_tmp))

    note = f"{f.name}: header row = {meta.get('header_row')}"
    if "sep" in meta:
        note += f", sep='{meta['sep']}'"
    if "sheet" in meta:
        note += f", sheet='{meta['sheet']}'"
    if meta.get("on_bad_lines"):
        note += f" | Used on_bad_lines='{meta['on_bad_lines']}' (some rows may be skipped)."
    read_notes.append(note)

if read_notes:
    st.info("Auto-import details:\n\n" + "\n".join(read_notes))

# Combine / select base
if combine_mode == "Single file (first)":
    df_raw = dfs_raw[0][1].copy()
elif combine_mode == "Append rows (stack files)":
    normed = []
    for name, d in dfs_raw:
        d2 = d.copy()
        d2.columns = fix_duplicate_colnames([normalize_colname(c) for c in d2.columns])
        if add_source_file:
            d2["_source_file"] = name
        normed.append(d2)
    df_raw = pd.concat(normed, ignore_index=True, sort=False)
else:
    normed = []
    for name, d in dfs_raw:
        d2 = d.copy()
        d2.columns = fix_duplicate_colnames([normalize_colname(c) for c in d2.columns])
        normed.append((name, d2))

    _, merged = normed[0]
    common_cols = set(merged.columns)
    for _, d in normed[1:]:
        common_cols &= set(d.columns)
    common_cols = sorted(list(common_cols))

    if not common_cols:
        st.error("No common columns found to join on. Use Append rows instead.")
        st.stop()

    with st.sidebar:
        join_keys = st.multiselect("Join keys", common_cols, default=common_cols[:1])
        join_how = st.selectbox("Join type", ["inner", "left", "right", "outer"], index=0)

    for name, d in normed[1:]:
        merged = merged.merge(d, on=join_keys, how=join_how, suffixes=("", f"_{normalize_colname(name)}"))
    df_raw = merged

# Normalize for processing
df = df_raw.copy()
df.columns = fix_duplicate_colnames([normalize_colname(c) for c in df.columns])

report = {
    "input_files": [n for n, _ in dfs_raw],
    "combine_mode": combine_mode,
    "input_shape": list(df_raw.shape),
    "steps": []
}

# Layout
left, right = st.columns([1, 1])

with right:
    st.subheader("Raw preview (combined/original)")
    st.write(f"Shape: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
    st.dataframe(df_raw.head(50), use_container_width=True)

with left:
    st.subheader("Pipeline mode")
    mode = st.radio("Choose mode", ["ML Ready (Impute)", "Keep Missing (Do not impute)"], index=0)

    st.subheader("Missing tokens")
    missing_tokens_text = st.text_area(
        "Missing tokens (comma or newline separated)",
        value=", ".join(sorted(DEFAULT_MISSING_TOKENS)),
        height=90
    )
    missing_tokens = set([t.strip().lower() for t in re.split(r"[,\n]+", missing_tokens_text) if t.strip()])

    st.subheader("Cleaning settings")
    standardize_na = st.checkbox("Standardize missing tokens → NaN", value=True)

    missing_threshold = st.slider("Drop columns with missing fraction >", 0.0, 0.95, 0.70, 0.05)

    st.subheader("Safety controls")
    high_missing_action = st.radio("High-missing column action", ["Drop columns", "Do NOT drop; only flag"], index=0)
    protected_cols = st.multiselect("Protected columns (never drop)", options=df.columns.tolist(), default=[])

    # OPTIONAL (universal, but default OFF): duplicates drop
    drop_dups = st.checkbox("Drop duplicate rows (optional)", value=False)

    st.subheader("Type overrides (advanced)")
    st.caption("Defaults are auto-inferred. Override if the app misclassifies a column.")
    force_numeric = st.multiselect("Treat as numeric", options=df.columns.tolist(), default=[])
    force_datetime = st.multiselect(
        "Treat as datetime",
        options=[c for c in df.columns.tolist() if c not in force_numeric],
        default=[]
    )
    force_categorical = st.multiselect(
        "Force categorical (never numeric/datetime)",
        options=[c for c in df.columns.tolist() if c not in force_numeric and c not in force_datetime],
        default=[]
    )

    st.subheader("Missing-value fill policies (ML Ready only)")
    numeric_strategy = st.selectbox("Numeric strategy", ["median", "mean", "constant", "keep_missing"], index=0)
    numeric_fill_value = None
    if numeric_strategy == "constant":
        numeric_fill_value = st.number_input("Numeric constant fill value", value=0.0)

    categorical_strategy = st.selectbox("Categorical strategy", ["most_frequent", "constant", "keep_missing"], index=0)
    categorical_fill_value = None
    if categorical_strategy == "constant":
        categorical_fill_value = st.text_input("Categorical constant fill value", value="Unknown")

    st.subheader("Schema / label alignment")
    use_schema = st.checkbox("Align columns to a schema template", value=False)
    if use_schema:
        st.caption("Example schema: local_site_time, utc_time, site, ... , label (optional)")
    schema_text = st.text_area("Schema columns (comma-separated)", value="", disabled=not use_schema, height=90)
    schema_min_score = st.slider("Schema match strictness (fuzzy score)", 50, 95, 80, 1, disabled=not use_schema)

    st.subheader("Download")
    export_missing_as = st.text_input("Export missing as (CSV only)", value="N/A")
    fmt = st.selectbox("Download format", ["CSV", "XLSX", "PARQUET"], index=0)

# ----------------------------
# Pipeline execution
# ----------------------------
df_work = df.copy()

if standardize_na:
    df_work = standardize_missing(df_work, missing_tokens=missing_tokens)
    report["steps"].append({"standardize_missing_tokens": True, "missing_tokens": sorted(list(missing_tokens))})

num_cols, dt_cols, cat_cols = guess_column_types(
    df_work,
    force_numeric=force_numeric,
    force_datetime=force_datetime,
    force_categorical=force_categorical
)
df_work = convert_types(df_work, num_cols, dt_cols)
report["steps"].append({"type_inference": {"numeric": num_cols, "datetime": dt_cols, "categorical": cat_cols}})

# Always drop fully empty columns (safe)
empty_cols = df_work.columns[df_work.isna().all()].tolist()
if empty_cols:
    df_work = df_work.drop(columns=empty_cols)
    report["steps"].append({"dropped_all_missing_cols": empty_cols})

# Optional duplicates drop (default OFF)
if drop_dups:
    before = len(df_work)
    df_work = df_work.drop_duplicates()
    removed = before - len(df_work)
    report["steps"].append({"drop_duplicates": int(removed)})

# High-missing handling
flagged = high_missing_cols(df_work, missing_threshold)
flagged_cols = flagged.index.tolist()

# Never drop protected columns
flagged_drop_candidates = [c for c in flagged_cols if c not in protected_cols]

if high_missing_action == "Drop columns":
    df_work = drop_cols(df_work, flagged_drop_candidates)
    report["steps"].append({"high_missing_action": {"action": "drop", "threshold": missing_threshold, "dropped": flagged_drop_candidates}})
else:
    report["steps"].append({"high_missing_action": {"action": "flag_only", "threshold": missing_threshold, "flagged": flagged_cols}})

# Recompute types after dropping columns
num_cols, dt_cols, cat_cols = guess_column_types(
    df_work,
    force_numeric=force_numeric,
    force_datetime=force_datetime,
    force_categorical=force_categorical
)

# Impute
if mode.startswith("ML Ready"):
    if numeric_strategy in ("median", "mean", "most_frequent"):
        if num_cols:
            imp_num = SimpleImputer(strategy=numeric_strategy)
            df_work[num_cols] = imp_num.fit_transform(df_work[num_cols])
    elif numeric_strategy == "constant":
        if num_cols:
            for col in num_cols:
                s = df_work[col]
                fv = int(numeric_fill_value) if pd.api.types.is_integer_dtype(s.dtype) else float(numeric_fill_value)
                imp = SimpleImputer(strategy="constant", fill_value=fv)
                df_work[[col]] = imp.fit_transform(df_work[[col]])

    if categorical_strategy == "most_frequent":
        if cat_cols:
            imp_cat = SimpleImputer(strategy="most_frequent")
            df_work[cat_cols] = imp_cat.fit_transform(df_work[cat_cols])
    elif categorical_strategy == "constant":
        if cat_cols:
            imp_cat = SimpleImputer(strategy="constant", fill_value=categorical_fill_value)
            df_work[cat_cols] = imp_cat.fit_transform(df_work[cat_cols])

    report["steps"].append({
        "imputation": {
            "numeric": numeric_strategy if numeric_strategy != "constant" else f"constant({numeric_fill_value})",
            "categorical": categorical_strategy if categorical_strategy != "constant" else f"constant({categorical_fill_value})"
        }
    })
else:
    report["steps"].append({"imputation": "skipped_keep_missing_mode"})

# Schema alignment
schema_cols = []
label_in_schema = False
label_name = None
schema_mapping = {}
unmatched_schema = []
unmatched_source = []

if use_schema:
    schema_cols = fix_duplicate_colnames([normalize_colname(x) for x in schema_text.split(",") if x.strip()])
    if schema_cols:
        df_work, schema_mapping, unmatched_schema, unmatched_source = align_to_schema(df_work, schema_cols, min_score=schema_min_score)
        report["steps"].append({
            "schema_alignment": {
                "schema_cols": schema_cols,
                "min_score": schema_min_score,
                "mapping_source_to_schema": schema_mapping,
                "unmatched_schema": unmatched_schema,
                "unmatched_source": unmatched_source,
            }
        })
        for candidate in ("label", "target", "y"):
            if candidate in schema_cols:
                label_in_schema = True
                label_name = candidate
                break

# Label definition
if use_schema and label_in_schema and label_name:
    st.subheader(f"Label definition (schema field: '{label_name}')")
    label_mode = st.radio(
        f"How should '{label_name}' be defined?",
        ["Leave empty (None)", "Use existing column", "Create from rule", "Create from mapping (category → value)"],
        index=0,
    )

    if label_mode == "Use existing column":
        src = st.selectbox("Select source column", [c for c in df_work.columns if c != label_name])
        df_work[label_name] = df_work[src]
        report["steps"].append({"label_definition": {"mode": "use_existing", "source": src}})

    elif label_mode == "Create from rule":
        rule_col = st.selectbox("Column for rule", [c for c in df_work.columns if c != label_name])
        rule_type = st.selectbox("Rule type", ["Numeric threshold", "Exact match (string/number)"], index=0)

        pos_val = st.text_input("Value if rule is TRUE", value="1")
        neg_val = st.text_input("Value if rule is FALSE", value="0")

        if rule_type == "Numeric threshold":
            op = st.selectbox("Operator", [">=", ">", "<=", "<", "=="], index=0)
            threshold = st.number_input("Threshold", value=0.0)

            s_num = pd.to_numeric(df_work[rule_col], errors="coerce")
            if op == ">=":
                mask = s_num >= threshold
            elif op == ">":
                mask = s_num > threshold
            elif op == "<=":
                mask = s_num <= threshold
            elif op == "<":
                mask = s_num < threshold
            else:
                mask = s_num == threshold

            df_work[label_name] = np.where(mask, pos_val, neg_val)
            report["steps"].append({
                "label_definition": {
                    "mode": "rule_numeric_threshold",
                    "column": rule_col,
                    "operator": op,
                    "threshold": threshold,
                    "true_value": pos_val,
                    "false_value": neg_val,
                }
            })
        else:
            match_value = st.text_input("Match value", value="")
            s = df_work[rule_col].astype(str).str.strip()
            mask = s == str(match_value).strip()
            df_work[label_name] = np.where(mask, pos_val, neg_val)
            report["steps"].append({
                "label_definition": {
                    "mode": "rule_exact_match",
                    "column": rule_col,
                    "match_value": match_value,
                    "true_value": pos_val,
                    "false_value": neg_val,
                }
            })

    elif label_mode == "Create from mapping (category → value)":
        map_col = st.selectbox("Column to map", [c for c in df_work.columns if c != label_name])
        uniq = pd.Series(df_work[map_col].astype(str).str.strip().unique()).sort_values().tolist()
        st.write("Unique values (first 30):", uniq[:30])
        if len(uniq) > 30:
            st.info(f"Showing first 30 of {len(uniq)} unique values to keep UI fast.")

        mapping = {}
        for v in uniq[:30]:
            mapping[v] = st.text_input(f"Map '{v}' →", value="")

        default_unmapped = st.text_input("Value for unmapped/missing (default None)", value="")

        if st.button("Apply mapping to label"):
            s = df_work[map_col].astype(str).str.strip()
            y = s.map(mapping)
            if default_unmapped != "":
                y = y.fillna(default_unmapped)
            df_work[label_name] = y

            report["steps"].append({
                "label_definition": {
                    "mode": "mapping",
                    "source_column": map_col,
                    "mapping_preview_first_30": mapping,
                    "default_unmapped": default_unmapped if default_unmapped != "" else None,
                }
            })

    if label_name in df_work.columns:
        cols = [c for c in df_work.columns if c != label_name] + [label_name]
        df_work = df_work[cols]
        st.write("Label distribution:")
        st.dataframe(df_work[label_name].value_counts(dropna=False).to_frame("count"), use_container_width=True)

# ----------------------------
# Reporting
# ----------------------------
st.subheader("Cleaning summary")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Rows (current)", df_work.shape[0])
with c2:
    st.metric("Columns (current)", df_work.shape[1])
with c3:
    st.metric("Fully-empty cols dropped", len(empty_cols))
with c4:
    st.metric("Missing cells (current)", int(df_work.isna().sum().sum()))

if len(flagged_cols) > 0:
    st.write("High-missing columns (flagged)")
    st.dataframe(flagged.to_frame("missing_fraction"), use_container_width=True)
    if high_missing_action == "Drop columns" and len(flagged_drop_candidates) > 0:
        st.warning(f"Dropped (high-missing > {missing_threshold:.2f}) excluding protected: {flagged_drop_candidates}")
    elif high_missing_action != "Drop columns":
        st.info("High-missing columns were NOT dropped (flag-only mode).")

miss_after = (df_work.isna().mean().sort_values(ascending=False) * 100).round(2)
st.write("Missing % by column (after current steps)")
st.dataframe(miss_after.to_frame("missing_%"), use_container_width=True)

st.subheader("Cleaned preview")
st.dataframe(df_work.head(50), use_container_width=True)

# ----------------------------
# Visualizations
# ----------------------------
st.subheader("Visualizations")
viz_df_choice = st.radio("Use dataset", ["Cleaned", "Raw"], horizontal=True, key="viz_df_choice")
viz_df = df_work if viz_df_choice == "Cleaned" else df

num_cols = viz_df.select_dtypes(include="number").columns.tolist()
dt_cols = viz_df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
cat_cols = [c for c in viz_df.columns if c not in num_cols and c not in dt_cols]

st.markdown("### Column selection")
col_mode = st.radio("Selection mode", ["Single column", "X/Y", "Multiple columns (batch)"], horizontal=True)

x_col = y_col = None
multi_cols = []

if col_mode == "Single column":
    x_col = st.selectbox("Column", viz_df.columns.tolist())
elif col_mode == "X/Y":
    x_col = st.selectbox("X column", viz_df.columns.tolist())
    y_col = st.selectbox("Y column", [c for c in viz_df.columns.tolist() if c != x_col])
else:
    multi_cols = st.multiselect("Columns", viz_df.columns.tolist(), default=viz_df.columns.tolist()[:3])

with st.expander("Appearance / grouping (optional)"):
    palette = st.selectbox("Palette", ["tab10", "tab20", "Set2", "Dark2", "Accent"], index=0)
    color_by = None
    if cat_cols:
        cb = st.selectbox("Color/group by (categorical)", ["(none)"] + cat_cols, index=0)
        color_by = None if cb == "(none)" else cb

cmap = plt.get_cmap(palette)

def is_numeric(c): return c in num_cols
def is_categorical(c): return c in cat_cols
def is_datetime(c): return c in dt_cols

choices = []
if col_mode == "Single column" and x_col:
    if is_numeric(x_col):
        choices = ["Histogram", "Box plot", "Missingness bar (all columns)"]
    else:
        choices = ["Bar (value counts)", "Missingness bar (all columns)"]
elif col_mode == "X/Y" and x_col and y_col:
    if is_numeric(x_col) and is_numeric(y_col):
        choices = ["Scatter", "Correlation (numeric pair)"]
    if (is_datetime(x_col) and is_numeric(y_col)) or (is_datetime(y_col) and is_numeric(x_col)):
        choices = list(dict.fromkeys(choices + ["Line (time series)"]))
    if (is_categorical(x_col) and is_numeric(y_col)) or (is_categorical(y_col) and is_numeric(x_col)):
        choices = list(dict.fromkeys(choices + ["Box by category", "Bar (agg numeric by category)"]))
else:
    choices = ["Missingness bar (all columns)"]
    if any(is_numeric(c) for c in multi_cols):
        choices = ["Missingness bar (all columns)", "Correlation heatmap (numeric subset)"]

chart = st.selectbox("Chart type", choices)

if chart == "Histogram":
    if not is_numeric(x_col):
        st.info("Pick a numeric column.")
    else:
        bins = st.slider("Bins", 5, 100, 30)
        fig = plt.figure()
        plt.hist(viz_df[x_col].dropna(), bins=bins)
        plt.title(f"Histogram: {x_col}")
        st.pyplot(fig)
        download_plot(fig, f"hist_{x_col}.png")

elif chart == "Box plot":
    if not is_numeric(x_col):
        st.info("Pick a numeric column.")
    else:
        fig = plt.figure()
        plt.boxplot(viz_df[x_col].dropna(), vert=True)
        plt.title(f"Box plot: {x_col}")
        st.pyplot(fig)
        download_plot(fig, f"box_{x_col}.png")

elif chart == "Bar (value counts)":
    if x_col is None:
        st.info("Pick a column.")
    else:
        topn = st.slider("Top N categories", 5, 50, 15)
        vc = viz_df[x_col].astype(str).value_counts(dropna=False).head(topn)
        fig = plt.figure()
        plt.bar(vc.index.astype(str), vc.values)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Value counts: {x_col} (top {topn})")
        st.pyplot(fig)
        download_plot(fig, f"bar_{x_col}.png")

elif chart == "Scatter":
    if not (is_numeric(x_col) and is_numeric(y_col)):
        st.info("Pick two numeric columns.")
    else:
        fig = plt.figure()
        if color_by and color_by in viz_df.columns:
            groups = viz_df[color_by].astype(str).fillna("NA").unique().tolist()
            colors = [cmap(i % cmap.N) for i in range(len(groups))]
            for g, colr in zip(groups, colors):
                sub = viz_df[viz_df[color_by].astype(str).fillna("NA") == g]
                plt.scatter(sub[x_col], sub[y_col], label=g, alpha=0.8)
            plt.legend(title=color_by, bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            plt.scatter(viz_df[x_col], viz_df[y_col])
        plt.xlabel(x_col); plt.ylabel(y_col)
        plt.title(f"Scatter: {x_col} vs {y_col}")
        st.pyplot(fig)
        download_plot(fig, f"scatter_{x_col}_{y_col}.png")

elif chart == "Line (time series)":
    if is_datetime(x_col) and is_numeric(y_col):
        time_col, val_col = x_col, y_col
    elif is_datetime(y_col) and is_numeric(x_col):
        time_col, val_col = y_col, x_col
    else:
        st.info("Pick one datetime column and one numeric column.")
        time_col = None
        val_col = None

    if time_col:
        df_ts = viz_df[[time_col, val_col] + ([color_by] if color_by else [])].copy()
        df_ts[time_col] = pd.to_datetime(df_ts[time_col], errors="coerce")
        df_ts = df_ts.dropna(subset=[time_col])
        fig = plt.figure()
        if color_by:
            for g, sub in df_ts.groupby(color_by):
                sub = sub.sort_values(time_col)
                plt.plot(sub[time_col], sub[val_col], label=str(g))
            plt.legend(title=color_by, bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            df_ts = df_ts.sort_values(time_col)
            plt.plot(df_ts[time_col], df_ts[val_col])
        plt.xlabel(time_col); plt.ylabel(val_col)
        plt.title(f"Time series: {val_col} over {time_col}")
        st.pyplot(fig)
        download_plot(fig, f"line_{val_col}_over_{time_col}.png")

elif chart == "Correlation (numeric pair)":
    if not (is_numeric(x_col) and is_numeric(y_col)):
        st.info("Pick two numeric columns.")
    else:
        r = viz_df[[x_col, y_col]].corr(numeric_only=True).iloc[0, 1]
        st.metric("Pearson correlation", float(r) if pd.notna(r) else np.nan)

elif chart == "Box by category":
    if is_categorical(x_col) and is_numeric(y_col):
        cat, val = x_col, y_col
    elif is_categorical(y_col) and is_numeric(x_col):
        cat, val = y_col, x_col
    else:
        st.info("Pick one categorical and one numeric column.")
        cat = val = None

    if cat:
        topn = st.slider("Show top N categories", 5, 30, 15)
        counts = viz_df[cat].astype(str).value_counts(dropna=False).head(topn).index.tolist()
        df_box = viz_df[viz_df[cat].astype(str).isin(counts)][[cat, val]].copy()
        groups = [g for g, _ in df_box.groupby(cat)]
        data = [df_box[df_box[cat] == g][val].dropna().values for g in groups]
        fig = plt.figure()
        plt.boxplot(data, labels=[str(g) for g in groups], vert=True)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Box: {val} by {cat} (top {topn})")
        st.pyplot(fig)
        download_plot(fig, f"boxby_{val}_by_{cat}.png")

elif chart == "Bar (agg numeric by category)":
    if is_categorical(x_col) and is_numeric(y_col):
        cat, val = x_col, y_col
    elif is_categorical(y_col) and is_numeric(x_col):
        cat, val = y_col, x_col
    else:
        st.info("Pick one categorical and one numeric column.")
        cat = val = None

    if cat:
        agg = st.selectbox("Aggregation", ["mean", "median", "sum", "count"], index=0)
        topn = st.slider("Top N categories", 5, 30, 15)
        df_agg = viz_df[[cat, val]].copy()
        df_agg[cat] = df_agg[cat].astype(str).fillna("NA")

        if agg == "count":
            out = df_agg.groupby(cat)[val].count().sort_values(ascending=False).head(topn)
        elif agg == "sum":
            out = df_agg.groupby(cat)[val].sum().sort_values(ascending=False).head(topn)
        elif agg == "median":
            out = df_agg.groupby(cat)[val].median().sort_values(ascending=False).head(topn)
        else:
            out = df_agg.groupby(cat)[val].mean().sort_values(ascending=False).head(topn)

        fig = plt.figure()
        plt.bar(out.index.astype(str), out.values)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{agg}({val}) by {cat} (top {topn})")
        st.pyplot(fig)
        download_plot(fig, f"baragg_{agg}_{val}_by_{cat}.png")

elif chart == "Correlation heatmap (numeric subset)":
    cols = [c for c in multi_cols if c in num_cols]
    if len(cols) < 2:
        st.info("Select at least 2 numeric columns.")
    else:
        cols = cols[:30]
        corr = viz_df[cols].corr(numeric_only=True)
        fig = plt.figure()
        plt.imshow(corr.values)
        plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
        plt.yticks(range(len(cols)), cols)
        plt.title("Correlation heatmap")
        plt.colorbar()
        st.pyplot(fig)
        download_plot(fig, "corr_heatmap.png")

elif chart == "Missingness bar (all columns)":
    miss = (viz_df.isna().mean() * 100).sort_values(ascending=False)
    topn = st.slider("Show top N columns", 5, min(50, len(miss)), min(20, len(miss)))
    miss = miss.head(topn)
    fig = plt.figure()
    plt.bar(miss.index.astype(str), miss.values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Missing %")
    plt.title(f"Missingness (top {topn})")
    st.pyplot(fig)
    download_plot(fig, "missingness.png")

# ----------------------------
# Downloads
# ----------------------------
st.subheader("Download outputs")
data_bytes, mime, fname = df_to_download_bytes(df_work, fmt, export_missing_as=export_missing_as)
st.download_button("Download cleaned dataset", data=data_bytes, file_name=fname, mime=mime)

report_bytes = json.dumps(report, indent=2, default=str).encode("utf-8")
st.download_button("Download cleaning report (JSON)", data=report_bytes, file_name="cleaning_report.json", mime="application/json")
