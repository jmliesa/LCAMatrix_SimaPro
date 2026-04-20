# This is code to compute LCA results from SimaPro raw matrix export. Originally published by Xun Liao (former PhD
# candidate at EPFL, Switzerland) as a R code and modified by @CedricFurrer and @jmliesa as published here in Python.
# The motivation for this work can be multiple:
#       1- For teaching purposes, so LCA students can understand the logic behind LCA calculations.
#       2- As an alternative / double-check procedure of LCA calculations done in any other LCA software.
#       3- To handle large database at ease, and thus, potentially very useful if you want to compute selected
#       impact results for databases at once.

# This code can be adapted for any database exported from SimaPro. Note a Phd or developer license is required to
# export LCA databases in SimaPro.

# To better follow this code, we provide a summary on how LCA calculations work, based on the lectures from the
# Life Cycle Assessment course from NTNU and the book 'Methodological Essentials of Life Cycle Assessment' by
# Anders H. Strømman.


#%%
# Import packages

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve


#%%
# ---------------------------------------------------------
# Section 0: Paths
# ---------------------------------------------------------
# We compute, for the entire SimaPro matrix:
#
#   A      : technology / requirement matrix      (n_proc × n_proc)
#   S      : stressor matrix (elem. flows)       (n_flows × n_proc)
#   L      : Leontief inverse (I − A)^(-1)       (n_proc × n_proc)   [applied via linear solves]
#   E      : life cycle inventory (LCI)          (n_flows × n_proc)  = S · L
#   C      : characterization matrix              (n_flows × n_cats) (flows → impact categories)
#   D_pro  : impacts per process                  (n_proc × n_cats)  = Eᵀ · C
#
# where:
#   n_proc  = number of processes / products
#   n_flows = number of elementary flows (resources + emissions)
#   n_cats  = number of impact categories

working_dir = Path('/Users/.../Data').expanduser()

directory_data_1 = working_dir / "Matrix_to_Analyse"  # SimaPro matrix (A + S)
directory_data_2 = working_dir / "Method_CSV"         # SimaPro method export (CFs)
directory_data_3 = working_dir / "Results"            # Output impacts per process
directory_data_3.mkdir(parents=True, exist_ok=True)


def get_single_file(directory, pattern=None):
    """
    Returns the first regular file found in the directory.
    """
    if pattern is None:
        files = sorted(directory.iterdir())
    else:
        files = sorted(directory.glob(pattern))
    files = [f for f in files if f.is_file()]
    if not files:
        raise FileNotFoundError(f"No files found in {directory}")
    return files[0]



#%%
# ---------------------------------------------------------
# Section 1: Build A and S, then compute E = S · (I − A)^(-1)
# ---------------------------------------------------------

# Read matrix Excel file (technology + emissions for all processes)
matrix_file = get_single_file(directory_data_1, "*.xlsx")

# Sheet 1 is mandatory
matrix_df_1 = pd.read_excel(matrix_file, sheet_name=0, header=None)

# Sheet 2 is optional, whenever there's more processes than the maximum columns allowed in excel.
# Thus, if present, it adds more process columns to matrix_df.
try:
    matrix_df_2 = pd.read_excel(matrix_file, sheet_name=1, header=None)
    sheet2_exists = True
except Exception:
    sheet2_exists = False

# If a second sheet exists, append its process columns (from column 5 onward)
if sheet2_exists:
    # In R: data2[, 5:dim(data2)[2]] -> Python: data2.iloc[:, 4:]
    matrix_df = pd.concat([matrix_df_1, matrix_df_2.iloc[:, 4:]], axis=1)
else:
    matrix_df = matrix_df_1.copy()

nrow, ncol = matrix_df.shape  # full size of the SimaPro-like matrix


# ---------------------------------------------------------
# Reference output vector q (only because of SimaPro output matrix format)
# ---------------------------------------------------------
# Row 3 in the matrix (1-based) usually holds the reference outputs –
# i.e. how much product each column/process provides (often 1, but it can be also 0.001, etc for unit conversions).
#
# q_j (here 'q') is the reference output of process j.
# q is only needed because of how SimaPro exports the matrix.
# In standard matrix-LCA theory, (A) and (S) are already normalized, so no explicit q vector appears.
# Dimension: q : (n_proc,)
# ---------------------------------------------------------
q = matrix_df.iloc[2, 4:ncol].to_numpy(dtype=float)
q = np.nan_to_num(q, nan=0.0)  # treat missing as zero


# ---------------------------------------------------------
# Stressor matrix S (elementary flows × processes)
# ---------------------------------------------------------
# Block layout in the SimaPro export:
# - A (requirements) block: rows 7..(ncol+2)
# - S (elementary flows) block: rows (ncol+3)..nrow
# - Both A and S share the process columns 5..ncol.
#
# Here we first extract the *unnormalised* S_total, then divide
# each column j by q_j to obtain S (per unit output).
#
# S_total: absolute stressor amounts for the reference outputs q.
# S      : stressors per unit output (n_flows × n_proc).
# ---------------------------------------------------------
S_total_df = matrix_df.iloc[ncol + 2 : nrow, 4:ncol]
S_total = S_total_df.to_numpy(dtype=float)
S_total = np.nan_to_num(S_total, nan=0.0)

# Normalise to "per unit output": S_ij = S_total_ij / q_j
with np.errstate(divide="ignore", invalid="ignore"):
    S = S_total / q
S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
# Dimension: S : (n_flows × n_proc)
# n_flows = S.shape[0], n_proc = S.shape[1]

S_sparse = sparse.csr_matrix(S)


# ---------------------------------------------------------
# Technology / requirement matrix A (process × process)
# ---------------------------------------------------------
# Extract the technology block:
#   rows: 7 .. (ncol+2)   (1-based)  -> iloc[6 : ncol+2]
#   cols: 5 .. ncol       (1-based)  -> iloc[:, 4:ncol]
#
# A_total_ij is the total input of process i used to produce q_j of process j.
# We normalise by q_j to get a_ij = input from i per unit output of j.
# ---------------------------------------------------------
A_total_df = matrix_df.iloc[6 : ncol + 2, 4:ncol]
A_total = A_total_df.to_numpy(dtype=float)
A_total = np.nan_to_num(A_total, nan=0.0)

n_proc = A_total.shape[1]  # number of processes (columns of A)
# Check consistency: S.shape[1] == n_proc
assert S.shape[1] == n_proc, "S and A must have the same number of process columns"

# Normalise: A_ij = A_total_ij / q_j
with np.errstate(divide="ignore", invalid="ignore"):
    A = A_total / q
A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
# Dimension: A : (n_proc × n_proc)

# Identity matrix I (same size as A)
I_proc = sparse.identity(n_proc, format="csr")

# Build (I − A) as sparse matrix
A_leontief = I_proc - sparse.csr_matrix(A)
# Dimension: A_leontief : (n_proc × n_proc)


# ---------------------------------------------------------
# Compute E = S · (I − A)^(-1) y
# ---------------------------------------------------------
# ***** Note some matrices are here transposed, because of the SimaPro output matrices ******
# We want:
#   L  = (I − A)^(-1)      (Leontief inverse, not formed explicitly)
#   E  = S · L            (LCI matrix, n_flows × n_proc)
# ---------------------------------------------------------

# ---------------------------------------------------------
# Y: explicit demand matrix (to make it coherent with LCA classical notation)
# ***** Note vector y = functional unit is here not used and instead we use Y matrix since we aim to calculate all processes from all A martix ******

# Here we set Y = I, meaning:
# - column 1 = demand 1 unit of process 1
# - column 2 = demand 1 unit of process 2
# - ...
# So this reproduces the "all processes at once" result
# Shape: (n_proc × n_proc)
# ---------------------------------------------------------
Y = sparse.identity(n_proc, format="csr")


# ---------------------------------------------------------
# X = L Y = (I - A)^(-1) Y
# Instead of building L explicitly, solve:
#   (I - A) X = Y
# Shape:
#   X: (n_proc × n_proc)
# ---------------------------------------------------------
X = spsolve(A_leontief, Y.toarray())


# ---------------------------------------------------------
# E = S X
# Stressor contributions by process for all unit-demand columns in Y
# Shape:
#   S: (n_flows × n_proc)
#   X: (n_proc × n_proc)
#   E: (n_flows × n_proc)
# ---------------------------------------------------------
E = S @ X




#%%
# ---------------------------------------------------------
# Section 2: Build C (characterisation) and align with flows
# ---------------------------------------------------------
# In the summary:
#   C_{k,i} = CF for stressor i in impact category k.
#
# Here we keep the convention:
#   C has shape (n_flows × n_cats),
#   i.e. each row corresponds to an elementary flow,
#   each column to an impact category.
#
# Then for a process j, its LCI is column j of E (size n_flows),
# and impacts are:
#   d_j = E[:, j]^T · C      (1 × n_cats),
# which in matrix form becomes:
#   D_pro = E^T · C         (n_proc × n_cats).

method_file = get_single_file(directory_data_2)

encodings_to_try = ["utf-8", "cp1252", "latin1"]
for enc in encodings_to_try:
    try:
        with open(method_file, "r", encoding=enc) as f:
            method_lines = [line.rstrip("\n\r") for line in f]
        print(f"Loaded method file using encoding: {enc}")
        break
    except UnicodeDecodeError:
        continue
else:
    # Last resort: force-read with replacement characters
    with open(method_file, "r", encoding="latin1", errors="replace") as f:
        method_lines = [line.rstrip("\n\r") for line in f]
    print("Loaded method file using latin1 with replacement for invalid bytes.")

# One column with full lines
method_df = pd.DataFrame(method_lines, columns=[0])

# Strip outer braces: "{SimaPro ...}" -> "SimaPro ..."
method_df[0] = method_df[0].astype(str).str.strip("{} ")

# Split lines at semicolons into X1, X2, ...
CF = method_df[0].str.split(";", expand=True)
CF.columns = [f"X{i+1}" for i in range(CF.shape[1])]
CF = CF.fillna("")

# ---------------------------------------------------------
# Identify impact category blocks and names
# ---------------------------------------------------------
impact_cat_idx = np.where(CF["X1"] == "Impact category")[0]
if len(impact_cat_idx) == 0:
    raise ValueError("No 'Impact category' rows found in method file.")

segment_ends = np.append(impact_cat_idx[1:], len(CF) - 1)

# Names and units of impact categories:
Name = CF.loc[impact_cat_idx + 1, "X1"].to_numpy()
NameUnit = CF.loc[impact_cat_idx + 1, "X2"].to_numpy()

# Labels: "Name;Unit"
impact_labels = np.array([f"{n};{u}" for n, u in zip(Name, NameUnit)])
n_cats = len(impact_labels)

# CF ID similar to R:
#   "compartment;subcompartment;substance" = X1;X3;X2
CF["ID"] = (
    CF["X1"].astype(str)
    + ";" +
    CF["X3"].astype(str)
    + ";" +
    CF["X2"].astype(str)
)


# ---------------------------------------------------------
# Elementary flow meta-data and IDs
# ---------------------------------------------------------
# Rows of S / E correspond to elementary flows.
# In the SimaPro matrix they are described in the leftmost columns:
#   V1: compartment     (e.g. "Raw", "Air")
#   V2: sub-compartment (e.g. "in ground", "urban air")
#   V3: flow name
#   V4: unit
#
# We build two IDs for matching:
#   ID1 = [compartment; sub-compartment; name]
#   ID2 = [compartment; sub-compartment; "(unspecified)"]  (fallback)
# plus a display label with unit (ID3).
# ---------------------------------------------------------
elem_flows_df = matrix_df.iloc[ncol + 2 : nrow, 0:4].copy()
elem_flows_df.columns = [f"V{i+1}" for i in range(elem_flows_df.shape[1])]

elem_flows_df["ID1"] = (
    elem_flows_df["V1"].astype(str) + ";" +
    elem_flows_df["V2"].astype(str) + ";" +
    elem_flows_df["V3"].astype(str)
)

elem_flows_df["ID2"] = (
    elem_flows_df["V1"].astype(str) + ";" +
    elem_flows_df["V2"].astype(str) + ";" +
    "(unspecified)"
)

elem_flows_df["ID3"] = (
    elem_flows_df["V1"].astype(str) + ";" +
    elem_flows_df["V2"].astype(str) + ";" +
    elem_flows_df["V3"].astype(str) + ";" +
    elem_flows_df["V4"].astype(str)
)

n_flows = elem_flows_df.shape[0]  # number of elementary flows = rows of S/E
assert S.shape[0] == n_flows, "S/E and elementary flow list must have same number of rows"

# IDs as numpy arrays (trimmed)
ID1_arr = elem_flows_df["ID1"].astype(str).str.strip().to_numpy()
ID2_arr = elem_flows_df["ID2"].astype(str).str.strip().to_numpy()

# ---------------------------------------------------------
# Build characterization matrix C (flows × impact categories)
# ---------------------------------------------------------
# C[i, k] = characterization factor of flow i in category k
# Dimensions:
#   C : (n_flows × n_cats)
# ---------------------------------------------------------
C = np.zeros((n_flows, n_cats), dtype=float)

for k, (start, end) in enumerate(zip(impact_cat_idx, segment_ends)):
    seg = CF.loc[start:end].copy()

    # IDs for this category block
    seg_ids = seg["ID"].astype(str).str.strip()
    id_to_row = dict(zip(seg_ids, seg.index))

    col_vals = np.zeros(n_flows, dtype=float)

    # Match each elementary flow in the database to a CF in this category
    for i_flow in range(n_flows):
        idx = id_to_row.get(ID1_arr[i_flow])
        if idx is None:
            idx = id_to_row.get(ID2_arr[i_flow])
        if idx is not None:
            val_str = CF.at[idx, "X5"]
            try:
                col_vals[i_flow] = float(val_str)
            except (TypeError, ValueError):
                col_vals[i_flow] = 0.0

    C[:, k] = col_vals

C = np.nan_to_num(C, nan=0.0)
C_df = pd.DataFrame(C, columns=impact_labels)

# Optional: full CF database for inspection
CEI = pd.concat(
    [elem_flows_df["ID3"].reset_index(drop=True), C_df.reset_index(drop=True)],
    axis=1
)
CEI.columns = ["ElementaryFlow"] + list(impact_labels)



#%%
# ---------------------------------------------------------
# Section 3: Compute D_pro = C · E
# ---------------------------------------------------------
# For each process j (column of E):
#   e_j = E[:, j]              (LCI = vector of stressors, n_flows)
#   d_j = C · e_j              (n_imp × 1)
#
# In matrix form:
#   D_pro = C · E              (n_imp × n_proc)
#
# Dimensions:
#   C     : (n_imp × n_flows)
#   E     : (n_flows × n_proc)
#   D_pro : (n_imp × n_proc)
# ---------------------------------------------------------

D_pro = C.T @ E   # since current C has to be transposed, as it is stored here.

# Transpose for export (process × impact)
D_pro_df = pd.DataFrame(D_pro.T, columns=impact_labels)




# ---------------------------------------------------------
# Attach process meta-data (names, units, categories)
# ---------------------------------------------------------
# The meta-information for each process lives in the top rows of the matrix:
#   matrix_df[0:5, 3:ncol]   (rows 1..5, columns 4..end, 1-based)
#
# Transpose and clean up to obtain a process table:
#   process_info : (n_proc × n_meta_cols)
# ---------------------------------------------------------
proc_meta = matrix_df.iloc[0:5, 3:ncol].T  # transpose rows 1..5, cols 4..ncol
proc_meta.columns = [f"X{i+1}" for i in range(proc_meta.shape[1])]

# Drop columns that you don't need (as in R code)
for col in ["X3", "X4"]:
    if col in proc_meta.columns:
        proc_meta = proc_meta.drop(columns=col)

# First row contains column names (e.g. "Product", "Unit", "Category", ...)
proc_meta.columns = proc_meta.iloc[0]
process_info = proc_meta.iloc[1:].reset_index(drop=True)

# Sanity check: process_info rows must match n_proc
assert process_info.shape[0] == n_proc, "Process meta rows must equal number of processes"

# Final table: process meta + impacts per process (D_pro)
output = pd.concat([process_info.reset_index(drop=True), D_pro_df], axis=1)

# Write CSV with ';' separator
out_path = directory_data_3 / "Impact_Assessment.csv"
output.to_csv(out_path, sep=";", index=False, encoding="utf-8")

print(f"Impact assessment results written to: {out_path}")
print(f"Dimensions: A {A.shape}, S {S.shape}, E {E.shape}, C {C.shape}, D_pro {D_pro.shape}")



pass







#%%
# In case we want to further adjust the output file:

# ==============================================================
# 1) Replace "Cut-off, S" with "Cut-off, U" in the first column (usually "Product")
first_col = output.columns[0]
output[first_col] = (
    output[first_col]
    .astype(str)
    .str.replace("Cut-off, S", "Cut-off, U", regex=False)
)


# ==============================================================
#2) Use the list of activities to keep
activity_filter_file = working_dir / "mapping_SimaPro_to_ecoinvent_311+names.xlsx"
activity_filter_df = pd.read_excel(activity_filter_file)

# Column in the filter file that contains the activity names
filter_col_name = "simapro_name"

# Get the list of products to keep
activities_to_keep = (
    activity_filter_df[filter_col_name]
    .astype(str)
    .str.strip()
    .unique()
)

# Filter 'output' to only keep rows whose Product is in the list
output = output[
    output["Product"].astype(str).str.strip().isin(activities_to_keep)].reset_index(drop=True)



# Change impact assessment names
# dictionary: from SimaPro EF v3.1 column name → to short standardized name
# Mapping from original EF3.1 column names (with units) to your preferred literal strings with parentheses and quotes
impact_name_map = {
    "Acidification;mol H+ eq":                                "('EF v3.1', 'Acidification')",
    "Climate change;kg CO2 eq":                               "('EF v3.1', 'Climate change')",
    "Climate change - Biogenic;kg CO2 eq":                    "('EF v3.1', 'Climate change - Biogenic')",
    "Climate change - Fossil;kg CO2 eq":                      "('EF v3.1', 'Climate change - Fossil')",
    "Climate change - Land use and LU change;kg CO2 eq":      "('EF v3.1', 'Climate change - Land use and LU change')",
    "Ecotoxicity, freshwater;CTUe":                           "('EF v3.1', 'Ecotoxicity, freshwater')",
    "Ecotoxicity, freshwater - inorganics;CTUe":              "('EF v3.1', 'Ecotoxicity, freshwater - inorganics')",
    "Ecotoxicity, freshwater - organics;CTUe":                "('EF v3.1', 'Ecotoxicity, freshwater - organics')",
    "Particulate matter;disease inc.":                        "('EF v3.1', 'Particulate matter')",
    "Eutrophication, marine;kg N eq":                         "('EF v3.1', 'Eutrophication, marine')",
    "Eutrophication, freshwater;kg P eq":                     "('EF v3.1', 'Eutrophication, freshwater')",
    "Eutrophication, terrestrial;mol N eq":                   "('EF v3.1', 'Eutrophication, terrestrial')",
    "Human toxicity, cancer;CTUh":                            "('EF v3.1', 'Human toxicity, cancer')",
    "Human toxicity, cancer - inorganics;CTUh":               "('EF v3.1', 'Human toxicity, cancer - inorganics')",
    "Human toxicity, cancer - organics;CTUh":                 "('EF v3.1', 'Human toxicity, cancer - organics')",
    "Human toxicity, non-cancer;CTUh":                        "('EF v3.1', 'Human toxicity, non-cancer')",
    "Human toxicity, non-cancer - inorganics;CTUh":           "('EF v3.1', 'Human toxicity, non-cancer - inorganics')",
    "Human toxicity, non-cancer - organics;CTUh":             "('EF v3.1', 'Human toxicity, non-cancer - organics')",
    "Ionising radiation;kBq U-235 eq":                        "('EF v3.1', 'Ionising radiation')",
    "Ozone depletion;kg CFC11 eq":                            "('EF v3.1', 'Ozone depletion')",
    "Photochemical ozone formation;kg NMVOC eq":              "('EF v3.1', 'Photochemical ozone formation')",
    "Resource use, fossils;MJ":                               "('EF v3.1', 'Resource use, fossils')",
    "Resource use, minerals and metals;kg Sb eq":             "('EF v3.1', 'Resource use, minerals and metals')",
    "Water use;m3 depriv.":                                   "('EF v3.1', 'Water use')",
    "Land use;Pt":                                            "('EF v3.1', 'Land use')",
}


# ==============================================================
#3) Transpose the matrix to Long format LCIA dataset (transpose impacts: columns → rows)
# Meta columns (process info) are those taken from process_info
output_for_long = output.rename(columns=impact_name_map)

# Meta columns = process info (unchanged)
meta_cols = list(process_info.columns)

# Impact columns in the original wide result are still the original EF names
# (impact_labels = array of SimaPro "Name;Unit" strings)
# We map them to their new names in the same order:
impact_cols_long = [impact_name_map.get(col, col) for col in impact_labels]


output_long = output_for_long.melt(
    id_vars=meta_cols,
    value_vars=impact_cols_long,
    var_name="ImpactCategory",
    value_name="ImpactValue",
)

# Optional: reorder columns
output_long = output_long[meta_cols + ["ImpactCategory", "ImpactValue"]]


# ==============================================================
#4) Export results
out_path_wide = directory_data_3 / "Impact_Assessment_wide.csv"
output.to_csv(out_path_wide, sep=";", index=False, encoding="utf-8")

out_path_long = directory_data_3 / "Impact_Assessment_long.csv"
output_long.to_csv(out_path_long, sep=";", index=False, encoding="utf-8")

print(f"Wide impact assessment written to: {out_path_wide}")
print(f"Long impact assessment written to: {out_path_long}")