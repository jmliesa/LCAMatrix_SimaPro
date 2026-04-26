
# LCA matrix calculation from SimaPro export: LCAMatrix_SimaPro
This script computes Life Cycle Assessment (LCA) results from a SimaPro raw matrix export using a fully matrix-based approach in Python.
Originally developed as an R script by Xun Liao (former PhD candidate at EPFL, Switzerland), it has been adapted and extended in Python by @CedricFurrer and @jmliesa from the LCA group at @Agroscope, the Swiss centre of excellence for agricultural research (Zürich).

## Motivation
This implementation serves several purposes, including, for instance:

1. **Teaching**  
   Helps students understand the underlying logic of LCA calculations (matrix formulation, Leontief system, inventory and impact assessment).

2. **Verification / transparency**  
   Provides an independent way to reproduce and double-check results from LCA software such as SimaPro or Brightway.

3. **Scalability**  
   Enables computation of LCA results for large databases, allowing simultaneous evaluation of all processes.

## Scope and applicability
- Works with any database exported from **SimaPro** (PhD or Developer license required for matrix export)
- Computes:
  - Life Cycle Inventory (LCI)
  - Life Cycle Impact Assessment (LCIA)
- Outputs impact scores for all processes in the database

## Methodological background
The script follows the standard matrix-based LCA formulation:
- \x = (I - A)^{-1} y\ → total production (supply chain)  
- \e = Sx\ → life cycle inventory  
- \d = Ce\ → impact assessment  

where:
- \A\: requirement (technosphere) matrix  
- \(S)\: stressor (biosphere) matrix  
- \C\: characterization matrix  
- \y\: functional unit (demand vector)  

The implementation computes results for all processes simultaneously by using an identity demand matrix.
This formulation follows standard LCA theory as taught in:

- To better follow this code, we provide a summary on how LCA calculations work, based on the lectures from the Life Cycle Assessment course from NTNU and the book 'Methodological Essentials of Life Cycle Assessment' by Anders H. Strømman.

---

## 1. Purpose of the script
The script is designed to:
- read a SimaPro-exported matrix from Excel  
- normalize raw exchanges by the reference output  
- construct technosphere (`A`) and stressor (`S`) matrices  
- compute inventory results  
- read LCIA methods and match characterization factors  
- compute impact scores per process  
- export results to CSV  

---

## 2. Folder structure expected
```text
Data/
├── Matrix_to_Analyse/
│   └── your_matrix_file.xlsx
├── Method_CSV/
│   └── your_method_file.csv
└── Results/
```

---

## 3. Core calculation (summary)
The script performs the following steps:

1. **Normalization**  
   Raw matrices are divided by the reference output (`q`) to obtain:
   - `A` (technosphere matrix)  
   - `S` (stressor matrix)  

2. **System solution**  
   The Leontief system is solved:
   ```text
   X = (I - A)⁻¹
using an identity demand matrix (`Y = I`), meaning:
- column 1 = demand of 1 unit of process 1  
- column 2 = demand of 1 unit of process 2  
- ...  

This allows computing the LCA results for **all processes simultaneously**, without looping over individual functional units.

3. **Inventory calculation**
```text
E = S · X
```
→ full life cycle inventory per process  

4. **Impact assessment**
```text
D = C · E
```
→ impact scores per process and category
The output matrix is equivalent to computing, per process:

```text
d = C S (I - A)⁻¹ y
```

## 4. Output
Main output: Results/Impact_Assessment.csv
- rows = processes  
- columns = impact categories  

Each value represents the impact of **1 unit of that process**.

---

## 5. Matrix dimensions (quick reference)

| Matrix | Shape | Meaning |
|--------|------|--------|
| A | (proc × proc) | requirements |
| S | (flows × proc) | stressors |
| X | (proc × proc) | total outputs |
| E | (flows × proc) | inventory |
| C | (imp × flows) | characterization |
| D | (imp × proc) | impacts |

---

## 6. Key notes
- `q` is only used to normalize SimaPro exports (not part of standard LCA theory)  
- `Y = I` means the script computes **one LCA per process simultaneously**  
- results correspond to **total impacts per process**, not contribution analysis  

---

## 7. Requirements
```bash
pip install pandas numpy scipy openpyxl
```

---

## 8. Usage

1. Export the matrix and LCIA method from SimaPro  
2. Place the files in the corresponding folders  
3. Set the `working_dir` in the script  
4. Run the script  
5. Check results in the `/Results` folder 

