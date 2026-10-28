# 📘 Learning Guide

This guide teaches you **why** the project is built the way it is, so you can explain every line in an internship or entry-level Data Science interview. Read it side by side with the code.

**Suggested order:** `config.py` → `data_generator.py` → `data_loader.py` → `data_cleaning.py` → `grading.py` → `numpy_analysis.py` → `pandas_analysis.py` → `visualizations.py` → `statistical_charts.py` → `report_export.py` → `main.py`.

**For each module** you will find: ① purpose · ② key functions explained · ③ NumPy/Pandas vs plain Python · ④ sample input → output · ⑤ common errors · ⑥ exercises.

> 💡 **Work actively.** After reading a section, open a Python shell (`python` inside `src/`) and try the examples yourself. Break things on purpose and read the error messages. That is how debugging skill is built.

---

## 0. The big picture

### How the modules talk to each other

```
                     config.py  (settings used by everyone)
                         │
 data_generator ──► data/students.csv ──► data_loader ──► data_cleaning ──► grading
                                                                              │
                                  ┌───────────────┬───────────────┬──────────┴────┐
                              numpy_analysis  pandas_analysis  visualizations  report_export
                                  └───────────────┴───────┬───────┴───────────────┘
                                                        main.py  (menu, uses ui.py to print)
```

Each module has **one job**. Analysis modules never print and never read files; they receive a DataFrame and return a result. That separation is why they can be tested with tiny hand-made data.

### NumPy vs Pandas vs plain Python: when to use which

| Need | Best tool | Why |
|------|-----------|-----|
| One column of numbers: mean, std, percentiles, masks | **NumPy** array | Compact memory, calculations in compiled C code, no loops |
| A table with named columns and mixed types (text + numbers) | **Pandas** DataFrame | Labels, missing-value handling, grouping, joining, CSV I/O |
| A picture of the data | **Matplotlib** (full control) or **seaborn** (statistical charts in one line) | seaborn is built on Matplotlib and understands DataFrames |
| Quick small logic (a few items) | Plain Python list/dict | Simple, no overhead |

**Rule of thumb:** *Pandas* for the table, *NumPy* for the maths on its columns. Pandas is actually built on top of NumPy.

### Series vs DataFrame

```python
df["final_marks"]            # single brackets -> Series    (one column)
df[["name", "final_marks"]]  # double brackets -> DataFrame (a table)
```

* A **Series** is a labelled one-dimensional array: values + an index.
* A **DataFrame** is a table: several Series that share the same row index.
* `df["final_marks"].shape` → `(155,)`, while `df[["name","final_marks"]].shape` → `(155, 2)`.

Menu option **1** shows this live.

### Vectorisation, the key idea

```python
marks = np.array([45, 50, 72, 90])
marks >= 50                 # array([False,  True,  True,  True])
(marks >= 50).sum()         # 3        (True counts as 1)
(marks >= 50).mean()        # 0.75     (share of students)
```

No `for` loop, no `if`. The operation is applied to every element at once. Menu option **4 → section 6** times a Python loop against `np.sum` on a million numbers (NumPy is typically 10–50× faster).

---

## 1. `config.py`: one place for every setting

**Purpose.** Paths, weights, thresholds, valid ranges and colours live here. If a lecturer says "make the pass mark 45", you change **one number**, not five scattered ones.

**Key ideas**

* `Path(__file__).resolve().parent.parent` finds the project folder **wherever it is installed**, and `pathlib` uses the right slash for Windows (`\`) automatically. Never hard-code `"C:\\Users\\..."`.
* `GRADE_WEIGHTS` is a **dictionary**: column name → weight. Using the same column names as the DataFrame lets the grading code line them up automatically.
* `GRADE_BANDS = [(85, "A"), (75, "B"), ...]` is a **list of tuples** checked from top to bottom.

**Common errors**

| Error | Cause → fix |
|-------|-------------|
| Weights "must add up to 1.0" | You changed one weight but not the others → make them total 1.0 (100 %). |

**Exercises**

1. Change the weights to 20/20/20/40. Run menu 5 → 3 and notice how the pass rate changes.
2. Add a new grade band `(95, "A+")` above the A. *Hint:* the list must stay in descending order, or `grading.py` raises an error. Why does it check this?

---

## 2. `data_generator.py`: building a realistic fake dataset

**Purpose.** You need practice data that is safe (no real students) and *imperfect* (like real data). The generator creates 150 students, then injects realistic problems.

**Key functions**

* `np.random.default_rng(seed)`: NumPy's modern random generator. The **seed** makes it reproducible: the same seed always produces the same "random" numbers on any computer.
* `rng.normal(mean, std, n)`: n numbers from a bell curve. `rng.integers(low, high, n)`: random whole numbers (high is excluded). `rng.choice(list, n)`: random picks.
* `np.clip(values, 0, 100)`: forces numbers into a range (so marks stay 0–100).
* **A hidden "ability" variable** drives study hours, attendance and marks together, so the data contains realistic relationships (that is what the later correlation finds).
* `_inject_messiness()` adds: ~3 % missing values per column, one impossible value of each kind (e.g. final mark 105), inconsistent labels (`"cs"`, `"COMPUTER SCIENCE"`, `"  Data Science "`), two missing departments and five duplicated rows.

**Why NumPy here?** `rng.normal(12, 4.5, 150)` makes 150 numbers in one call. With plain Python you would write a loop calling `random.gauss` 150 times.

**Sample**

```python
import data_generator as dg
df = dg.generate_students()
df.shape                       # (155, 12)  -> 150 students + 5 duplicate rows
dg.generate_students(seed=1).equals(dg.generate_students(seed=1))   # True
```

**Common errors**

| Error | Cause → fix |
|-------|-------------|
| `ValueError: n must be at least 1` | You asked for 0 students. |
| Data looks different on a friend's PC | Someone changed `RANDOM_SEED`, or `generate_students(seed=...)`. |

**Exercises**

1. Generate 300 students (`generate_students(n=300)`) and check `shape`. You should see 305 rows. Why?
2. Generate a *clean* dataset with `messy=False`. Verify `df.isnull().sum().sum()` is `0`.
3. Change the generator so attendance matters twice as much for marks. Does the correlation in menu 5 → 7 change?

---

## 3. `data_loader.py`: reading the CSV safely

**Purpose.** Most beginner errors happen when loading files. This module turns every likely problem into a clear message, so the rest of the program can assume it has a proper DataFrame.

**Key functions**

* `pd.read_csv(path)`: reads a CSV into a DataFrame. Pandas guesses each column's data type.
* `df.shape` → `(rows, columns)` · `df.columns` · `df.dtypes` · `df.head()` · `df.describe()` · `df.isnull().sum()`.
* `df[config.COLUMNS].copy()`: selects columns in a fixed order and makes an independent copy.

**Why Pandas instead of the `csv` module + lists?** With the `csv` module every value is a string; you would convert types, handle blanks and compute column averages by hand. `read_csv` does it in one call, and `describe()` replaces dozens of lines of statistics code.

**Sample**

```text
>>> dl.dtypes_table(df).head(3)
      column   dtype  missing
  student_id     str        0
        name     str        0
         age float64        4        <- float because NaN cannot be stored in an int column
```

**Common errors**

| Error | Cause → fix |
|-------|-------------|
| `FileNotFoundError: Could not find '...students.csv'` | Wrong folder, or you never generated it → menu offers to create it. |
| `ValueError: 'x.csv' is empty` | The file has no content (`pandas.errors.EmptyDataError` is translated). |
| `missing required column(s): ...` | Column names differ (typos, extra spaces). Compare with `config.COLUMNS`. |
| `age` is `float64` instead of `int` | Missing values (`NaN`) are floats; cleaning converts it to `int` afterwards. |

**Exercises**

1. In a Python shell: `df = dl.load_students()`, then print `df["department"].value_counts()`. Why are there so many different department labels?
2. Find the five youngest students: `df.sort_values("age").head()`. One age is impossible for a university student. Which?
3. Make `load_students` also reject a file with fewer than 10 rows (raise a `ValueError`).

---

## 4. `data_cleaning.py`: fixing data without hiding anything

**Purpose.** Real data is messy. A professional analyst *fixes what can be fixed, removes what cannot, and writes down every decision*.

### The decision guide

| Situation | Decision | Reason |
|-----------|----------|--------|
| Duplicate student | **Remove** (keep first) | Otherwise a student is counted twice |
| Missing student ID / name / department / semester | **Remove** | Identity cannot be guessed |
| Missing or invalid **midterm / final** mark | **Remove** | Inventing an exam score would change a grade based on a guess |
| Missing assignment / quiz / attendance / study hours / age | **Fill with a group median**, flag the row | A fair estimate exists; the row still carries valuable information |
| Impossible value (mark 105, attendance 112 %) | **Treat as missing** (do **not** clip to 100) | We cannot know whether it was a typo for 10.5 or 100 |
| `"cs"`, `"COMPUTER SCIENCE"`, `" Data Science "` | **Standardise** | Otherwise `groupby` makes separate groups for one department |

**Why the median?** Mean is pulled by extreme values; the median is "the middle student", so one 0 or 100 does not distort the fill value. Example: median of `[50, 52, 51, 99]` is 51.5 but the mean is 63.

### Key functions

* `df.isnull()` / `df.isna()`: True where a value is missing. `.sum()` counts them per column.
* `df.duplicated(subset="student_id")`: True for the *second and later* occurrence of an ID.
* `pd.to_numeric(series, errors="coerce")`: converts to numbers; anything unparsable (like `"N/A"`) becomes `NaN`.
* `series.str.strip()` / `.map(function)`: text clean-up and label mapping.
* `series.between(...)` and comparisons build **boolean masks** (`(values < 0) | (values > 100)`).
* `df.groupby("department")["quiz_marks"].transform("median")`: returns, for *every row*, the median of its own group. That is perfect for `fillna`.
* `df.copy()`: the first line of `clean_students`. Never edit the original (that is what "no data leakage" means here).

### The function returns a dictionary

```python
result = dc.clean_students(raw)
result["data"]      # the cleaned DataFrame
result["log"]       # list of sentences explaining each step
result["removed"]   # the removed rows, each with a removal_reason
```

**Sample (real log lines)**

```text
Step 4: Removed 5 duplicate record(s) (5 exact copies). We keep the FIRST occurrence ...
Step 5: 1 impossible value(s) in 'final_marks' (valid range 0-100; e.g. STU-0126=105) ...
Step 6: Removed 11 row(s) in total (IDs: STU-0001, STU-0018, ...) ...
Step 7: Filled 5 missing 'assignment_marks' value(s) with the median of the student's department group ...
```

Before → after: **155 rows → 139 rows**, missing cells **29 → 0**, department labels **21 → 6**.

**Common errors**

| Error | Cause → fix |
|-------|-------------|
| `SettingWithCopyWarning` (older pandas) | Editing a slice of a DataFrame. Use `.copy()` and `.loc[mask, col] = value`. |
| `KeyError: 'Department'` | Column names are **case-sensitive**: it is `department`. |
| `ValueError: cannot convert float NaN to integer` | You called `.astype(int)` before filling missing values. |
| `AttributeError: Can only use .str accessor with string values` | The column is numeric (or all-NaN); convert with `.astype("string")` first. |

**Exercises**

1. Add `"cse": "Computer Science"` to `DEPARTMENT_ALIASES` in `config.py`. Test it with a one-row DataFrame.
2. Change the rule so that a missing **quiz** mark also removes the row. Which tests in `tests/test_data_cleaning.py` fail, and why is that useful?
3. *Challenge:* instead of the department median for `study_hours_per_week`, fill with the *semester* median. Compare the final averages.

---

## 5. `grading.py`: weighted averages

**Purpose.** Turn four marks into one total, a grade, a category and a pass/fail result, using configurable example rules.

### How a weighted average works

Each component counts for a different share:

```
total = 0.10×assignments + 0.10×quizzes + 0.30×midterm + 0.50×final
```

Worked example: assignments 80, quizzes 70, midterm 60, final 90

```
0.10×80 + 0.10×70 + 0.30×60 + 0.50×90  =  8 + 7 + 18 + 45  =  78.0
```

This only works if **all marks use the same scale** (here 0–100) **and the weights sum to 1**. If the quiz were marked out of 20, you would first convert it to a percentage (`quiz / 20 * 100`). `validate_weights` and `validate_marks` enforce this.

### Key functions

* `marks @ weight_vector` (matrix product): multiplies each student's four marks by the four weights and adds them, **for all students at once**. It replaces nested loops.
* `np.isclose(total, 1.0)`: compares floats safely (because `0.1 + 0.2 != 0.3` exactly in computers).
* `np.select(conditions, choices)`: a vectorised `if / elif / else`. Used for grade bands: `conditions = [total >= 85, total >= 75, ...]`.
* `np.where(total >= pass_mark, "Pass", "Fail")`: vectorised if/else.
* `Series.rank(ascending=False, method="min")`: rank 1 = best.

**Sample**

```python
import pandas as pd
import config, grading

df = pd.DataFrame({"assignment_marks": [80], "quiz_marks": [70],
                   "midterm_marks": [60], "final_marks": [90]})
grading.weighted_total(df).iloc[0]                                 # 78.0
grading.assign_labels([85, 84.99, 49.9], config.GRADE_BANDS).tolist()
# ['A', 'B', 'F']
```

> ⚠️ The bands (85/75/65/50) and pass mark (50) are **example rules**, not an official policy.

**Common errors**

| Error | Cause → fix |
|-------|-------------|
| `Grading weights must add up to 1.0 ... they add up to 0.90` | Fix the weights (menu 5 → 8 shows this message and keeps the old ones). |
| `'quiz_marks' still contains missing values - clean the data first` | You graded before cleaning. |
| `Bands must be listed from the highest minimum to the lowest` | `np.select` checks conditions in order; a wrong order gives wrong grades. |

**Exercises**

1. Calculate by hand the total for assignments 100, quizzes 50, midterm 40, final 60. Check with `weighted_total`.
2. Add a "distinction" flag column: `total_marks >= 90`.
3. What happens to a student with 49.99? Why might rounding totals to 2 decimals matter near a boundary?

---

## 6. `numpy_analysis.py`: numerical analysis

**Purpose.** Practise core NumPy operations on the marks arrays. Each section in menu **4** prints **Input / Output / Purpose / Why**.

| Operation | Example | Output | Why NumPy suits it |
|-----------|---------|--------|--------------------|
| Mean / median / min / max | `np.mean(arr)` | one number | Whole array processed at once |
| Std / variance | `np.std(arr, ddof=0)` | spread of marks | One call instead of a manual formula loop |
| Percentiles | `np.percentile(arr, [25, 50, 75])` | marks at those positions | Sorting + interpolation built in |
| Boolean mask | `(total >= 85) & (att >= 85)` | array of True/False | Comparison applied to all students; `&` combines conditions |
| Unique + counts | `np.unique(arr, return_counts=True)` | `(values, counts)` | Replaces a dictionary-of-counters loop |
| Array arithmetic | `final - midterm` | per-student change | Element-by-element subtraction |
| `np.argmax` | `np.argmax(change)` | position of the biggest value | Find "who improved most" |
| z-score | `(x - x.mean()) / x.std()` | distance from the mean in std units | Spot unusual scores |

### Population vs sample (`ddof`), a classic interview question

* `np.std(arr)` divides by **n** (`ddof=0`) → use when you have the *whole population*.
* `pandas.Series.std()` divides by **n − 1** (`ddof=1`) → use for a *sample*.
* So NumPy and pandas give slightly different standard deviations by default. This is expected; the code states which one it uses.

### Boolean masking, step by step

```python
total      = np.array([45, 90, 72, 88])
attendance = np.array([90, 92, 60, 86])

mask = (total >= 85) & (attendance >= 85)
# array([False,  True, False,  True])
mask.sum()           # 2 students
df.loc[mask]         # in pandas, a mask selects rows
```

Use `&` (and), `|` (or), `~` (not), with **parentheses around each comparison**. Python's `and`/`or` do not work on arrays.

**Sample**

```python
labels, counts = np.unique(["CS", "DS", "CS", "EE", "CS"], return_counts=True)
labels   # array(['CS', 'DS', 'EE'], dtype='<U2')
counts   # array([3, 1, 1])
```

**Common errors**

| Error | Cause → fix |
|-------|-------------|
| `ValueError: The truth value of an array ... is ambiguous` | You used `and` / `or` / `if mask:` with arrays → use `&`, `\|`, and `mask.any()` / `mask.all()`. |
| `ValueError: operands could not be broadcast together` | Two arrays have different lengths (`score_change` checks this for you). |
| `nan` results from `np.mean` | The array contains `NaN`. Clean first, or use `np.nanmean`. |
| `TypeError: '>=' not supported between 'str' and 'int'` | The column holds text. Convert with `pd.to_numeric`. |

**Exercises**

1. Use a mask to count students with `final_marks > midterm_marks + 10`.
2. Calculate the 95th percentile of total marks. What does it mean in words?
3. Use `np.unique` on `performance_category`. Why is the output order alphabetical and not "Excellent, Good, ..."?

---

## 7. `pandas_analysis.py`: analysing the table

**Purpose.** Answer business questions: *who is top? which department is strongest? who needs help?*

### Key methods

| Method | Used for | Example in the project |
|--------|----------|------------------------|
| `sort_values(by, ascending)` | ordering rows | top/bottom 10 (ties broken by `student_id` for stable results) |
| `groupby(col).agg(name=(col, func))` | statistics per group | `department_stats` |
| `value_counts()` | counting categories | `department_counts` |
| `df.loc[mask, columns]` | select rows + columns | high attendance / low performance |
| `df.query("attendance_pct < @threshold")` | readable filtering; `@` uses a Python variable | low attendance list |
| `df.merge(other, on="department", how="left")` | joining two tables | adds the `faculty` column |
| `pd.cut(values, bins, labels)` | turning numbers into ranges | study-hour buckets |
| `.corr()` | correlation between columns | study hours vs total |
| `reindex(order, fill_value=0)` | fixed category order, no missing categories | performance categories |

### Group-by in three lines

```python
df = pd.DataFrame({"dept": ["CS", "CS", "DS"], "total": [60, 80, 50]})
df.groupby("dept")["total"].mean()
# dept
# CS    70.0
# DS    50.0
```

**Split** the rows by department → **apply** `mean` to each group → **combine** the results. (Interview phrase: "split-apply-combine".)

### Correlation, carefully

`study_hours_association` reports Pearson and Spearman values (about **+0.38** and **+0.35** on this data: a *weak positive association*), plus a bucket table: students studying 20+ hours average much higher than those studying under 5.

> **Association ≠ causation.** Studying more *is associated* with higher marks, but this data cannot prove it *causes* them: prior knowledge or motivation could drive both. The program prints this warning every time.
>
> *Spearman* = Pearson correlation of the **ranks**. It is less sensitive to outliers; the code computes it as `x.rank().corr(y.rank())`.

**Common errors**

| Error | Cause → fix |
|-------|-------------|
| `KeyError: 'total_marks'` | Analysis was run on the cleaned data *before* `add_performance_columns`. |
| `ValueError: Pass threshold must be between 0 and 100` | Bad input; the functions validate thresholds. |
| `FutureWarning` about `observed` in `groupby` | Categorical groups; the code passes `observed=True`. |
| `ValueError: The truth value of a Series is ambiguous` | Used `and` with Series; use `&` and parentheses. |
| `SettingWithCopyWarning` | Modifying a filtered slice; use `.copy()`. |

**Exercises**

1. Compute the average total **by gender** with `groupby`. Is the difference big enough to matter? (What does group size tell you?)
2. Use `query` to list students in semester 5 whose attendance is below 80 %.
3. *Challenge:* add `pd.crosstab(df["department"], df["grade"])` to see grades per department.

---

## 8. `visualizations.py`: Matplotlib

**Purpose.** A picture often reveals what a table hides. Four charts, one consistent theme.

**Vocabulary:** *Figure* = canvas · *Axes* = one plot with its axes, title and labels · `fig, ax = plt.subplots()` makes both.

| Chart | Method | Teaches |
|-------|--------|---------|
| Average marks by department | `ax.barh` | comparing groups; reference line (`axvline`) |
| Histogram of totals | `ax.hist` | distribution shape; mean vs median |
| Study hours vs marks | `ax.scatter` + `np.polyfit` trend line | relationships between two numbers |
| Students per category | `ax.bar` with value labels | counts and percentages |

**Design choices to talk about**

* Titles, axis labels and units on **every** chart.
* Light grid, no top/right borders, one colour palette from `config.THEME` (the same colour means the same category everywhere).
* A footer that says the data is synthetic.
* `bbox_inches="tight"` stops labels being cut off; `plt.close(fig)` frees memory.

**Common errors**

| Error | Cause → fix |
|-------|-------------|
| Chart windows never appear | Use menu 7 and answer `y`. If your environment has no display, PNG files are still saved. |
| Labels overlap or are cut off | `fig.tight_layout()` / `bbox_inches="tight"`; make long labels horizontal (`barh`). |
| `ModuleNotFoundError: matplotlib` | Activate the virtual environment and `pip install -r requirements.txt`. |
| Histogram axis shows 2.5 students | Use `MaxNLocator(integer=True)` (already done here). |

**Exercises**

1. Change the histogram bin width from 5 to 10. Which tells the story better?
2. Add a box plot of total marks per department (`ax.boxplot`).
3. Colour the scatter points by *department* instead of performance category.

---

## 8b. `statistical_charts.py`: seaborn

**Purpose.** Matplotlib draws from basic shapes: *you* decide every bar, line and label. **seaborn** is built **on top of Matplotlib** and understands DataFrames, so one line can produce a statistical chart that would take dozens of Matplotlib lines. This module adds four of them.

```
Matplotlib = the drawing engine          seaborn = statistical charts, built on it
```

Every seaborn chart is still a Matplotlib `Figure`, so you can add titles, lines and labels with the Matplotlib methods you already know (`ax.axvline`, `ax.set_xlabel`, ...). That is why both libraries work together in one project.

### The four charts

| Chart | seaborn call | What it reveals that an average cannot |
|-------|--------------|----------------------------------------|
| **Box plot + dots** by department | `sns.boxplot` + `sns.stripplot` | the **spread**: the box is the middle 50 % of students, the line is the median, the dots are every student |
| **Correlation heatmap** | `sns.heatmap(corr, annot=True)` | which habits and marks move together (colour = strength) |
| **Department x grade heatmap** | `sns.heatmap(pd.crosstab(...))` | where the A's and F's cluster |
| **Violin plot** per assessment | `sns.violinplot` | the whole **shape** of each component's marks, not just its mean |

### Key ideas

**1. Wide vs long data (`pd.melt`).** Our table is *wide*: one row per student, one column per component. seaborn usually wants *long* data: one row per (student, component) pair.

```python
long = df[["student_id", "assignment_marks", "quiz_marks"]].melt(
    id_vars="student_id", var_name="component", value_name="marks")
```

`pandas_analysis.components_long()` does this (139 students x 4 components = 556 rows). Then `sns.violinplot(data=long, x="component", y="marks")` knows exactly which column is which. Reshaping between wide and long is a core pandas skill.

**2. `pd.crosstab` feeds a heatmap.** `pandas_analysis.department_grade_table()` counts students per department and grade:

```text
grade                   A  B  C   D  F
Aerospace Engineering   2  3  6  10  5
Computer Science        2  5  4  11  0
Mechanical Engineering  2  7  6  12  3
```

`sns.heatmap(table, annot=True, fmt="d")` colours the cells and prints each count. Notice that Computer Science has no F grades on this data, but with groups of 20-30 students, treat small differences as noise.

**3. The correlation matrix.** `df[columns].corr()` gives every pairwise Pearson correlation. On this data, Assignments and Midterm are the closest pair (0.69), while Attendance and Study hours are only 0.16. The program **leaves out `total_marks`** on purpose: it is *calculated* from the four marks, so its correlation with them is built in and would teach nothing. The heatmap hides the duplicate upper triangle with a mask (`np.triu`).

**4. A shared theme.** `sns.set_theme(style="whitegrid", rc=RC, palette=[...])` is seaborn's one-call styling. We pass the *same* settings dictionary (`RC`) that Matplotlib uses, so all eight charts look like one family.

**5. One narrow warning filter.** seaborn 0.13 passes an old option (`vert=`) that Matplotlib 3.10 reports as "pending deprecation". It is harmless noise from inside the libraries, so the module hides **only that exact message**. Every other warning stays visible. Never silence all warnings.

### Why seaborn rather than only Matplotlib?

A box plot in pure Matplotlib needs you to group the data into lists yourself, then style each part. A heatmap with printed numbers needs a loop that places text in every cell. In seaborn each is one call, and the statistics (quartiles, density estimates) are computed for you.

### Sample

```python
import statistical_charts as sc, visualizations as viz
viz.apply_theme(); sc.apply_theme()
fig = sc.department_boxplot(perf)
fig.savefig("boxplot.png", bbox_inches="tight")
```

Menu option **7** creates all eight charts at once (four Matplotlib, four seaborn).

### Common errors

| Error | Cause -> fix |
|-------|--------------|
| `ModuleNotFoundError: No module named 'seaborn'` | The library is not installed in the active environment -> `pip install -r requirements.txt` |
| `FutureWarning` about `palette` without `hue` | Newer seaborn wants `hue=` when you pass `palette=`. This project passes a single `color=` instead |
| Heatmap shows grid lines over the cells | Our theme turns the grid on. Call `ax.grid(False)` on heatmaps (done here) |
| Heatmap squares are cut off or labels overlap | Increase `figsize`, rotate labels with `plt.setp(ax.get_xticklabels(), rotation=30, ha="right")` |
| Violin looks empty or odd | Too few data points per group, or you forgot `cut=0` (which stops the shape extending past the real min/max) |
| `ValueError` about `x`/`y` and `data` | You passed wide data where seaborn expects long data -> use `pd.melt` |

### Exercises

1. Change the box plot to a **vertical** one (swap `x` and `y`, `orient="v"`). Which reads better for long department names?
2. Add `sns.histplot(df["total_marks"], kde=True)` as a fifth seaborn chart. What does the `kde` line add to a histogram?
3. Make the correlation heatmap show **all** cells by removing the mask. Why is the upper half redundant?
4. *Challenge:* use `sns.regplot(data=df, x="study_hours_per_week", y="total_marks")` and compare its trend line with the `np.polyfit` line in the Matplotlib scatter. Are they the same?

### Checkpoint

- What is the relationship between seaborn and Matplotlib?
- What is the difference between wide and long data, and which function reshapes it?
- Why does the correlation heatmap exclude `total_marks`?
- What does a violin plot show that an average does not?

---

## 9. `report_export.py`: sharing results

**Purpose.** Analysis is only useful if others can open it. Five CSV reports + a cleaning log.

| File | Use |
|------|-----|
| `cleaned_students.csv` | Trusted dataset for later analysis, or a future ML project |
| `student_performance_report.csv` | Advisors: totals, grades, ranks per student |
| `department_statistics.csv` | Management: compare departments |
| `low_attendance_students.csv` | Follow-up list for attendance counselling |
| `top_students.csv` | Recognition / awards shortlist |
| `cleaning_log.txt` | Audit trail: proof of how data was changed |

**Key details**

* `df.to_csv(path, index=False)`: `index=False` avoids an extra unnamed column of row numbers.
* `encoding="utf-8-sig"`: Excel on Windows reads special characters properly.
* `Path.mkdir(parents=True, exist_ok=True)`: creates the folder if needed, without error if it exists.
* Each table is created from a **copy**; exporting cannot alter data used elsewhere.

**Common errors**

| Error | Cause → fix |
|-------|-------------|
| `PermissionError` | The CSV is open in Excel → close it. |
| Extra "Unnamed: 0" column after reloading | The file was saved without `index=False`. |
| Garbled characters in Excel | Save with `utf-8-sig`. |

**Exercises**

1. Export a `failed_students.csv` containing only students whose result is `"Fail"`.
2. Reload `student_performance_report.csv` with `pd.read_csv` and confirm the row count equals the cleaned dataset.
3. Add a `generated_on` column with today's date (`pd.Timestamp.today().date()`).

---

## 10. `main.py` and `ui.py`: the command-line app

**Purpose.** `main.py` only decides *which function to call* and *how to print the result*. `ui.py` holds the visual theme (colours, banner, tables, validated input).

**Ideas worth knowing**

* A **table-driven menu** (`MENU = [(key, label, function), ...]`, a list of tuples) means a new menu item is one new line.
* **State dictionary** (`raw`, `clean`, `performance`, `settings`): data flows `raw → clean → performance`. Later stages run earlier ones automatically if you skip ahead, and *say so*.
* **Input validation:** `ui.ask_int` / `ui.ask_float` keep asking until the answer is valid, so a typo cannot crash the program.
* **`guard()`** turns expected errors (`FileNotFoundError`, `ValueError`, `PermissionError`) into friendly messages. Use `--debug` to see the full traceback while learning.
* **`argparse`** provides `--pipeline`, `--regenerate`, `--debug`.

**Common errors**

| Error | Cause → fix |
|-------|-------------|
| `ModuleNotFoundError: config` | Run `python src\main.py` from the project folder. |
| Menu keeps saying "Please enter one of ..." | Type only the number shown. |
| Colour codes like `←[36m` | Old terminal; use Windows Terminal / VS Code, or set `NO_COLOR=1`. |

**Exercises**

1. Add menu option `Show data quality report` that prints `dc.missing_report(state["raw"])`.
2. Add a `--seed` command-line argument that changes `config.RANDOM_SEED`.
3. Add a "Search by department and semester" view in `action_view_records`.

---

## 11. `tests/`: proving your code works

```powershell
python -m unittest discover -s tests -t . -v
```

* Use **tiny hand-made data** where you can calculate the answer by hand (see `tests/helpers.py`), e.g. total of 78.0.
* Test the *boundaries* (84.99 vs 85) and the *failure cases* (`assertRaises(ValueError)`).
* One test calls the cleaner and then checks that the **original DataFrame is unchanged** (no leakage).
* While building this project, a test caught a real bug: a plain Python list has an `.index` *method*, which `getattr(totals, "index")` picked up by mistake. That is exactly why tests exist.

**Exercises**

1. Write a test that checks `weighted_total` returns 0 for a student with all zeros.
2. Write a test that a duplicate row with a *different* name but the same ID is still removed.
3. Break `validate_weights` on purpose and watch which tests fail.

---

## 12. Interview cheat-sheet

**What is NumPy? What is Pandas? Why both?**
NumPy provides fast n-dimensional arrays and vectorised maths. Pandas builds on it to add labelled tables (DataFrame/Series), missing-value handling, group-by, joins and I/O. NumPy for numerical computation, Pandas for tabular data analysis.

**Series vs DataFrame?**
Series = one labelled column; DataFrame = a table of Series sharing an index.

**Why vectorise?**
Operations run in compiled code across the whole array: faster and shorter than Python loops. (Demonstrated in menu 4.)

**How did you handle missing data?**
Removed rows where identity or exam marks were missing (cannot be guessed); filled low-stakes numeric fields with group medians; flagged every imputed row; logged every decision. Median is robust to outliers.

**Why not clip impossible values to 100?**
A mark of 105 may be a typo for 10.5 or 100; clipping invents a number. Treating it as missing is more honest.

**Mean vs median?**
The median resists outliers; the mean uses all values but is pulled by extremes. Compare them in the histogram: if they differ a lot, the distribution is skewed.

**Population vs sample standard deviation?**
Divide by `n` (population, NumPy default) or `n − 1` (sample, pandas default). `ddof` controls this.

**What is `groupby`?**
Split-apply-combine: split rows into groups, compute something per group, combine into a table.

**Matplotlib vs seaborn: when do you use which?**
Matplotlib is the low-level drawing engine, with full control over every element. seaborn is built on top of it, works directly with DataFrames, and gives statistical charts (box, violin, heatmap) in one call with sensible styling. I use seaborn for statistical charts and Matplotlib for custom touches (reference lines, labels, trend lines), and the two combine because seaborn returns ordinary Matplotlib objects.

**Wide vs long data?**
Wide: one row per entity, one column per measurement. Long: one row per (entity, measurement) pair. seaborn prefers long; `pd.melt` converts wide to long.

**Does the correlation prove studying causes higher marks?**
No. Correlation shows association; confounders (prior knowledge, motivation) can drive both.

**How do you prevent data leakage between steps?**
Copy before modifying; each stage returns a new DataFrame; a test proves the raw data is unchanged.

**How do you make results reproducible?**
Fixed random seed, pinned logic in `config.py`, tests, `requirements.txt`.

**What would you do next?**
Load real (anonymised) data, add statistical tests, a dashboard, and *then* machine learning (for example predicting at-risk students), always keeping the cleaning and testing discipline from this project.

---

*All data is synthetic. Grading rules are examples, not an official policy.*
