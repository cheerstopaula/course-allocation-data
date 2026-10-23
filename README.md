# Course Allocation Data

This repository contains course-preference data collected from students of the Computer Science department at the University of Massachusetts Amherst (Manning College of Information and Computer Sciences, CICS) for the Fall 2024 semester, together with the code that processes the data and runs the course-allocation experiments.

The allocation mechanisms (Yankee Swap, Round Robin, Serial Dictatorship, and an integer linear program) and the welfare and fairness metrics are implemented in the [yankee-swap-allocation-framework](https://github.com/cheerstopaula/yankee-swap-allocation-framework) repository, which is installed automatically as a dependency.

## Contents

```
.
├── resources/                     # survey responses, course schedule, column mapping
│   ├── survey_data.csv
│   ├── anonymized_courses.xlsx
│   ├── survey_column_mapping.csv
│   └── random_survey.csv
├── scripts/                       # experiments (results go to results/<experiment>/, created when a script runs)
│   ├── exp_full.py                # real + synthetic students (full department scale)
│   ├── exp_reduced.py             # real students only
│   ├── exp_stress_test.py         # fixed capacities, growing number of students
│   └── exp_time.py                # runtime, capacities scaled with the number of students
├── src/qsurvey/                   # parses the survey and schedule, builds students, generates synthetic students
├── requirements.txt
└── pyproject.toml
```

## The survey

We invited all undergraduate and graduate students in the department to participate in May 2024, shortly after the final examination period, when students had recently indicated their course preferences for Fall 2024 on the university portal. Participation was voluntary, and the study was approved by the university's Institutional Review Board. The survey consisted of the following questions:

1. Current status: 1: Freshman, 2: Sophomore, 3: Junior, 4: Senior, 5: MS, 6: PhD.
2. How many CICS classes are you planning to enroll in before the add/drop period of the Fall 2024 semester? (You might want to take more classes than you are planning to keep by the end of the semester.)
3. How many CICS classes are you planning to stay enrolled in after the add/drop period of the Fall 2024 semester?
4. Which categories of courses are you interested in taking? Select all that apply.
5. Are there particular times of the day you would prefer not to have classes? For each time slot (before 9:00 AM, each hour from 9:00 AM to 6:00 PM, and after 6:00 PM), respondents selected the days (1: Monday, ..., 5: Friday) on which the slot is acceptable.
6. Email address (used only to pay respondents; **not included in the data**).
7. For each course offered to the respondent's academic status: rate your interest on a scale from 1 (not interested) to 7 (very interested), or 8 if the course is required for your program.

## Data files

- `resources/survey_data.csv`: 1,061 responses. Besides the answers to the questions above (columns `1`–`5` and the course ratings of question 7), each response includes:
    - `ResponseId`: a unique identifier for each response, assigned by the survey platform;
    - `Finished`: whether the survey was fully completed;
    - `Progress`: the completion percentage;
    - `Duration (in seconds)`: the time taken to complete the survey.
- `resources/anonymized_courses.xlsx`: the 96 course sections offered in Fall 2024, anonymized so that specific classes and instructors cannot be identified:
    - `Catalog`: an anonymized identifier for each course (not the actual course number);
    - `Subject`: the department offering the course (anonymized);
    - `Categories`: the course level (undergraduate or graduate);
    - `Enrl Capacity`: the number of seats;
    - `Section`: the section of the course, which may vary by schedule or instructor;
    - `zc.days`: the days of the week the class meets;
    - `Mtg Time`: the time slot of the class;
    - `InstructorPrint`: the teaching staff (anonymized);
    - `Credits`: the number of credits.
- `resources/survey_column_mapping.csv`: maps the survey's columns to questions and course sections in `anonymized_courses.xlsx`.
- `resources/random_survey.csv`: a small, randomly generated survey with the same structure as `survey_data.csv`, for testing.

## Setup

Python 3.10 or later (tested with 3.12).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` also installs the `fair` package from the [yankee-swap-allocation-framework](https://github.com/cheerstopaula/yankee-swap-allocation-framework) repository. `requirements.txt` pins the versions we used. Use SciPy 1.15.2 or later (versions 1.15.0 and 1.15.1 have a quadratic-time loop in `scipy.optimize.milp` that makes the ILP extremely slow). 

## Running the experiments

The scripts can be run from any directory:

```bash
python scripts/exp_<experiment>.py    # <experiment>: full, reduced, stress_test, or time
```

| Script | Instance | Output in `results/` |
|---|---|---|
| `exp_full.py` | All 700 real respondents, complemented with synthetic students to match the department's population (2,308 students) | `full/full_experiment.csv`, `full/data/data_<seed>.npz` |
| `exp_reduced.py` | Real students only: in each sample, 20.4% of each academic status (471 students), with course capacities scaled by the same rate | `reduced/reduced_experiment.csv`, `reduced/data/data_<seed>.npz` |
| `exp_stress_test.py` | Real course capacities (7,389 seats), growing number of students (2,000 to 10,000, set by `COHORT_SIZE`) | `stress/stress_experiment.csv` |
| `exp_time.py` | Course capacities scaled with the number of students (100 to 15,000, set by `COHORT_SIZE`); runtimes only | `time/time_experiment.csv` |

The CSV files contain one row per sample and mechanism, with the welfare and fairness metrics; the `.npz` files contain each sample's valuations and the allocation computed by each mechanism. Rerunning a script appends to its CSV and overwrites the `.npz` files of the seeds it runs, so move previous results aside first.

### Runtimes

Algorithm runtimes per seed on our cluster (AMD EPYC 7763 nodes, medians):

| Instance | ILP | SD | RR | YS |
|---|---|---|---|---|
| Real students only (471 students) | 2 s | 2 s | 32 s | 51 s |
| Full department (2,308 students) | 20 s | 8 s | 168 s | 268 s |

The fairness metrics (envy, and the exact PMMS check) take longer than the allocation algorithms: one full-department seed takes about an hour end to end on a laptop.

## Reproducibility notes

* Yankee Swap, Round Robin, and Serial Dictatorship are deterministic given the instance.
* Synthetic students are sampled from a kernel density estimate built from multivariate Beta distributions fitted to the real responses of each academic status. The sampling is deterministic on a given platform, but not bit-identical across platforms and numerical-library builds, so results for instances with synthetic students can differ slightly between machines.
* The ILP may return a different optimal allocation with the same objective value, since ties between optimal solutions are broken by the solver.
* The exact PMMS check solves an integer program per pair of students; solver tolerances can change a few borderline comparisons.

## Citing

If you use this data or code, please cite the paper that introduces the dataset and the software:

> George Bissias, Cyrus Cousins, Paula Navarrete Diaz, and Yair Zick. *Deploying Fair and Efficient Course Allocation Mechanisms*. arXiv:2502.10592, 2025.

```bibtex
@misc{bissias2025deploying,
  title         = {Deploying Fair and Efficient Course Allocation Mechanisms},
  author        = {Bissias, George and Cousins, Cyrus and Navarrete Diaz, Paula and Zick, Yair},
  year          = {2025},
  eprint        = {2502.10592},
  archivePrefix = {arXiv},
  primaryClass  = {cs.GT},
  url           = {https://arxiv.org/abs/2502.10592}
}
```

This version of the paper predates the extension to non-binary preferences; we will update this reference when the new version is available.

## License

This project is open source and available under the MIT License.
