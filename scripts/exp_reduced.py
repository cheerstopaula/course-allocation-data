"""Reduced experiment: real students only, sampled per status at the lowest response rate,
with course capacities scaled by the same rate. Metrics are computed exactly as in exp_full.py. 
"""

import os
import random
import sys
import time

import numpy as np
import pandas as pd

from fair.allocation import (
    yankee_swap,
    round_robin,
    serial_dictatorship,
    integer_linear_program,
)
from fair.welfare_metrics import (
    utilitarian_welfare,
    nash_welfare,
    first_preference_count,
)
from fair.fairness_metrics import (
    EF_violations_responses,
    EF1_violations_responses,
    PMMS_violations_responses,
)
import qsurvey

HERE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results", "reduced"
)  # outputs go to results/reduced/
os.makedirs(HERE, exist_ok=True)
RESOURCES = os.path.join(HERE, "..", "..", "resources")

SPARSE = False
pref_thresh = 100

status_max_course_map = {1: 6, 2: 6, 3: 6, 4: 6, 5: 4, 6: 4}
NUM_STUDENTS_PER_STATUS = {1: 239, 2: 327, 3: 408, 4: 573, 5: 613, 6: 148}

survey_file = os.path.join(RESOURCES, "survey_data.csv")
schedule_file = os.path.join(RESOURCES, "anonymized_courses.xlsx")
mapping_file = os.path.join(RESOURCES, "survey_column_mapping.csv")
data_dir = os.path.join(HERE, "data")
os.makedirs(data_dir, exist_ok=True)
csv_file_path = os.path.join(HERE, "reduced_experiment.csv")

mp = qsurvey.QMapper(mapping_file)
qd = qsurvey.QSchedule(schedule_file)
crs_sec_cap_map = qd.capacities()
qs = qsurvey.QSurvey(survey_file, mp, list(crs_sec_cap_map.keys()))
course_map = mp.mapping(qs.all_courses)
all_courses = [crs for crs in course_map.keys()]
features = mp.features(course_map)
course, slot, weekday, section = features
schedule = mp.schedule(course_map, crs_sec_cap_map, features)
students, responses, statuses = qs.students(
    course_map,
    all_courses,
    features,
    schedule,
    status_max_course_map,
    pref_thresh,
    SPARSE,
)
student_status_map = {students[i]: status for i, status in enumerate(statuses)}
student_resp_map = {students[i]: response for i, response in enumerate(responses)}
all_students = [
    student for student in students if len(student.student.preferred_courses) > 0
]

n_responses_per_status = np.zeros(6)
for student in all_students:
    n_responses_per_status[int(student_status_map[student]) - 1] += 1

rates = [n_responses_per_status[i] / NUM_STUDENTS_PER_STATUS[i + 1] for i in range(6)]
rate = min(rates)
n_per_status = [round(NUM_STUDENTS_PER_STATUS[i + 1] * rate) for i in range(6)]

for sche in schedule:
    sche.capacity = round(sche.capacity * rate)


def add_experiment_result(NUM_STUDENTS, seed, alg, runtime, X, students, schedule, c):
    """Same metrics and columns as exp_full.py (plus pref_thresh)."""
    USW = utilitarian_welfare(X, students, schedule, valuations=c)
    seats = sum(sum(X)[:NUM_STUDENTS])
    zeros, nash = nash_welfare(X, students, schedule, valuations=c)
    first_count = first_preference_count(X, c)
    (
        total_envy,
        num_envious,
        status_envy,
        num_status_envious,
        downward_envy,
        num_downward_envious,
        EF_matrix,
        memo,
    ) = EF_violations_responses(
        X, students, schedule, valuations=c, student_status_map=student_status_map
    )
    (
        total_ef1,
        num_ef1_envious,
        status_ef1,
        num_status_ef1_envious,
        downward_ef1,
        num_downward_ef1_envious,
        _,
        _,
    ) = EF1_violations_responses(
        X,
        students,
        schedule,
        valuations=c,
        student_status_map=student_status_map,
        EF_matrix=EF_matrix,
        memo=memo,
    )
    (
        total_pmms,
        num_pmms_envious,
        status_pmms,
        num_status_pmms,
        downward_pmms,
        num_downward_pmms,
        _,
    ) = PMMS_violations_responses(
        X,
        students,
        schedule,
        valuations=c,
        student_status_map=student_status_map,
        EF_matrix=EF_matrix,
        memo=memo,
    )

    new_row = pd.DataFrame(
        {
            "NUM_STUDENTS": [NUM_STUDENTS],
            "pref_thresh": [pref_thresh],
            "seed": [seed],
            "alg": [alg],
            "USW": [USW],
            "seats": [seats],
            "zeros": [zeros],
            "nash": [nash],
            "first": [first_count],
            "total_envy": [total_envy],
            "num_envious": [num_envious],
            "status_envy": [status_envy],
            "num_status_envious": [num_status_envious],
            "downward_envy": [downward_envy],
            "num_downward_envious": [num_downward_envious],
            "total_ef1": [total_ef1],
            "num_ef1_envious": [num_ef1_envious],
            "status_ef1": [status_ef1],
            "num_status_ef1_envious": [num_status_ef1_envious],
            "downward_ef1": [downward_ef1],
            "num_downward_ef1_envious": [num_downward_ef1_envious],
            "total_pmms": [total_pmms],
            "num_pmms_envious": [num_pmms_envious],
            "status_pmms": [status_pmms],
            "num_status_pmms": [num_status_pmms],
            "downward_pmms": [downward_pmms],
            "num_downward_pmms": [num_downward_pmms],
            "runtime": [runtime],
        }
    )
    new_row.to_csv(
        csv_file_path,
        mode="a" if os.path.isfile(csv_file_path) else "w",
        header=not os.path.isfile(csv_file_path),
        index=False,
    )


for seed in range(100):
    # Same sampling as the original reduced runs, so a seed draws the same students.
    random.seed(seed)
    reduced_students = []
    for status in range(1, 7):
        students_status = [
            student for student in all_students if student_status_map[student] == status
        ]
        selected_students = random.sample(students_status, n_per_status[status - 1])
        reduced_students = [*reduced_students, *selected_students]

    students = reduced_students
    NUM_STUDENTS = len(students)
    students.sort(key=lambda x: student_status_map[x])
    students.reverse()
    c = np.vstack([student_resp_map[student] for student in students]) - 1
    print(f"seed {seed}: {NUM_STUDENTS} students", flush=True)

    allocations = {}
    for alg, run in (
        ("ILP", lambda: integer_linear_program(students, schedule, valuations=c)),
        ("SD", lambda: serial_dictatorship(students, schedule, c)),
        ("RR", lambda: round_robin(students, schedule, c)),
        ("YS", lambda: yankee_swap(students, schedule, valuations=c)),
    ):
        start = time.time()
        X = run()
        runtime = time.time() - start
        add_experiment_result(
            NUM_STUDENTS, seed, alg, runtime, X, students, schedule, c
        )
        allocations[f"X_{alg}"] = X
        print(f"  {alg}: {runtime:.1f}s", flush=True)

    np.savez(
        f"{data_dir}/data_{seed}.npz",
        c=c,
        status=np.array([int(student_status_map[s]) for s in students]),
        **allocations,
    )
