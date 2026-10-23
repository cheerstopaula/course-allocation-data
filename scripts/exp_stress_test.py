"""
Stress-test experiment: fixed course capacities, increasing cohort sizes.
Cohort sizes to run: 2000 3000 4000 5000 8000 10000
"""

import numpy as np
import pandas as pd
import time
import os

from fair.stats.survey import Corpus, SingleTopicSurvey
from fair.agent import LegacyStudent
from fair.allocation import (
    yankee_swap,
    round_robin,
    serial_dictatorship,
    integer_linear_program,
)
from fair.welfare_metrics import utilitarian_welfare, nash_welfare
from fair.fairness_metrics import EF_violations_responses, PMMS_violations_responses
import qsurvey

HERE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results", "stress"
)  # outputs go to results/stress/
os.makedirs(HERE, exist_ok=True)
RESOURCES = os.path.join(HERE, "..", "..", "resources")

os.environ["PYTHONHASHSEED"] = "0"

COHORT_SIZE = 2000

ILP_MAX_COHORT = 20000  # skip ILP above this
PMMS_MAX_COHORT = 4000  # skip PMMS above this (O(n^2))
SKIP_ILP = COHORT_SIZE > ILP_MAX_COHORT
SKIP_PMMS = COHORT_SIZE > PMMS_MAX_COHORT

NUM_SUB_KERNELS = 3
SAMPLE_PER_STUDENT = 10
SPARSE = False
pref_thresh = 100

status_max_course_map = {1: 6, 2: 6, 3: 6, 4: 6, 5: 4, 6: 4}
status_crs_prefix_map = {
    1: ["1", "2", "3"],
    2: ["1", "2", "3", "4"],
    3: ["1", "2", "3", "4", "5"],
    4: ["2", "3", "4", "5", "6"],
    5: ["5", "6"],
    6: ["5", "6"],
}
FULL_NUM_STUDENTS_PER_STATUS = {1: 239, 2: 327, 3: 408, 4: 573, 5: 613, 6: 148}
FULL_COHORT = sum(FULL_NUM_STUDENTS_PER_STATUS.values())  # 2308

survey_file = os.path.join(RESOURCES, "survey_data.csv")
schedule_file = os.path.join(RESOURCES, "anonymized_courses.xlsx")
mapping_file = os.path.join(RESOURCES, "survey_column_mapping.csv")
csv_file_path = os.path.join(HERE, "stress_experiment.csv")

print(
    f"=== Stress test: cohort={COHORT_SIZE}, skip_ILP={SKIP_ILP}, skip_PMMS={SKIP_PMMS} ==="
)

mp = qsurvey.QMapper(mapping_file)
qd = qsurvey.QSchedule(schedule_file)
crs_sec_cap_map = qd.capacities()
qs = qsurvey.QSurvey(survey_file, mp, list(crs_sec_cap_map.keys()))
course_map = mp.mapping(qs.all_courses)
all_courses = [crs for crs in course_map.keys()]
features = mp.features(course_map)
course, slot, weekday, section = features
schedule = mp.schedule(course_map, crs_sec_cap_map, features)
real_students, responses, statuses = qs.students(
    course_map,
    all_courses,
    features,
    schedule,
    status_max_course_map,
    pref_thresh,
    SPARSE,
)
student_status_map = {real_students[i]: status for i, status in enumerate(statuses)}
student_resp_map = {real_students[i]: response for i, response in enumerate(responses)}
real_students = [s for s in real_students if len(s.student.preferred_courses) > 0]

# Fixed original capacities — never scale them
original_capacities = [sche.capacity for sche in schedule]

# How many of each status to target
rate_prop = COHORT_SIZE / FULL_COHORT
NUM_STUDENTS_PER_STATUS = {
    s: max(1, round(n * rate_prop)) for s, n in FULL_NUM_STUDENTS_PER_STATUS.items()
}

n_responses_per_status = np.zeros(6)
for student in real_students:
    n_responses_per_status[int(student_status_map[student]) - 1] += 1

NUM_RAND_SAMP = {
    i + 1: max(0, NUM_STUDENTS_PER_STATUS[i + 1] - int(n_responses_per_status[i]))
    for i in range(6)
}

USE_SMALL = COHORT_SIZE <= len(real_students)
print(
    f"mode={'small' if USE_SMALL else 'big'}, target_per_status={NUM_STUDENTS_PER_STATUS}"
)


def save_row(NUM_STUDENTS, seed, alg, runtime, X, students, schedule, c):
    n = NUM_STUDENTS
    USW = utilitarian_welfare(X, students, schedule, valuations=c)
    seats = sum(sum(X)[:n])
    zeros, nash = nash_welfare(X, students, schedule, valuations=c)

    (total_envy, _, status_envy, _, downward_envy, _, EF_matrix, memo) = (
        EF_violations_responses(
            X, students, schedule, valuations=c, student_status_map=student_status_map
        )
    )

    total_pmms = downward_pmms = np.nan
    if not SKIP_PMMS:
        (total_pmms, _, _, _, downward_pmms, _, _) = PMMS_violations_responses(
            X,
            students,
            schedule,
            valuations=c,
            student_status_map=student_status_map,
            EF_matrix=EF_matrix,
            memo=memo,
        )

    row = pd.DataFrame(
        {
            "NUM_STUDENTS": [n],
            "seed": [seed],
            "alg": [alg],
            "USW": [USW],
            "seats": [seats],
            "zeros": [zeros],
            "nash": [nash],
            "total_envy": [total_envy],
            "status_envy": [status_envy],
            "downward_envy": [downward_envy],
            "total_pmms": [total_pmms],
            "downward_pmms": [downward_pmms],
            "runtime": [runtime],
        }
    )
    file_exists = os.path.isfile(csv_file_path)
    row.to_csv(csv_file_path, mode="a", header=not file_exists, index=False)


for seed in range(10):
    print(f"\n--- seed {seed} ---")
    RNG = np.random.default_rng(seed)

    # Restore original capacities (never scale)
    for i, sche in enumerate(schedule):
        sche.capacity = original_capacities[i]

    student_type_map = {student: "real" for student in real_students}

    if USE_SMALL:
        students = []
        for status in range(1, 7):
            pool = [s for s in real_students if student_status_map[s] == status]
            n = min(NUM_STUDENTS_PER_STATUS[status], len(pool))
            idxs = RNG.choice(len(pool), size=n, replace=False)
            students.extend(pool[i] for i in idxs)
    else:
        students = [*real_students]
        status_mbeta_map = {}
        status_surveys_map = {}
        status_students_map = {}

        for status in range(1, 7):
            relevant_idxs = qsurvey.get_status_relevant(
                status, all_courses, course_map, status_crs_prefix_map
            )
            status_students = [
                s for s in real_students if student_status_map[s] == status
            ]
            status_students_map[status] = status_students
            status_surveys = [
                SingleTopicSurvey(
                    [sch for i, sch in enumerate(schedule) if i in relevant_idxs],
                    student_resp_map[student][relevant_idxs],
                    student.student.total_courses,
                    1,
                    8,
                )
                for student in status_students
            ]
            status_corpus = Corpus(status_surveys, RNG)
            status_mbeta_map[status] = status_corpus.kde_distribution(
                SAMPLE_PER_STUDENT, NUM_SUB_KERNELS
            )
            status_surveys_map[status] = status_surveys

        for status in qsurvey.STATUS_LABEL_MAP.keys():
            synth_students, data = qsurvey.synthesize_students(
                NUM_RAND_SAMP[status],
                course,
                section,
                features,
                schedule,
                qs,
                status_surveys_map[status],
                status_mbeta_map[status],
                course_map,
                status_max_course_map[status],
                qsurvey.get_status_relevant(
                    status, all_courses, course_map, status_crs_prefix_map
                ),
                rng=RNG,
                pref_thresh=pref_thresh,
                total_course_list=[
                    s.student.total_courses for s in status_students_map[status]
                ],
            )
            synth_students = [
                LegacyStudent(s, s.preferred_courses, course) for s in synth_students
            ]
            if synth_students:
                data1 = data[-len(synth_students) :]
                for i, resp in enumerate(data1):
                    student_resp_map[synth_students[i]] = [int(v) for v in resp]
                    student_type_map[synth_students[i]] = "synth"
                students = [*students, *synth_students]
                for s in synth_students:
                    student_status_map[s] = status

    creation_order = {s: i for i, s in enumerate(students)}
    students.sort(key=lambda x: (student_status_map[x], creation_order[x]))
    students = list(reversed(students))

    c = np.vstack([student_resp_map[s] for s in students]) - 1
    NUM_STUDENTS = len(students)
    print(f"  NUM_STUDENTS={NUM_STUDENTS}")

    if not SKIP_ILP:
        print("  run ILP")
        t0 = time.time()
        X_ILP = integer_linear_program(students, schedule, valuations=c)
        rt = time.time() - t0
        print(f"  ILP done in {rt:.1f}s")
        save_row(NUM_STUDENTS, seed, "ILP", rt, X_ILP, students, schedule, c)
    else:
        print("  ILP skipped (cohort too large)")

    print("  run SD")
    t0 = time.time()
    X_SD = serial_dictatorship(students, schedule, c)
    rt = time.time() - t0
    print(f"  SD done in {rt:.1f}s")
    save_row(NUM_STUDENTS, seed, "SD", rt, X_SD, students, schedule, c)

    print("  run RR")
    t0 = time.time()
    X_RR = round_robin(students, schedule, c)
    rt = time.time() - t0
    print(f"  RR done in {rt:.1f}s")
    save_row(NUM_STUDENTS, seed, "RR", rt, X_RR, students, schedule, c)

    print("  run YS")
    t0 = time.time()
    X_YS = yankee_swap(students, schedule, valuations=c)
    rt = time.time() - t0
    print(f"  YS done in {rt:.1f}s")
    save_row(NUM_STUDENTS, seed, "YS", rt, X_YS, students, schedule, c)

print(f"\nDone. Results saved to {csv_file_path}")
