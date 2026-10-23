import argparse
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
import qsurvey

HERE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results", "time"
)  # outputs go to results/time/
os.makedirs(HERE, exist_ok=True)
RESOURCES = os.path.join(HERE, "..", "..", "resources")

os.environ["PYTHONHASHSEED"] = "0"

COHORT_SIZE = 2000

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
csv_file_path = os.path.join(HERE, "time_experiment.csv")

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

# Proportionally scale target counts per status
rate_prop = COHORT_SIZE / FULL_COHORT
NUM_STUDENTS_PER_STATUS = {
    s: max(1, round(n * rate_prop)) for s, n in FULL_NUM_STUDENTS_PER_STATUS.items()
}

# How many synthetic students are needed per status (for big mode)
n_responses_per_status = np.zeros(6)
for student in real_students:
    n_responses_per_status[int(student_status_map[student]) - 1] += 1
NUM_RAND_SAMP = {
    i + 1: max(0, NUM_STUDENTS_PER_STATUS[i + 1] - int(n_responses_per_status[i]))
    for i in range(6)
}

# Small mode: subsample real students. Big mode: synthesize students.
USE_SMALL = COHORT_SIZE <= len(real_students)
print(
    f"cohort_size={COHORT_SIZE}, n_real={len(real_students)}, mode={'small' if USE_SMALL else 'big'}"
)

original_capacities = [sche.capacity for sche in schedule]


def save_runtime(NUM_STUDENTS, seed, alg, runtime):
    file_exists = os.path.isfile(csv_file_path)
    new_row = pd.DataFrame(
        {
            "NUM_STUDENTS": [NUM_STUDENTS],
            "seed": [seed],
            "alg": [alg],
            "runtime": [runtime],
        }
    )
    if file_exists:
        new_row.to_csv(csv_file_path, mode="a", header=False, index=False)
    else:
        new_row.to_csv(csv_file_path, mode="w", header=True, index=False)


for seed in range(10):
    RNG = np.random.default_rng(seed)

    # Restore original capacities at the start of each iteration
    for i, sche in enumerate(schedule):
        sche.capacity = original_capacities[i]

    student_type_map = {student: "real" for student in real_students}

    if USE_SMALL:
        # Subsample real students proportionally by status
        students = []
        for status in range(1, 7):
            status_pool = [s for s in real_students if student_status_map[s] == status]
            n = min(NUM_STUDENTS_PER_STATUS[status], len(status_pool))
            idxs = RNG.choice(len(status_pool), size=n, replace=False)
            students.extend(status_pool[i] for i in idxs)
    else:
        # Synthesize students to reach target cohort size
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
            status_mbeta = status_corpus.kde_distribution(
                SAMPLE_PER_STUDENT, NUM_SUB_KERNELS
            )
            status_mbeta_map[status] = status_mbeta
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
                for i, response in enumerate(data1):
                    student_resp_map[synth_students[i]] = [int(v) for v in response]
                    student_type_map[synth_students[i]] = "synth"
                students = [*students, *synth_students]
                for student in synth_students:
                    student_status_map[student] = status

    # Scale capacities proportionally to actual cohort size
    actual_cohort = len(students)
    cap_rate = actual_cohort / FULL_COHORT
    for i, sche in enumerate(schedule):
        sche.capacity = max(1, round(original_capacities[i] * cap_rate))

    creation_order = {s: i for i, s in enumerate(students)}
    students.sort(key=lambda x: (student_status_map[x], creation_order[x]))
    students = list(reversed(students))

    c = np.vstack([student_resp_map[student] for student in students]) - 1
    NUM_STUDENTS = len(students)
    print(f"seed={seed}, NUM_STUDENTS={NUM_STUDENTS}")

    print("run ILP")
    start = time.time()
    X_ILP = integer_linear_program(students, schedule, valuations=c)
    runtime = time.time() - start
    print(f"ILP runtime = {runtime}")
    save_runtime(NUM_STUDENTS, seed, "ILP", runtime)

    print("run SD")
    start = time.time()
    X_SD = serial_dictatorship(students, schedule, c)
    runtime = time.time() - start
    print(f"SD runtime = {runtime}")
    save_runtime(NUM_STUDENTS, seed, "SD", runtime)

    print("run RR")
    start = time.time()
    X_RR = round_robin(students, schedule, c)
    runtime = time.time() - start
    print(f"RR runtime = {runtime}")
    save_runtime(NUM_STUDENTS, seed, "RR", runtime)

    print("run YS")
    start = time.time()
    X_YS = yankee_swap(students, schedule, valuations=c)
    runtime = time.time() - start
    print(f"YS runtime = {runtime}")
    save_runtime(NUM_STUDENTS, seed, "YS", runtime)
