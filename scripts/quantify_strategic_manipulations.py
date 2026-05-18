import numpy as np
import copy
import random
import pandas as pd
import os

from fair.allocation import (
    yankee_swap,
    round_robin,
    serial_dictatorship,
    integer_linear_program,
)
from fair.optimization import StudentAllocationProgram
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import qsurvey


status_color_map = {
    1: "lightsteelblue",
    2: "blue",
    3: "forestgreen",
    4: "darkkhaki",
    5: "darkorange",
    6: "red",
}
status_max_course_map = {
    1: 6,
    2: 6,
    3: 6,
    4: 6,
    5: 4,
    6: 4,
}
status_crs_prefix_map = {
    1: ["1", "2", "3"],
    2: ["1", "2", "3", "4"],
    3: ["1", "2", "3", "4", "5"],
    4: ["2", "3", "4", "5", "6"],
    5: ["5", "6"],
    6: ["5", "6"],
}
NUM_STUDENTS_PER_STATUS = {
    1: 239,
    2: 327,
    3: 408,
    4: 573,
    5: 613,
    6: 148,
}

pref_thresh = 100
SPARSE = False

survey_file = "../resources/survey_data.csv"
schedule_file = "../resources/anonymized_courses.xlsx"
mapping_file = "../resources/survey_column_mapping.csv"

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
course_cap_map = {
    crs: crs_sec_cap_map[course_map[crs]["course num"]][int(course_map[crs]["section"])]
    for crs in all_courses
}
all_students = [
    student for student in students if len(student.student.preferred_courses) > 0
]

n_responses_per_status = np.zeros(6)
for student in all_students:
    student_status = int(student_status_map[student])
    n_responses_per_status[student_status - 1] += 1


survey_file = "../resources/survey_data.csv"
schedule_file = "../resources/anonymized_courses.xlsx"
mapping_file = "../resources/survey_column_mapping.csv"

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
course_cap_map = {
    crs: crs_sec_cap_map[course_map[crs]["course num"]][int(course_map[crs]["section"])]
    for crs in all_courses
}
all_students = [
    student for student in students if len(student.student.preferred_courses) > 0
]

n_responses_per_status = np.zeros(6)
for student in all_students:
    student_status = int(student_status_map[student])
    n_responses_per_status[student_status - 1] += 1

rate = 0.1
n_per_status = [round(NUM_STUDENTS_PER_STATUS[i + 1] * rate) for i in range(6)]
for sche in schedule:
    sche.capacity = max(1, round(sche.capacity * rate))

random.seed(0)
reduced_students = []
for status in range(1, 7):
    students_status = [
        student for student in all_students if student_status_map[student] == status
    ]
    selected_students = random.sample(students_status, n_per_status[status - 1])
    reduced_students = [*reduced_students, *selected_students]

students = reduced_students
NUM_STUDENTS = len(students)
print("Num students,", NUM_STUDENTS)

students.sort(key=lambda x: student_status_map[x])
students.reverse()

valuations = np.vstack([student_resp_map[student] for student in students]) - 1

schedule_copy = [copy.copy(item) for item in schedule]
for i in range(len(schedule_copy)):
    schedule_copy[i].capacity = 1

X_OPT = np.zeros([len(schedule), len(students)], dtype=int)
for student_idx in range(len(students)):
    small_ilp_students = students[student_idx : student_idx + 1]
    c_small_ilp = np.array([valuations[student_idx]])
    orig_students = [student.student for student in small_ilp_students]
    program = StudentAllocationProgram(orig_students, schedule_copy).compile()
    opt_alloc = program.formulateUSW(valuations=c_small_ilp.flatten()).solve()
    X_OPT[:, student_idx] = opt_alloc

X_YS = round_robin(students, schedule, valuations=valuations)

OPT_utilities = np.diag(np.dot(valuations, X_OPT))
YS_utilities = np.diag(np.dot(valuations, X_YS))

results_file = "manipulation_results_RR_real.csv"

# create empty file if it does not exist
if not os.path.exists(results_file):

    empty_df = pd.DataFrame(
        columns=[
            "student",
            "student_status",
            "strategy",
            "gain",
            "old_utility",
            "new_utility",
            "regret_before",
            "regret_after",
            "old_bundle",
            "new_bundle",
            "desired_items",
            "true_preferences",
            "reported_preferences",
            "total_usw_old",
            "total_usw_new",
        ]
    )

    empty_df.to_csv(results_file, index=False)


##############################################################################
# CONSTANTS
##############################################################################

MIN_UTILITY = 0
MAX_UTILITY = 7


##############################################################################
# HELPERS
##############################################################################


def clip_utilities(v):
    return np.clip(v, MIN_UTILITY, MAX_UTILITY)


def compute_utilities(valuations, allocation):
    return np.diag(valuations @ allocation)


def get_assigned_items(X, student_idx):
    return np.where(X[:, student_idx] != 0)[0]


def get_missed_opt_items(X_OPT, X_YS, student_idx):
    return np.where((X_OPT[:, student_idx] != 0) & (X_YS[:, student_idx] == 0))[0]


##############################################################################
# HEURISTICS
##############################################################################


def heuristic_boost_missed(true_vals, assigned_items, missed_items, boost=2, penalty=2):

    v = true_vals.copy()

    for c in missed_items:
        if v[c] > 0:
            v[c] = min(MAX_UTILITY, v[c] + boost)

    for c in assigned_items:
        if v[c] > 0:
            v[c] = max(MIN_UTILITY, v[c] - penalty)

    return clip_utilities(v)


def heuristic_only_missed(true_vals, missed_items):

    v = np.zeros_like(true_vals)

    for c in missed_items:
        if true_vals[c] > 0:
            v[c] = MAX_UTILITY

    return clip_utilities(v)


def heuristic_promote_single_missed(true_vals, missed_item):

    v = true_vals.copy()

    v[missed_item] = MAX_UTILITY

    for j in range(len(v)):
        if j != missed_item and v[j] > 0:
            v[j] = max(MIN_UTILITY, v[j] - 1)

    return clip_utilities(v)


def heuristic_demote_assigned(true_vals, assigned_items):

    v = true_vals.copy()

    for c in assigned_items:
        if v[c] > 0:
            v[c] = max(MIN_UTILITY, v[c] - 3)

    return clip_utilities(v)


##############################################################################
# MAIN EXPERIMENT
##############################################################################


def run_manipulation_experiment(
    valuations,
    X_OPT,
    X_YS,
    students,
    schedule,
    threshold=5,
    max_students=5,  # <-- NEW
):
    """
    Runs manipulation experiments on unhappy students.

    Students are processed in decreasing regret order.
    Only the top `max_students` most unhappy students are tested.
    """

    results = []

    ##########################################################################
    # TRUE UTILITIES
    ##########################################################################

    OPT_utilities = compute_utilities(valuations, X_OPT)
    YS_utilities = compute_utilities(valuations, X_YS)

    regret = OPT_utilities - YS_utilities

    YS_usw = float(np.sum(YS_utilities))

    ##########################################################################
    # SORT STUDENTS BY REGRET
    ##########################################################################

    unhappy_students = np.where(regret >= threshold)[0]

    unhappy_students = unhappy_students[np.argsort(regret[unhappy_students])[::-1]]

    ##########################################################################
    # KEEP ONLY TOP max_students
    ##########################################################################

    unhappy_students = unhappy_students[:max_students]

    print(f"\nTesting top {len(unhappy_students)} unhappy students")

    ##########################################################################
    # LOOP OVER STUDENTS
    ##########################################################################

    for s in unhappy_students:

        print("\n==================================================")
        print(f"Student {s}")

        assigned_items = get_assigned_items(X_YS, s)

        missed_items = get_missed_opt_items(X_OPT, X_YS, s)

        true_utility = YS_utilities[s]

        print("Current utility:", true_utility)
        print("OPT utility:", OPT_utilities[s])
        print("Regret:", regret[s])

        print("Assigned items:", assigned_items)
        print("Missed OPT items:", missed_items)

        if len(missed_items) == 0:
            print("No missed OPT items")
            continue

        ######################################################################
        # GENERATE STRATEGIES
        ######################################################################

        strategies = []

        strategies.append(
            (
                "boost_missed",
                heuristic_boost_missed(valuations[s], assigned_items, missed_items),
            )
        )

        strategies.append(
            ("only_missed", heuristic_only_missed(valuations[s], missed_items))
        )

        strategies.append(
            (
                "demote_assigned",
                heuristic_demote_assigned(valuations[s], assigned_items),
            )
        )

        for c in missed_items:

            strategies.append(
                (
                    f"promote_single_{c}",
                    heuristic_promote_single_missed(valuations[s], c),
                )
            )

        ######################################################################
        # TEST STRATEGIES
        ######################################################################

        for strategy_name, fake_vals in strategies:

            print(f"\nTrying strategy: {strategy_name}")

            ##################################################################
            # MODIFY ONLY THIS STUDENT
            ##################################################################

            reported_vals = valuations.copy()

            reported_vals[s] = fake_vals

            ##################################################################
            # RUN YANKEE SWAP
            ##################################################################

            X_new = round_robin(
                students,
                schedule,
                valuations=reported_vals,
            )

            ##################################################################
            # TRUE UTILITY AND USW UNDER NEW ALLOCATION
            ##################################################################

            new_utility = valuations[s] @ X_new[:, s]

            gain = new_utility - true_utility

            new_usw = float(np.sum(compute_utilities(valuations, X_new)))

            new_bundle = get_assigned_items(X_new, s)

            if gain > 0:
                print("PROFITABLE DEVIATION FOUND")
            elif gain == 0:
                print("NO GAIN")
            else:
                print("LOSS:", gain)
            print("Gain:", gain)

            ##################################################################
            # SAVE ONLY ITEMS THE STUDENT ACTUALLY CARES ABOUT
            ##################################################################

            desired_items = np.where(valuations[s] > 0)[0]

            true_preferences = valuations[s, desired_items]
            reported_preferences = fake_vals[desired_items]

            ##################################################################
            # STORE RESULT
            ##################################################################

            result_row = {
                "student": s,
                "student_status": student_status_map[students[s]],
                "strategy": strategy_name,
                "gain": gain,
                "old_utility": true_utility,
                "new_utility": new_utility,
                "regret_before": regret[s],
                "regret_after": OPT_utilities[s] - new_utility,
                "old_bundle": assigned_items.tolist(),
                "new_bundle": new_bundle.tolist(),
                "desired_items": desired_items.tolist(),
                "true_preferences": true_preferences.tolist(),
                "reported_preferences": reported_preferences.tolist(),
                "total_usw_old": YS_usw,
                "total_usw_new": new_usw,
            }
            results.append(result_row)

            pd.DataFrame([result_row]).to_csv(
                results_file, mode="a", header=False, index=False
            )

    return pd.DataFrame(results)


df = run_manipulation_experiment(
    valuations, X_OPT, X_YS, students, schedule, threshold=1, max_students=200
)
