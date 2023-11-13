# The MIT License (MIT)
#
# Copyright (c) 2019 Simon Kassing (ETH)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os.path
import sys

try:
    from . import verifier_mmfa
except (ImportError, SystemError):
    import verifier_mmfa

try:
    from . import verifier_max_min_lp
except (ImportError, SystemError):
    import verifier_max_min_lp

try:
    from . import verifier_max_min_lp_objective_only
except (ImportError, SystemError):
    import verifier_max_min_lp_objective_only

try:
    from . import assignment_parameters
except (ImportError, SystemError):
    import assignment_parameters


def check_solution_present(files):
    for f in files:
        if not os.path.isfile(f):
            return False, "ERROR: file %s is missing." % f
    return True, "SUCCESS"

def evaluate_myself(ground_truth_folder, student_folder):

    # Part A: Check all test cases
    part_a_results = []
    for i in range(assignment_parameters.num_tests_a_public):
        check = check_solution_present([
            "%s/input/a/graph%d.graph" % (ground_truth_folder, i),
            "%s/input/a/demand%d.demand" % (ground_truth_folder, i),
            "%s/input/a/path%d.path" % (ground_truth_folder, i),
            "%s/output/a/rate%d.rate" % (ground_truth_folder, i),
            "%s/output/a/rate%d.rate" % (student_folder, i)
        ])
        if check[0]:
            part_a_results.append(verifier_mmfa.verify_mmfa(
                "%s/input/a/graph%d.graph" % (ground_truth_folder, i),
                "%s/input/a/demand%d.demand" % (ground_truth_folder, i),
                "%s/input/a/path%d.path" % (ground_truth_folder, i),
                "%s/output/a/rate%d.rate" % (ground_truth_folder, i),
                "%s/output/a/rate%d.rate" % (student_folder, i),
                500
            ))
        else:
            part_a_results.append(check)

    # Part B: Check all test cases
    part_b_results = []
    for i in range(assignment_parameters.num_tests_b_public):
        check = check_solution_present([
            "%s/input/b/graph%d.graph" % (ground_truth_folder, i),
            "%s/input/b/demand%d.demand" % (ground_truth_folder, i),
            "%s/input/b/path%d.path" % (ground_truth_folder, i),
            "%s/output/b/rate%d.rate" % (ground_truth_folder, i),
            "%s/output/b/rate%d.rate" % (student_folder, i)
        ])
        if check[0]:
            part_b_results.append(verifier_max_min_lp.verify_max_min_lp(
                "%s/input/b/graph%d.graph" % (ground_truth_folder, i),
                "%s/input/b/demand%d.demand" % (ground_truth_folder, i),
                "%s/input/b/path%d.path" % (ground_truth_folder, i),
                "%s/output/b/rate%d.rate" % (ground_truth_folder, i),
                "%s/output/b/rate%d.rate" % (student_folder, i),
                500
            ))
        else:
            part_b_results.append(check)

    # Part C: Retrieve all solutions
    part_c_results = []
    for i in range(assignment_parameters.num_tests_c):
        check = check_solution_present([
            "%s/input/c/graph%d.graph" % (ground_truth_folder, i),
            "%s/input/c/demand.demand" % ground_truth_folder,
            "%s/output/c/path%d.path" % (student_folder, i),
            "%s/output/c/rate%d.rate" % (student_folder, i)
        ])
        if check[0]:
            part_c_results.append(verifier_max_min_lp_objective_only.verify_max_min_lp_objective(
                "%s/input/c/graph%d.graph" % (ground_truth_folder, i),
                "%s/input/c/demand.demand" % ground_truth_folder,
                "%s/output/c/path%d.path" % (student_folder, i),
                "%s/output/c/rate%d.rate" % (student_folder, i),
                assignment_parameters.part_c_k_limit
            ))
        else:
            part_c_results.append(check)

    # Evaluation results
    print("----------------------")
    print("Evaluation results")
    print("----------------------")
    print("Part A:")
    success_a = 0
    i = 0
    for result in part_a_results:
        if result[0]:
            success_a += 1
            print("%2d: PASSED" % i)
        else:
            print("%2d: FAILED (%s)" % (i, result[1]))
        i += 1
    print("----------------------")
    print("Part B:")
    success_b = 0
    i = 0
    for result in part_b_results:
        if result[0]:
            success_b += 1
            print("%2d: PASSED" % i)
        else:
            print("%2d: FAILED (%s)" % (i, result[1]))
        i += 1
    print("----------------------")
    print("Part C:")
    sum_c = 0.0
    i = 0
    for result in part_c_results:
        if result[0]:
            sum_c += result[1]
            print("%2d: PASSED (score: %f)" % (i, result[1]))
        else:
            print("%2d: FAILED (%s)" % (i, result[1]))
        i += 1
    print("----------------------")
    print("Summary:")
    print(" > Part A: %d/%d" % (success_a, assignment_parameters.num_tests_a_public))
    print(" > Part B: %d/%d" % (success_b, assignment_parameters.num_tests_b_public))
    print(" > Part C-public: %f" % (sum_c / float(assignment_parameters.num_tests_c)))
    print("----------------------")


##
# Main function called.
##
def main():
    args = sys.argv[1:]
    if len(args) != 0:
        print("Usage: python evaluator_myself.py")
    else:
        evaluate_myself("../ground_truth", "../myself")


if __name__ == "__main__":
    main()
