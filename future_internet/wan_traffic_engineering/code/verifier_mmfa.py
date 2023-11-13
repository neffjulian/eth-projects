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

try:
    from . import wanteutility
except (ImportError, SystemError):
    import wanteutility

try:
    from . import verifier_correctness
except (ImportError, SystemError):
    import verifier_correctness

PRECISION = 0.001


def verify_mmfa(graph_filename, demand_filename, path_filename, rate_filename, ground_truth_rate_filename, k_limit):

    verification = verifier_correctness.verify(graph_filename, demand_filename, path_filename, rate_filename, k_limit)
    if not verification[0]:
        return verification

    with open(rate_filename, "r") as rate_file, open(ground_truth_rate_filename, "r") as ground_truth_rate_file:
        for line_rate, line_ground_truth_rate in zip(rate_file, ground_truth_rate_file):
            rate = float(line_rate)
            ground_truth_rate = float(line_ground_truth_rate)
            if abs(rate - ground_truth_rate) > PRECISION:
                return False, "ERROR: incorrect MMFA rate (got %f but expected %f)" % (rate, ground_truth_rate)
        if rate_file.readline() != "" or ground_truth_rate_file.readline() != "":
            return False, "Different length rate and ground truth rate file."

    return verification
