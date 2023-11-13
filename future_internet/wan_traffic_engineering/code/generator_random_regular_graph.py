# The MIT License (MIT)
#
# Copyright (c) 2017 Asaf Valadarsky, Gal Shahaf, Michael Shapira, Simon Kassing
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


import sys
import numpy as np


MAX_TRIES_BEFORE_CLIQUE_CONCLUSION = 1000
MAX_TRIES_BEFORE_IMPOSSIBLE_CONCLUSION = 1000


##
# Create a well-connected random graph by randomly connecting switches
# and afterwards checking that it is a well-connected.
#
# @param    n           Number of switches
# @param    d           Network degree of switch
# @param    seed        (Optional) PRNG seed
#
# @return   Adjacency matrix
##
def random_regular_graph(n, d, seed=None):

    # Set numpy's global random seed
    if seed is not None:
        np.random.seed(seed)

    for i in range(MAX_TRIES_BEFORE_IMPOSSIBLE_CONCLUSION):

        # Resulting adjacency matrix
        adjacency_matrix = np.zeros((n, n))

        # Randomly create bi-directional links in the graph
        links_filled = np.zeros(n)
        remaining = list(range(0, n))
        while len(remaining) > 1:

            # Choose source and destination
            tries = 0
            src_idx = np.random.randint(len(remaining))
            dst_idx = np.random.randint(len(remaining))
            while ((src_idx == dst_idx or adjacency_matrix[remaining[src_idx]][remaining[dst_idx]] == 1)
                   and tries < MAX_TRIES_BEFORE_CLIQUE_CONCLUSION):
                dst_idx = np.random.randint(len(remaining))
                tries = tries + 1
            src = remaining[src_idx]
            dst = remaining[dst_idx]

            # If we continued to be stuck in a clique,
            # we conclude this random process as unfeasible
            if tries == MAX_TRIES_BEFORE_CLIQUE_CONCLUSION:
                break

            # Add to adjacency matrix
            adjacency_matrix[src][dst] = 1
            adjacency_matrix[dst][src] = 1

            # Update progress
            links_filled[src] = links_filled[src] + 1
            links_filled[dst] = links_filled[dst] + 1
            if links_filled[src] == d:
                remaining.remove(src)
            if links_filled[dst] == d:
                remaining.remove(dst)

        # Check that there is no clique left at the end (forcing regularity), else try again
        if len(remaining) != 0:
            continue
        else:
            # Final resulting graph
            return adjacency_matrix

    print("Unable to generate regular random graph: likely impossible dimensions.")
    exit(1)


##
# Create a random Jellyfish with n switches each with a network degree of d.
# Afterwards, write it to the specified output file.
#
# @param    out_filename  Output folder
# @param    n             Number of switches
# @param    d             Switch network degree
# @param    capacity      Capacity
# @param    seed          (Optional) PRNG seed
#
##
def create_random_jellyfish(out_filename, n, d, capacity, seed=None):

    # Perform k-lift to get resulting graph in the form of an adjacency matrix
    adjacency_matrix = random_regular_graph(n, d, seed)

    # Save the adjacency list
    with open(out_filename, 'w') as f:
        for i in range(n):
            for j in range(n):
                if i == j or adjacency_matrix[i, j] == 0:
                    continue
                else:
                    f.write(str(i) + "," + str(j) + "," + str(capacity) + "\n")


##
# Main function called.
##
def main():
    args = sys.argv[1:]
    if len(args) != 4 and len(args) != 5:
        print("Usage: python generator_random_regular_graph.py <out_filename> <n>(int) <d>(int) <capacity>(int)"
              " [optional: <seed>(int)]")
    else:
        if len(args) == 4:
            create_random_jellyfish(args[0], int(args[1]), int(args[2]), int(args[3]))
        else:
            create_random_jellyfish(args[0], int(args[1]), int(args[2]), int(args[3]), int(args[4]))


if __name__ == "__main__":
    main()
