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

import random
import sys


##
# Create a random permutation pairing list.
#
# @param    n             Number of nodes
# @param    seed          (Optional) PRNG seed
#
##
def random_permutation_pairing(n, seed):

    if n % 2 != 0:
        print("Cannot create random permutation pairing with odd n.")
        exit(1)

    # Initialize randomness for reproducibility
    random.seed(seed)

    # As long as possible to create a pair
    pair_list = []
    remaining = list(range(0, n))
    while len(remaining) >= 2:
        first = random.randint(0, len(remaining) - 1)
        first_val = remaining[first]
        second = random.randint(0, len(remaining) - 1)
        second_val = remaining[second]
        while first == second:
            second = random.randint(0, len(remaining) - 1)
            second_val = remaining[second]

        pair_list.append((first_val, second_val))
        pair_list.append((second_val, first_val))

        remaining.remove(first_val)
        remaining.remove(second_val)

    # Return pairing list
    return pair_list


##
# Create a random permutation pairing list and write to file.
#
# @param    out_filename  output filename
# @param    n             Number of nodes
# @param    seed          (Optional) PRNG seed
#
##
def create_random_permutation_pairing(out_filename, n, seed=None):

    # Generate pairing
    pair_list = random_permutation_pairing(n, seed)

    # Save the adjacency list
    with open(out_filename, 'w') as f:
        for p in pair_list:
            f.write(str(p[0]) + "," + str(p[1]) + "\n")


##
# Main function called.
##
def main():
    args = sys.argv[1:]
    if len(args) != 2 and len(args) != 3:
        print("Usage: python generator_random_permutation_pairing.py <out_filename> <n>(int) [optional: <seed>(int)]")
    else:
        if len(args) == 2:
            create_random_permutation_pairing(args[0], int(args[1]))
        else:
            create_random_permutation_pairing(args[0], int(args[1]), int(args[2]))


if __name__ == "__main__":
    main()
