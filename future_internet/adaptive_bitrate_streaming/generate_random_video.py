import argparse
import os
import sys
from glob import glob
import numpy as np
from random import seed, randint

MIN_CHUNKS = 30
MAX_CHUNKS = 100

MIN_CHUNK_BYTES = 700

VIDEO_ORIGINAL = './video_trace'
video_size_file_pattern = 'video_sizes_{}'

A_DIM = 6


def PROPERTIES(VIDEO_ORIGINAL):
    
    BASE_QUALITY_DISTRIBUTION = {}
    DOWNSCALING_NOISES = {}

    BASE_QUALITY_FILE = os.path.join(VIDEO_ORIGINAL, video_size_file_pattern.format(A_DIM-1))

    with open(BASE_QUALITY_FILE, 'r') as fin:
        base_quality_sizes = [int(segment) for segment in fin.readlines()]

    BASE_QUALITY_DISTRIBUTION["mean"] =  np.mean(base_quality_sizes)
    BASE_QUALITY_DISTRIBUTION["std"] = np.std(base_quality_sizes)

    QUALITY_FILES = glob(os.path.join(VIDEO_ORIGINAL, video_size_file_pattern.format('*')))
    QUALITY_FILES.sort(key=lambda x: int(x.split('_')[-1]))

    for i, base_quality_file in reversed(list(enumerate(QUALITY_FILES))):
        if i != 0:
            downgraded_file = os.path.join(VIDEO_ORIGINAL, video_size_file_pattern.format(i-1))
            
            with open(base_quality_file, 'r') as base_file, open(downgraded_file, 'r') as down_file:
                base_size = [ int(x) for x in base_file.readlines()]
                down_size = [int(x) for x in down_file.readlines()]
                
                assert len(base_size) == len(down_size)

                difference_list = [ base_size[j] - down_size[j] for j in range(len(base_size)) ]
            
            DOWNSCALING_NOISES[i-1] = {}
            DOWNSCALING_NOISES[i-1]["mean"] = np.mean(difference_list)
            DOWNSCALING_NOISES[i-1]["std"] = np.std(difference_list)

    return BASE_QUALITY_DISTRIBUTION, DOWNSCALING_NOISES
    


if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--dest', type=str, required=True)
    args = parser.parse_args()
    
    if not os.path.exists(VIDEO_ORIGINAL):
        print('Original video does not exist')
        sys.exit(-1)

    if not os.path.exists(args.dest):
        print('Creating dest directory')
        os.makedirs(args.dest)

    BASE, DOWNSCALING = PROPERTIES(VIDEO_ORIGINAL)

    seed(args.seed)
    np.random.seed(args.seed)

    chunks_no = randint(MIN_CHUNKS, MAX_CHUNKS)
    segments_size = [[ 0 for seg in range(chunks_no)] for quality in range(A_DIM)]
    
    segments_size[-1] = [int(abs(size)) for size in np.random.normal(BASE['mean'], BASE['std'], chunks_no)]
    
    for i in reversed(range(A_DIM - 1)):
        downgrading_difference = [int(abs(size)) for size in np.random.normal(DOWNSCALING[i]['mean'], DOWNSCALING[i]['std'], chunks_no)]
        
        for j in range(chunks_no):
            while segments_size[i][j] <= MIN_CHUNK_BYTES:
                segments_size[i][j] = segments_size[i+1][j] - int(downgrading_difference[j])
                downgrading_difference[j] /= 2 
    
    for segment_no in range(chunks_no):
        for qual in range(A_DIM-1):
            assert segments_size[qual][segment_no] < segments_size[qual+1][segment_no]

    for i in range(A_DIM):
        file_out = os.path.join(args.dest, video_size_file_pattern.format(i))
        with open(file_out, 'w') as fout:
            for c in range(chunks_no):
                fout.write(str(segments_size[i][c]))
                if c != chunks_no - 1:
                    fout.write('\n')
