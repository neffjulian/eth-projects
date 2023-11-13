import os
import sys
import pandas as pd
from experiment import experiment
import argparse


BANDWIDTH_TRACE = 'bandwidth_trace/'
Mbs_to_Bps = 1000000./8

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_trace', help='Directory of the video to analyze, default "video_trace"', default="video_trace")
    parser.add_argument('--verbose', action='store_true', help='If specified, the network environment will have verbose output')
    args = parser.parse_args()

    experiments = {}
    video = []

    for fn in os.listdir(BANDWIDTH_TRACE):
        df = pd.read_csv(BANDWIDTH_TRACE+fn, delimiter=' ', header=None)
        experiments[fn] = {
            'time': df[0].tolist(),
            'bandwidth': (df[1]*Mbs_to_Bps).tolist(),
        }


    if not os.path.exists(args.video_trace):
        print("Video trace path doesn't exist")
        sys.exit(-1)

    for fn in sorted(os.listdir(args.video_trace)):
        df = pd.read_csv(os.path.join(args.video_trace, fn), header=None)
        video.append(df[df.columns[0]].tolist())

    total_reward = 0.
    write_data = []
    for exp in experiments:
        reward, rebuffer_time, switches_amplitude = experiment(
            exp,
            experiments[exp]['time'],
            experiments[exp]['bandwidth'],
            video,
            args.verbose
        )
        total_reward += reward
        write_data.append((exp, reward, rebuffer_time, switches_amplitude))

    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    if args.video_trace[-1] == '/':
        log_filename = os.path.join('logs', '{}.csv'.format(os.path.dirname(args.video_trace)))
    else:
        log_filename = os.path.join('logs', '{}.csv'.format(os.path.basename(args.video_trace)))
    
    with open(log_filename,'w') as f:
        f.write('log_name,reward,rebuffer_time,switches_amplitude\n')
        for x in write_data:
            f.write('%s,%f,%f,%f\n'%(x))

    print('Average reward:', total_reward/len(experiments))

if __name__ == '__main__':
    main()
