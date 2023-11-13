from abr import abr, CALLBACK_EVENT
import logging



SEGMENT_LENGTH = 4.
inf = 100000000.

BITRATES_REWARD = [300, 750, 1200, 1850, 2850, 4300] #Kbps




def call_abr(typ, state, logger):
    nxt_quality, nxt_chunk, timeout = abr(
        typ,
        state['time'],
        state['playback_time'],
        state['playback_chunk'],
        state['next_chunk'],
        state['next_chunk_quality'],
        state['next_chunk_downloaded'],
        state['video']
    )

    logger.debug("{}s: ABR selected quality {} for chunk {}".format(state['time'], nxt_quality, nxt_chunk))
    assert nxt_chunk == -1 or (nxt_chunk > state['playback_chunk'] and nxt_chunk < len(state['video'][0]))
    assert 0 <= nxt_quality <= 5
    
    assert timeout >= 0

    if timeout == 0:
        logger.debug("{}s: No timeout has been set".format(state['time']))
        timeout = inf
    
    elif timeout != state['timeout']:

        assert timeout - state['time']>= 0.199 # Condition to avoid too frequent timeouts
        logger.debug("{}s: Timeout is happening at {}".format(state['time'], timeout))

    assert timeout>state['time']

    return nxt_quality, nxt_chunk, timeout

def set_timeout(state, timeout):
    if timeout == 0:
        timeout = inf
    elif timeout != state['timeout']:
        assert timeout - state['time']>= 0.199
    assert timeout>state['time']
    state['timeout'] = timeout

def set_download(state, nxt_quality, nxt_chunk, timeout):
    state['timeout'] = timeout
    if nxt_chunk == -1:
        state['next_chunk'] = -1
        state['next_chunk_quality'] = 0
        state['next_chunk_downloaded'] = 0
        state['next_chunk_expected'] = inf
    else:
        state['next_chunk'] = nxt_chunk
        state['next_chunk_quality'] = nxt_quality
        state['next_chunk_downloaded'] = 0
        state['next_chunk_expected'] = state['video'][nxt_quality][nxt_chunk]/state['bandwidth']+state['time']

def experiment(exp_name, time_trace, bandwidth_trace, video, verbose_output=False):
    state = {
        'bandwidth': bandwidth_trace[0],
        'time': 0.,
        'video': video,

        'total_video_duration': len(video[0])*SEGMENT_LENGTH,
        'data': [-1]*len(video[0]),

        'playback_time': 0.,
        'playback_chunk': -1,
        'playback_change_time': inf,
        'rebuffering': True,

        'next_chunk': -1,
        'next_chunk_quality': -1,
        'next_chunk_downloaded': -1.,
        'next_chunk_expected': -1.,

        'timeout': inf
    }

    if verbose_output:
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level = logging.DEBUG)
        logger = logging.getLogger('Experiment Environment')
    else:
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level = logging.INFO)
        logger = logging.getLogger('Experiment Environment')
 

    #drop first change
    del time_trace[0]
    del bandwidth_trace[0]

    logger.info("Experiment {} started".format(exp_name)) 
    logger.debug("{}s: Calling ABR with event INIT".format(state['time']))
    
    #init abr, start downloading the first chunk

    nxt_quality, nxt_chunk, timeout = call_abr(CALLBACK_EVENT.INIT, state, logger)
    set_download(state, nxt_quality, nxt_chunk, timeout)

    while True:
        trace_min = inf if len(time_trace) == 0 else time_trace[0]

        min_time = min(
            state['playback_change_time'],
            state['next_chunk_expected'],
            state['timeout'],
            trace_min
        )
        

        if min_time != state['time']:
            #update progress
            time_diff = min_time - state['time']
            assert time_diff >= 0
            state['time'] = min_time
            
            logger.debug("{}s: {} seconds have elapsed from last loop".format(state['time'], time_diff))

            if state['playback_chunk'] != -1 and not state['rebuffering']:
                state['playback_time'] += time_diff
                logger.debug("{}s: Playback progress: {}s".format(state['time'], state['playback_time']))
            else:
                logger.debug("{}s: Playback is stalled, rebuffering is happening".format(state['time']))

            
            
            if state['next_chunk'] != -1:
                state['next_chunk_downloaded'] += time_diff*state['bandwidth']
                state['next_chunk_downloaded'] = int(round(state['next_chunk_downloaded']))
                logger.debug("{}s: Bandwidth is {}, bytes downloaded of {} chunk: {}/{}, ".format(  state['time'],\
                                                                                                    state['bandwidth'],\
                                                                                                    state['next_chunk'],\
                                                                                                    state['next_chunk_downloaded'],\
                                                                                                    video[state['next_chunk_quality']][state['next_chunk']],\
                                                                                                    time_diff*state['bandwidth']))

        if min_time == state['playback_change_time']:

            logger.debug("{}s: Player Event".format(state['time']))
            #rebuffering or reading the next chunk
            if state['playback_chunk'] + 1 == len(video[0]):
                logger.debug("{}s: Experiment {}, end of the video reached".format(state['time'], exp_name))
                #end of the video
                break

            if state['data'][state['playback_chunk']+1] != -1:
                state['playback_chunk'] += 1
                state['playback_change_time'] = state['time'] + SEGMENT_LENGTH
                logger.debug("{}s: Chunk no {} playing".format(state['time'], state['playback_chunk']))
                state['rebuffering'] = False
            else:
                logger.debug("{}s: player is rebuffering".format(state['time']))
                state['rebuffering'] = True
                state['playback_change_time'] = inf

                logger.debug("{}s: Calling ABR with event REBUFFERING".format(state['time']))
                nxt_quality, nxt_chunk, timeout = call_abr(CALLBACK_EVENT.REBUFFERING, state, logger)
                # if the user wants to change the decision
                if nxt_quality != state['next_chunk_quality'] or nxt_chunk != state['next_chunk']:
                    # download the next chunk                
                    logger.debug("{}s: ABR modified decision".format(state['time']))
                    logger.debug("{}s: Setting the new download for chunk {} at quality level {}".format(state['time'], nxt_chunk, nxt_quality))
                    set_download(state, nxt_quality, nxt_chunk, timeout)
                else:
                    logger.debug("{}s: ABR didn't modify decision".format(state['time'])) 
                    set_timeout(state, timeout)

        elif min_time == state['next_chunk_expected']:
            #chunk downloaded
            #if the new chunk is not already playing

            logger.debug("{}s: Chunk {} has been downloaded".format(state['time'], state['next_chunk']))

            if state['next_chunk'] != state['playback_chunk']:
                state['data'][state['next_chunk']] = state['next_chunk_quality']
                #play video if it was blocked

                if state['rebuffering'] and state['playback_chunk']+1 == state['next_chunk']:
                    logger.debug("{}s: Player was paused. Resuming playback".format(state['time']))
                    state['playback_chunk'] += 1
                    state['playback_change_time'] = state['time'] + SEGMENT_LENGTH
                    state['rebuffering'] = False


            logger.debug("{}s: Calling ABR with event DOWNLOAD COMPLETED".format(state['time']))
            nxt_quality, nxt_chunk, timeout = call_abr(CALLBACK_EVENT.DOWNLOAD_COMPLETED, state, logger)
            set_download(state, nxt_quality, nxt_chunk, timeout)

        elif min_time == trace_min:
            #bandwidth change
            bandwidth = bandwidth_trace[0]
            del time_trace[0]
            del bandwidth_trace[0]

            state['bandwidth'] = bandwidth
            logger.debug("{}s: New bandwidth is {}".format(state['time'], bandwidth))

            if state['next_chunk'] != -1:
                state['next_chunk_expected'] = (video[state['next_chunk_quality']][state['next_chunk']] - state['next_chunk_downloaded'])/state['bandwidth']+state['time']
                logger.debug("{}s: With constant bandwidth, chunk {} download will terminate at {}".format(state['time'], state['next_chunk'], state['next_chunk_expected']))


        elif min_time == state['timeout']:
            #call abr with the timeout option
            logger.debug("{}s: timeout is happening".format(state['time']))
            logger.debug("{}s: Calling ABR with event TIMEOUT".format(state['time']))
            nxt_quality, nxt_chunk, timeout = call_abr(CALLBACK_EVENT.TIMEOUT, state, logger)
            # if the abr wants to change the decision
            if nxt_quality != state['next_chunk_quality'] or nxt_chunk != state['next_chunk']:
                logger.debug("{}s: ABR modified decision".format(state['time']))
                logger.debug("{}s: Setting the new download for chunk {} at quality level {}".format(state['time'], nxt_chunk, nxt_quality))
                set_download(state, nxt_quality, nxt_chunk, timeout)
            else:
                logger.debug("{}s: ABR didn't modify decision".format(state['time'])) 
                set_timeout(state, timeout)
    
    all_switches =  [ abs(BITRATES_REWARD[state['data'][i]]-BITRATES_REWARD[state['data'][i+1]]) for i in range(len(state['data']) - 1)]
    switches_amplitude = sum(all_switches)
    rebuffer_time = state['time'] - len(video[0])*SEGMENT_LENGTH
    quality = sum(map(lambda x: BITRATES_REWARD[x], state['data']))

    REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
    SMOOTH_PENALTY = 1
    M_IN_K = 1000.0
    score = quality / M_IN_K - REBUF_PENALTY * rebuffer_time - SMOOTH_PENALTY * switches_amplitude / M_IN_K
    logger.info("Experiment {} score: {}".format(exp_name, score))
    
    return score, rebuffer_time, switches_amplitude
