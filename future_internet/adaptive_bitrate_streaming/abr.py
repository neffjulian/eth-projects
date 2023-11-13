from enum import Enum


class CALLBACK_EVENT( Enum ):
    INIT = 0
    DOWNLOAD_COMPLETED = 1
    TIMEOUT  = 2
    REBUFFERING = 3

def abr(
    typ,
    current_time,           # time from the beginning of the simulation in seconds
    playback_time,          # how much of the video has been shown (in seconds)
    playback_chunk,         # the chunk that is playing right now
    current_chunk,          # the chunk number that is downloading right now (or has been just finished)
    current_chunk_quality,  # the quality of the current_chunk
    current_chunk_download, # how much of current_chunk has been downloaded (in bytes)
    video # contains 6 video arrays (one per quality level) - Each subarray contain the size of each chunk in the video
):
    """
        typ: type of event
            INIT - initial call at time 0
            DOWNLOAD_COMPLETED - a chunk has been downloaded
            TIMEOUT - timeout has happend
            REBUFFERING - rebuffering started
        
        Returns: next chunk that should be downloaded
           * quality_to_download_now - quality of the next chunk from 0 to 5
           * chunk_to_download_now   - chunk number of the chunk that should be downloaded
                                     - next_chunk cannot be in the past, if the player plays chunk 10, chunk 9 shouldn't be downloaded
                                     - if you set next_chunk to -1, no chunk will be downloaded
                                     - if the previou download hasn't been completed (e.g. in case of rebuffering) you can change the chunk
                                       that is currently downloading. For instance, you started downloading a high quality chunk, but
                                       rebuffering happened and now you would like to lower the quality. In that case, return the same chunk
                                       number, but different quality.
           * timeout    - set a timer that will trigger the abr function again
                        - timeout is in absolute time, usually set it as current_time+X (where min X is 200ms)
                        - timeout 0 means no timeout
    """
    
    #initial
    if typ == CALLBACK_EVENT.INIT:
        return 0, 0, 0.0

    #rebuffering or timeout, ignore 
    if typ == CALLBACK_EVENT.TIMEOUT or typ == CALLBACK_EVENT.REBUFFERING:
        return current_chunk_quality, current_chunk, 0

    next_chunk = current_chunk + 1
    
    #if we arrived to the end of the stream or it is not the initial call and the download has finished
    
    timeout = 0
    if next_chunk == len(video[0]):
        next_chunk = -1

    if typ == CALLBACK_EVENT.DOWNLOAD_COMPLETED:
        diff = current_chunk - playback_chunk

        if diff > 10 and current_chunk_quality < 4:
            next_chunk_quality = current_chunk_quality + 2
        elif diff > 3 and current_chunk_quality < 5:
            next_chunk_quality = current_chunk_quality + 1
        elif diff < 1 and current_chunk_quality > 3:
            next_chunk_quality = current_chunk_quality - 3
        elif diff < 3 and current_chunk_quality > 0:
            next_chunk_quality = current_chunk_quality - 1
        else:
            next_chunk_quality = current_chunk_quality
        
        """

        if diff > 7 and current_chunk_quality < 4:
            next_chunk_quality = current_chunk_quality + 2

        elif diff > 3 and current_chunk_quality < 5:
            next_chunk_quality = current_chunk_quality + 1
        
        elif diff < 1 and current_chunk_quality > 0:
            next_chunk_quality = current_chunk_quality - 1
        """

        
        
        

    else:
        next_chunk_quality = current_chunk_quality

    return next_chunk_quality, next_chunk, current_time + 4000
