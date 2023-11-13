Describe your algorithm here
======
We used a buffer-based approach and the Idea behind it is quite intuitive:
- If the quality is low and the buffer is big   --> increase the quality by more than one
- If the quality is high and the buffer is big --> increase the quality by one
- If the quality is low and the buffer is small --> decrease the quality by more than one
- If the quality is high and the buffer is small --> decrease the quality by one
- If the quality is good enough and the buffer big enough -> leave it as it is.


With this approach we already reached the second threshold so we didn't improve the algorithm further with trying to calculate the speed of the internet or try to minimize the number of quality changes.

We got the specific numbers by trial and error.
