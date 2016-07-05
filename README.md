# voicechange

The goal of this project is to find a linear transformation that maps one voice to another voice. I am not even sure if such a thing is possible, but I will sure try.

# Current status

Right now, the thing outputs a bunch of horrible noise instead of changing your voice. If you train it with too little data, it is capable of overfitting, seeming to change your voice from the source clip to the destination clip perfectly. Really, the matrix has just "memorized" what to output for each input.

My suspicion is that least squares (per component) is not a good way to measure similarity between between two sounds, so optimizing for it is not the answer. This might be due to the fact that squared error takes phase into account, when all that really matters is the component frequencies of a signal. Perhaps I should be using Fourier transforms, discarding phase information, and training a matrix on that.
