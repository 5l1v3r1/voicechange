# voicechange

The goal of this project was to find some way to translate between two people's voices. Specifically, I wanted a program which, when given two recordings (one in person A's voice, one in person B's), could learn to transform anything person A says into person B's voice.

# Current status

Right now, the thing outputs a bunch of horrible noise instead of changing your voice. It can be used as an autoencoder between your voice and itself, and that actually works surprisingly well. Still, autoencoding was not the goal of the project.

My suspicion is that least squares (per component) is not a good way to measure similarity between two sounds, so optimizing for it is not the answer. I also tried playing around with FFTs with little luck, so I am not sure where that leaves me.
