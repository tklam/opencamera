# Open Camera
This is a fork of Mark Harman's superb [Open Camera](http://opencamera.sourceforge.net/) (being hosted in SourceForge).

## What's new in this repository?
A moving-objects-removal mode was added. I was inspired by the paper [Adaptive Background Mixture Models for Real-time Tracking](http://www.ai.mit.edu/projects/vsam/Publications/stauffer_cvpr98_track.pdf). The idea is to take multiple images and then filter out the moving objects by applying the expectation-maximization method.

In the current implementation, 9 pictures are taken within 2.7s. The result is not very impressive because there are still some "after-images" and the calculations is slow. RenderScript is needed for the code. But I still have not figured out the proper way of using it. I enjoyed developing this feature very much in yet another boring weekend, though.
