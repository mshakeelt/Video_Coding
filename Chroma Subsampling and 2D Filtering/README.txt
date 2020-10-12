In this exercise we have created the video encoder and decoder framework with chroma subsampling and pyramidal filtering.

In encoder we have:
	Taken the live vider stream
	converted the RGB to YCbCr
	applied 4:2:0 chroma subsampling on Cb and Cr components
	saved the raw frames

in decoder we have:
	Loaded the raw frames
	inseted the zeros in Cb and Cr components
	applied a 2D pyramidal lowpass filter
	converted YCbCr to RBG
	displayed the final frame

Usage:
python encoder.py to encode
python decoder.py to decode