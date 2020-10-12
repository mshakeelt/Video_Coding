In this framework we have applied the discrete cosine transform on live video stream from webcame.

in encoder we have:
	Taken the RGB frames from webcame and converted them to YCbCr
	Applied 420 crome subsampling on Cb and Cr components
	Applied DCT-2 and sat 3/4 of highest frequencies to zero for compression
	saved the raw data

in decoder we have:
	loaded the data
	applied inverse DCT
	upsampled the chroma components
	applied lowpass filter
	converted YCbCr to RGB
	displayed the final frame

Usage:
python encoder.py to encode the frame
python decoder.py to decode the frame	