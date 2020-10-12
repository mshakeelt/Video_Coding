In this Exercise we have RGB to YCBCR encoder and decoder.
In encoder we have:
	Accessed the webcam using open-CV
	Applied YCbCr conversion on live video stream
	Displayed each component seprately
	wrote the data on a text file using pickle.dump function
	converted the data type of the Y Cb and Cr components
In decoder we have:
	Loaded the raw data using pickle.load
	applied inverse color transform
	displayed the resultant RGB frame

Usge:
	python encoder.py to encode
	python decoder.py to decode

# make sure that the webcam is accessible and openCV is properly installed.