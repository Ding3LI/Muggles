In the zip, there are two more zips inside:
	- template.zip
	- visionCode.zip

template.zip:
	- index.html : homepage for our project
	- about.html : brief introduction for project
	- other css and js files.....

visionCode.zip:
	- 2 output images
	--> youFace.png : youngerMagic
	--> aaa.png     : pencilMagic
	- pencilMagic.py: src code for pencil drawing
	- youngMaggic.py: src code for making people younger
	- hist_kernel.py: code for histogram matching
	- redering.py   : directionally alpha blending
	- util.py       : performs rotation
	- clrMagic.py	: src code for filling color on pencil sketch

IMPORTANT:
	* installs:
		- dlib
		- face_recognition
		- numpy
		- scipy
		- cv2
		- PIL

Testing:
	- each file contains detailed comments for how to run script. Every main file has a test main function at the bottom.

Still working:
	- We are firstly working on Django framework, however we are still trying to connect front-end and back-end together. The zip for now contains only source code scripts, but we are looking forward to set up our first Django project on the website.

References:
	- Lu C, Xu L, Jia J. Combining sketch and tone for pencil drawing production[C]
	http://www.cse.cuhk.edu.hk/~leojia/projects/pencilsketch/npar12_pencil.pdf
	- https://github.com/duduainankai/pencil-python
	- https://www.cnblogs.com/Imageshop/p/4285566.html
	- https://github.com/ageitgey/face_recognition