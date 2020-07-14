template file:
	- index.html : homepage for our project
	- about.html : brief introduction for project
	- other css and js files.....

visionCode file:
	- 2 output images
	--> youFace.png : youngerMagic (src img: young_test.jpg)
	--> aaa.png     : pencilMagic  (src img: pencil_test.jpg)
	- pencilMagic.py: src code for pencil drawing
	- youngMaggic.py: src code for making people younger
	- hist_kernel.py: code for histogram matching
	- redering.py   : directionally alpha blending
	- util.py       : performs rotation
	- clrMagic.py	: src code for filling color on pencil sketch
	- pencilText.png: pencil texture

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

References:
	- Lu C, Xu L, Jia J. Combining sketch and tone for pencil drawing production[C]
	http://www.cse.cuhk.edu.hk/~leojia/projects/pencilsketch/npar12_pencil.pdf
	- https://github.com/duduainankai/pencil-python
	- https://www.cnblogs.com/Imageshop/p/4285566.html
	- https://github.com/ageitgey/face_recognition
