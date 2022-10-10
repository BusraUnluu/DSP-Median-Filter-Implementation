# DSP-Median-Filter-Implementation

Digital Signal Processing, Median Filtering and Center weighted Median Filter Implementation

In this project, two different (median filter and weighted Median Filter) filtering algorithms are implemented. These algortihms are tested on noisyImage.jpg and the results compared with Opencv(cv.boxFilter,cv.GaussianBlur,cv.medianBlur) fuctions. Their PSNR values are compared. Finally, the highest PSNR value is obtained from  weighted median filter.

PSNR is not always gold standart for image/noise quality, for example in sharpening and contrast changing low PSNR values are obtained and image quality is better than weighted median filter. (Expectation: High PSNR values=good quality images)
