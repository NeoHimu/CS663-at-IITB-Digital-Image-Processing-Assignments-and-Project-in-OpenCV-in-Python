%% Notch Filter
% Notch filter is useful when our image is corrupted by periodic noise. We can remove extra peaks in fourier transform using notch filter.
%
%% Images
% All the images are displayed here. We have saved the output images in assignment5_DFT/3/images/.
%% Your code here
tic;
Q3
toc;
%% Part (a)
% First image displayed here is the Original image with the low frequency noise pattern.
% Second image is the log magnitude of fourier transform of original image.
%% Part (b)
% By observing the log magnitude of fourier transform and using max() function of MATLAB, we found positions of two maximum peaks other than the centre peak.
% After using fft and then fftshift on image, the extra peaks are found to be located at (119,124) and (139,134) near the center peak which is located at (129,129). Since fftshift swaps the
% first and second quadrant with third quadrant and fourth quadrant respectively. We did postprocessing to find location of peaks in original fourier transform (without fftshift) of image.
% The peaks location turned out to be (247,252) and (11,6) in original fourier transform of image.
%% Part (c) 
% Ideal Notch filter is implemented by finding the peaks in the fourier transform and then setting a small window around the peaks to 0.
% The third figure is the Restored Image. The fourth figure is the fourier transform of the restored image.

