%% Ideal Lowpass & Gaussian
% In Ideal low pass filter, all the frequencies above a cut-off frequency are completely eliminated and those below the cut-off frequency remains unchanged.
%
% In gaussian low-pass filter, all the frequencies closer to center(i.e. lower frequencies) are given higher weight than those far from the center(i.e. higher frequencies). 
% It doesn't have a specific boundary. It's window is (-Infinity, +Infinity). The weights are defined by the standard deviation of filter.
%% Images
% All the images are displayed here. We have saved the output images in assignment5_DFT/4/images/.
%% Your code here
tic;
Q4
toc;
%% Part (a)
% The cut-off frequency used in Ideal low pass filter is 50.
% Here we have displayed the i) the Ideal low pass filter ii)Fourier Transform of Ideal Low pass filtered Image iii) Ideal Low Pass Filtered Image 
%% Part (b)
% The standard deviation used in Gaussian low pass filter is 40.
% Here we have displayed the i) the Gaussian low pass filter ii)Fourier Transform of Gaussian Low pass filtered Image iii) Gaussian Low Pass Filtered Image
%% Part (c)
% The image filtered by ideal low pass filter has ringing effects due to Gibbs Phenomenon, while the image filtered by gaussian low pass filter is smooth. 
% The frequency response of all filters has been shown.
