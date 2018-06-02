function Q4

%load image
im = imread('../data/barbara256.png');

%size for padding
mSize=max(size(im));
P=2^nextpow2(2*mSize);

%perform fast fourier transform on image
imF=fftshift(fft2(im,P,P));
figure; imshow(im,[]), title('Original Image'); pause(1);

%displaying the log magnitude of fourier transform of image.
figure; imshow(log(abs(imF)+1),[]), title('FFT of Original Image'); pause(1);

T=P/2;

M=repelem([-T:T-1], P, 1);
N=repelem([-T:T-1]', 1, P);


%calculating distances
D=sqrt(M.^2 + N.^2);

%Ideal Low pass filter
D0=50;


H=double(D<=D0);
G=imF.*H;

figure; imshow(log(abs(H)+1),[]), title('Ideal Low Pass Filter'); pause(1);
figure; imshow(log(abs(G)+1),[]), title('FFT of IdealLowPass FilteredImage'); pause(1);

g=real(ifft2(fftshift(G)));
g=g(1:size(im,1), 1:size(im,2));
figure; imshow(g,[]), title('IdealLowPass FilteredImage');
pause(1);

%Gaussian low pass filter
D0=40;
H = exp(-(D.^2)./(2*(D0^2)));
G=imF.*H;

figure; imshow(log(abs(H)+1),[]), title('Gaussian Filter'); pause(1);
figure; imshow(log(abs(G)+1),[]), title('FFT of Gaussian FilteredImage'); pause(1);

g=real(ifft2(fftshift(G)));
g=g(1:size(im,1), 1:size(im,2));
figure; imshow(g,[]), title('Gaussian FilteredImage');

end
