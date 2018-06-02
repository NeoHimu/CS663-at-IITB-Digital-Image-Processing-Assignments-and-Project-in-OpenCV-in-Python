function myHarrisCornerDetector
im=load('../data/boat.mat');

I=im.imageOrig;

%shifting and rescaling the intensities to [0 1]
maxI=max(I(:));
minI=min(I(:));
I=(I-minI)/(maxI-minI);

My=[-1 -1 -1;0 0 0;1 1 1];
Mx=My';

Ix=conv2(I,Mx,'same');
Iy=conv2(I,My,'same');

gMaskx=fspecial('gaussian',5,10); %1.67
gMasky=fspecial('gaussian',5,3);

Ixy=Ix.*Iy;

figure,  imshow(I,[]),title('Original Image'),colormap gray, colorbar;
pause(2);
figure,  imshow(Ix),title('X-Derivative'),colormap gray, colorbar;
pause(2);
figure, imshow(Iy),title('Y-Derivative'), colormap gray, colorbar;
pause(2);
%figure,  imshow(Ixy),title('XY-Derivative'),colormap gray, colorbar;
%pause(2);
Ix2=Ix.^2;
Iy2=Iy.^2;

Ix2=conv2(Ix2,gMaskx,'same');
Iy2=conv2(Iy2,gMasky,'same');
Ixy=conv2(Ixy,gMaskx,'same');



k=0.20; %kappa
win=7;

C=(Ix2.*Iy2 - Ixy.^2) - k*(Ix2+Iy2).^2;

MinEigenValue=zeros(size(I));
MaxEigenValue=zeros(size(I));

for i = 1:size(I,1)
	for j = 1:size(I,2)
		
		temp=[Ix2(i,j) Ixy(i,j);Ixy(i,j) Iy2(i,j)];
		Eigen=eig(temp);
		MinEigenValue(i,j)=min(Eigen(:));
		MaxEigenValue(i,j)=max(Eigen(:));
	end
end


thresh=0.1*max(C(:));

m=ordfilt2(C,win^2,ones(win,win));
C=(C==m)&(C>thresh);
[r,c]=find(C);

pause(2);
figure, imshow(MinEigenValue,[0 1]), title('MinEigenValue'),colormap gray, colorbar;
pause(2);
figure, imshow(MaxEigenValue,[0 1]), title('MaxEigenValue') ,colormap gray, colorbar;
pause(2);
figure, imshow(C,[]), title('Corners'),colormap gray, colorbar;
pause(2);
figure, imshow(I,[]), title('Corners Marked'),colormap gray, colorbar;
hold on;
plot(c,r,'g+');



end
