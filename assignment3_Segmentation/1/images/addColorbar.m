function op_img = addColorbar (img_name)

img = imread(img_name);
f = figure('visible', 'off'); imshow(img, [0 255]); colormap gray; colorbar;
saveas(f, img_name);
op_img = img;