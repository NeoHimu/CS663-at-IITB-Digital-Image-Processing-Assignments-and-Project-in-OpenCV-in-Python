gnome-terminal -e "python myShrinkImageByFactorD.py barbaraSmall.png 2"
read -p "Press any key to continue... " -n1 -s
gnome-terminal -e "python myShrinkImageByFactorD.py barbaraSmall.png 3" 
read -p "Press any key to continue... " -n1 -s
gnome-terminal -e "python myShrinkImageByFactorD.py circles_concentric.png 2"
read -p "Press any key to continue... " -n1 -s
gnome-terminal -e "python myShrinkImageByFactorD.py circles_concentric.png 3"
read -p "Press any key to continue... " -n1 -s

gnome-terminal -e "python myBilinearInterpolation.py barbaraSmall.png"
read -p "Press any key to continue... " -n1 -s
gnome-terminal -e "python myBilinearInterpolation.py circles_concentric.png"
read -p "Press any key to continue... " -n1 -s

gnome-terminal -e "python myNearestNeighborInterpolation.py barbaraSmall.png"
read -p "Press any key to continue... " -n1 -s
gnome-terminal -e "python myNearestNeighborInterpolation.py circles_concentric.png"
