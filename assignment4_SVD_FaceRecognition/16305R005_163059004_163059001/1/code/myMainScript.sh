#Eigen Value Decomposition
gnome-terminal -e "python 1.py"
read -p "Press any key to continue... " -n1 -s

#SVD
gnome-terminal -e "python 1_first_part_svd_small_modification.py" 
read -p "Press any key to continue... " -n1 -s

#Yale Database
#a.)
gnome-terminal -e "python 1_SVD_CroppedYale.py 0"
read -p "Press any key to continue... " -n1 -s

#b.)
#For removing top 3 eigenvectors
gnome-terminal -e "python 1_SVD_CroppedYale.py 3"
read -p "Press any key to continue... " -n1 -s
