###########################################################
# Virtual Environment ( Sanal ortam ) ve (Package Management) Paket Yönetimi
###########################################################

# sanal ortamların listelenmesi :
# conda env list

# sanal ortam oluşturma
# conda create -n myenv

#sana ortamı aktif etme :
# conda activate myenv

#yüklü paketleri listeleme
#conda list 

# paketlerin yüklenmesi 
# conda install numpy pandas scipy 

#paketlerin silinmesi 
#conda remove package_name 

#belirli versiyona göre paket yükleme : 
#conda install numpy =1.20.1  

#upgrade etme yükseltme 
# conda upgrade numpy 

# tüm paketlerin yükseltilmesi 
#conda upgrade --all 

#paket yükleme 
# pip install pandas  

# export etme 
# conda env export > environment.yaml ismini verdi

# environmentı kapama 
#conda deactivate  

#conda sanal ortamı silme 
#conda env remove -n myenv 
