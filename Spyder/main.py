import torch
import Unet 
import PatchGAN
import lib_Pix2Pix as p2p
import train



########## 1. Obtention des path de la BDD d'image SAR et Optique #############

path_zone1_local = "C:/Users/natsl/Documents/These/BDD/Zone1"

list_file_local = p2p.get_file_in_folder(path_zone1_local)

list_SAR_file_France = p2p.select_file_name(list_file_local, 'S1moy')
list_optique_file_France = p2p.select_file_name(list_file_local, 'S2')

################ 2. Création de la BDD ####################

#Define the data splits.
nb_train_data = 0.6
nb_val_data = 0.2
nb_test_data = 0.2

percent = 0.1

img_folder = {}
img_folder["train"] = []
img_folder["val"] = []
img_folder["test"] = []

SAR_folder = {}
SAR_folder["train"] = []
SAR_folder["val"] = []
SAR_folder["test"] = []

nb_imgs_zone = min(len(list_SAR_file_France), len(list_optique_file_France))
sorted_SAR_list = sorted(list_SAR_file_France)
sorted_optique_list = sorted(list_optique_file_France)
for k in range(int(nb_imgs_zone*nb_train_data*percent)):
    SAR_folder["train"].append(sorted_SAR_list[k])  
    img_folder["train"].append(sorted_optique_list[k]) 
for k in range(int(nb_imgs_zone*nb_train_data*percent),int(nb_imgs_zone*(nb_train_data+nb_val_data)*percent)):    
    SAR_folder["val"].append(sorted_SAR_list[k])  
    img_folder["val"].append(sorted_optique_list[k]) 
for k in range(int(nb_imgs_zone*(nb_train_data+nb_val_data)*percent),int(nb_imgs_zone*percent)):
    SAR_folder["test"].append(sorted_SAR_list[k])  
    img_folder["test"].append(sorted_optique_list[k]) 

    
### 3. Chargement de la BDD

batch_size = 16 #The batch size should generally be as big as your machine can take it.

training_dataset = p2p.ImgOptiqueSAR(img_folder["train"], SAR_folder["train"])
validate_dataset = p2p.ImgOptiqueSAR(img_folder["val"], SAR_folder["val"])

train_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True)
validate_loader = torch.utils.data.DataLoader(dataset=validate_dataset, batch_size=batch_size, shuffle=False)

x = torch.zeros(5, 3, 256, 256, dtype=torch.float, requires_grad=False)
y = torch.zeros(1, 2, 256, 256, dtype=torch.float, requires_grad=False)


# Initialize generator and discriminator
Generator = Unet.Unet()
# print(Generator)
# output_gen = Generator(x)
# print(output_gen)

Discriminator = PatchGAN.PatchGAN()
# print(Discriminator)
# output_disc = Discriminator(y)
# print(output_disc)

number_epochs = 2

train.train_GAN(number_epochs, Generator, Discriminator, train_loader, 
              validate_loader, batch_size)