
import torch
import lib_Pix2Pix as p2p

########## 1. Obtention des path de la BDD d'image SAR et Optique #############

# path_bdd = "/work/OT/ai4geo/DATA/DATA_MULTIMODAL/"
# path_bdd_France = "/work/OT/ai4geo/DATA/DATA_MULTIMODAL/France/"
# path_bdd_USA = "/work/OT/ai4geo/DATA/DATA_MULTIMODAL/USA/"
path_zone1_local = "C:/Users/natsl/Documents/These/BDD/Zone1"

# list_file = get_file_in_folder(path_bdd)
# list_file_France = get_file_in_folder(path_bdd_France)
# list_file_USA = get_file_in_folder(path_bdd_USA)
list_file_local = p2p.get_file_in_folder(path_zone1_local)

# list_SAR_file = select_file_name(list_file, 'S1moy')
# list_optique_file1 = select_file_name(list_file, 'S2')
# list_optique_file2 = select_file_name(list_file, 'S2mosa')
# list_optique_file = list_optique_file1 + list_optique_file2
# list_S1_file = select_file_name(list_file, 'S1_')
# list_SAR_file_France = select_file_name(list_file_France, 'S1moy')
# list_SAR_file_USA = select_file_name(list_file_USA, 'S1moy')
# list_optique_file_France = select_file_name(list_file_France, 'S2')
# list_optique_file_USA = select_file_name(list_file_USA, 'S2mosa')
list_SAR_file_France = p2p.select_file_name(list_file_local, 'S1moy')
list_optique_file_France = p2p.select_file_name(list_file_local, 'S2')

# list_SAR_file_France[12].remove([])
# list_optique_file_France[12].remove([])
# list_SAR_file_France[21].remove([])
# list_optique_file_France[21].remove([]) 

################ 2. Création de la BDD ####################


#Define the data splits.
nb_train_data = 0.6
nb_val_data = 0.2
nb_test_data = 0.2

percent = 0.1

img_folder = [[] for i in range(3)]
SAR_folder = [[] for i in range(3)]
# for i in range(len(list_SAR_file_France)):
#     nb_imgs_zone = min(len(list_SAR_file_France[i]), len(list_optique_file_France[i]))
#     sorted_SAR_list = sorted(list_SAR_file_France[i])
#     sorted_optique_list = sorted(list_optique_file_France[i])
#     for k in range(int(nb_imgs_zone*0.6*percent)):
#         img_folder[0].append(sorted_SAR_list[k])  
#         SAR_folder[0].append(sorted_optique_list[k]) 
#     for k in range(int(nb_imgs_zone*0.6*percent),int(nb_imgs_zone*0.8*percent)):    
#         img_folder[1].append(sorted_SAR_list[k])  
#         SAR_folder[1].append(sorted_optique_list[k]) 
#     for k in range(int(nb_imgs_zone*0.8*percent),int(nb_imgs_zone*percent)):
#         img_folder[2].append(sorted_SAR_list[k])  
#         SAR_folder[2].append(sorted_optique_list[k]) 

nb_imgs_zone = min(len(list_SAR_file_France), len(list_optique_file_France))
sorted_SAR_list = sorted(list_SAR_file_France)
sorted_optique_list = sorted(list_optique_file_France)
for k in range(int(nb_imgs_zone*0.6*percent)):
    img_folder[0].append(sorted_SAR_list[k])  
    SAR_folder[0].append(sorted_optique_list[k]) 
for k in range(int(nb_imgs_zone*0.6*percent),int(nb_imgs_zone*0.8*percent)):    
    img_folder[1].append(sorted_SAR_list[k])  
    SAR_folder[1].append(sorted_optique_list[k]) 
for k in range(int(nb_imgs_zone*0.8*percent),int(nb_imgs_zone*percent)):
    img_folder[2].append(sorted_SAR_list[k])  
    SAR_folder[2].append(sorted_optique_list[k]) 

    
### 3. Chargement de la BDD

batch_size = 2 #The batch size should generally be as big as your machine can take it.

training_dataset = p2p.ImgOptiqueSAR(img_folder[0], SAR_folder[0])
validate_dataset = p2p.ImgOptiqueSAR(img_folder[1], SAR_folder[1])

train_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True)
validate_loader = torch.utils.data.DataLoader(dataset=validate_dataset, batch_size=batch_size, shuffle=False)

