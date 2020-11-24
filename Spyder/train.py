# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 9:08:48 2020

@author: natsl
"""

import torch
from torch.autograd import Variable
import time
import datetime
import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt

def train_discriminator(optimizer_D, SARs, fake_SAR, loss_function, Discriminator, batch_size):
    N = 16
    conversion = T.ToTensor()
    valid = Variable(conversion(np.ones((N, batch_size))), requires_grad=False)
    fake = Variable(conversion(np.zeros((N, batch_size))), requires_grad=False)
    
    optimizer_D.zero_grad()
    
    # Train on Real Data
    Discriminator_tensor_real= torch.cat((SARs.float(),  SARs.float()),1)
    prediction_real = Discriminator(Discriminator_tensor_real)
    # Calculate error and backpropagate
    valid = torch.reshape(valid, (prediction_real.shape))
    error_real = loss_function(prediction_real, valid)
    error_real.backward(retain_graph=True)

    # Train on Fake Data
    Discriminator_tensor_fake= torch.cat((fake_SAR.float(),  SARs.float()),1)
    prediction_fake = Discriminator(Discriminator_tensor_fake)
    # Calculate error and backpropagate
    fake = torch.reshape(fake, (prediction_fake.shape))
    error_fake = loss_function(prediction_fake, fake)
    error_discriminator = (error_real + error_fake)*0.5 
    error_fake.backward(retain_graph=True)
    
    # Update weights
    optimizer_D.step()
    
    # Return error and predictions for real and fake inputs
    return error_discriminator 

def validate_discriminator(optimizer_D, SARs, fake_SAR, loss_function, Discriminator, batch_size):
    N = 16
    conversion = T.ToTensor()
    valid = Variable(conversion(np.ones((N, batch_size))), requires_grad=False)
    fake = Variable(conversion(np.zeros((N, batch_size))), requires_grad=False)
    
    # Validate on Real Data
    Discriminator_tensor_real= torch.cat((SARs.float(),  SARs.float()),1)
    prediction_real = Discriminator(Discriminator_tensor_real)
    # Calculate error 
    valid = torch.reshape(valid, (prediction_real.shape))
    error_real = loss_function(prediction_real, valid)

    # Validate on Fake Data
    Discriminator_tensor_fake= torch.cat((fake_SAR.float(),  SARs.float()),1)
    prediction_fake = Discriminator(Discriminator_tensor_fake)
    # Calculate error
    fake = torch.reshape(fake, (prediction_fake.shape))
    error_fake = loss_function(prediction_fake, fake)
    
    return (error_real + error_fake)*0.5 

def train_generator(optimizer_G, SARs, fake_SAR, loss_function, Discriminator, batch_size):
    N = 16
    conversion = T.ToTensor()
    
    optimizer_G.zero_grad()
    Discriminator_tensor= torch.cat((fake_SAR.float(),  SARs.float()),1)
    prediction = Discriminator(Discriminator_tensor)
    valid = Variable(conversion(np.ones((N,batch_size))), requires_grad=False)
    valid = torch.reshape(valid, (prediction.shape))
    
    # Calculate error and backpropagate
    error = loss_function(prediction, valid)
    error.backward(retain_graph=True)
    # Update weights with gradients
    optimizer_G.step()
    # Return error
    return error

def validate_generator(optimizer_G, SARs, fake_SAR, loss_function, Discriminator, batch_size):
    N = 16
    conversion = T.ToTensor()
    
    Discriminator_tensor= torch.cat((fake_SAR.float(),  SARs.float()),1)
    prediction = Discriminator(Discriminator_tensor)
    valid = Variable(conversion(np.ones((N,batch_size))), requires_grad=False)
    valid = torch.reshape(valid, (prediction.shape))
    # Calculate error 
    error = loss_function(prediction, valid)
    return error

def display_result_per_epoch():

        

    return

def train_GAN(number_epochs, Generator, Discriminator, train_loader, 
              validate_loader, batch_size, optimizer_G, optimizer_D,
              loss_function):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    
    Generator.to(device)
    Discriminator.to(device)
    
    marqueur = 0
    nb_display_result = 1
    start_time = time.time()
    training_losses_G = []
    training_losses_D = []
    validation_losses_G = []
    validation_losses_D = []
    
    training_loss_G = 0
    training_loss_D = 0
    
    validation_loss_G = 0
    validation_loss_D = 0
    
    for epoch in range(number_epochs):
        print("Starting epoch number " + str(epoch+1))
      
        Generator.train() 
        Discriminator.train()
      
        for i, (imgs, SARs) in enumerate(train_loader):
            # #Convert the imgs and SARs to torch Variable (will hold the computation
            # #graph).
            imgs = Variable(imgs).to(device)
            SARs = Variable(SARs).to(device)
            fake_SAR = Generator(imgs.float())  
            
            ##### Train Generator #########
            loss_G = train_generator(optimizer_G, SARs, fake_SAR, loss_function, Discriminator, batch_size)     
            
            ###### Train Discriminator ##########
            loss_D = train_discriminator(optimizer_D, SARs, fake_SAR, loss_function, Discriminator, batch_size)
            
            training_loss_G += loss_G.cpu().item()/len(train_loader.dataset.imgs)
            training_loss_D += loss_D.cpu().item()/len(train_loader.dataset.imgs)
            # Display Progress every few batches
        if epoch % nb_display_result == 0: 
            
            print("At epoch #" + str(epoch+1) +"/"+ str(number_epochs) 
                  + ", Generator training loss = " + str(training_loss_G) + 
                  ", Discriminator training loss = " + str(training_loss_D) 
                  )
            training_losses_G.append(training_loss_G)
            training_losses_D.append(training_loss_D)
            training_loss_G = 0
            training_loss_D = 0
  
            #Validation phase:
            Generator.eval()
            Discriminator.eval()
          
            #This line reduces the memory by not tracking the gradients. Also to be used
            #during inference.
            with torch.no_grad():
                for j,  (imgs_val, SARs_val) in enumerate(validate_loader):
                    imgs_val = Variable(imgs_val).to(device)
                    SARs_val = Variable(SARs_val).to(device)
                  
                    fake_SAR_val = Generator(imgs_val.float())
                    
                    loss_G = validate_generator(optimizer_G, SARs_val, fake_SAR_val, loss_function, Discriminator, batch_size)     
                    loss_D = validate_discriminator(optimizer_D, SARs_val, fake_SAR_val, loss_function, Discriminator, batch_size)
    
                    
                    validation_loss_G += loss_G.cpu().item() / len(validate_loader.dataset.imgs)
                    validation_loss_D += loss_D.cpu().item() / len(validate_loader.dataset.imgs)
              
                    
                if marqueur == 0:
                    SARs_val_display = SARs_val.cpu().data.numpy()[0]
                    SARs_val_display = SARs_val_display.reshape((256,256))
                    plt.figure()
                    plt.imshow(SARs_val_display, cmap = "gray")
                    plt.title("Image SAR de référence") 
                    plt.savefig("C:/Users/natsl/Documents/These/result/zone1/image1"+str(j)+"_ref.png")
                marqueur = 1
                # Enregistrement de la première image de validation pour visualiser l'évolution
                fake_SAR_val_display = fake_SAR_val.cpu().detach().numpy()[0]
                fake_SAR_val_display = fake_SAR_val_display.reshape((256,256))
                plt.figure()
                plt.imshow(fake_SAR_val_display, cmap = "gray")
                plt.title("Image SAR générée à l'epoch " + str(epoch)) 
                plt.savefig("C:/Users/natsl/Documents/These/result/zone1/image_generee_epoch_"+str(epoch)+".png")
            
            # Determine approximate time left
            epoch_done = epoch + 1
            epoch_left = number_epochs - epoch_done
            time_left = datetime.timedelta(seconds=epoch_left * (time.time() - start_time)/ epoch_done)
    
            print("At epoch #" + str(epoch+1) + ", Generator validation loss = " +
                  str(validation_loss_G)  + ", Discriminator validation loss = " +
                  str(validation_loss_D)+ " " + str(time_left) + " hours left" + "\n")
            validation_losses_G.append(validation_loss_G)
            validation_losses_D.append(validation_loss_D)
            training_loss_G = 0
            training_loss_D = 0
      
        if epoch > 0:
            
            plt.figure()
            plt.title("Courbes des loss durant l'apprentissage")
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.plot(np.arange(len(training_losses_G)), training_losses_G, label = 'Generator training loss')
            plt.plot(np.arange(len(training_losses_D)), training_losses_D, label = 'Discriminator training loss')
            plt.plot(np.arange(len(validation_losses_G)), validation_losses_G, label = 'Generator validation loss')
            plt.plot(np.arange(len(validation_losses_D)), validation_losses_D, label = 'Discriminator validation loss')
            plt.legend()
            plt.savefig("C:/Users/natsl/Documents/These/result/zone1/courbe_loss"+str(epoch)+".png")
            plt.show()
            
            
    
        # Save network
        state_generator = {
        'epoch': epoch,
        'state_dict': Generator.state_dict(),
        'optimizer': optimizer_G.state_dict()
        }
        state_discriminator = {
        'epoch': epoch,
        'state_dict': Discriminator.state_dict(),
        'optimizer': optimizer_D.state_dict()
        }
        torch.save(state_generator, "C:/Users/natsl/Documents/These/result/Generator_P2P_Unet_" + str(epoch) + "_epochs.pth")
        torch.save(state_discriminator, "C:/Users/natsl/Documents/These/result/Discriminator_P2P_PatchGAN_" + str(epoch) + "_epochs.pth")
    