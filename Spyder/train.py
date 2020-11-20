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
    N = SARs.size(0)
    conversion = T.ToTensor()
    valid = Variable(conversion(np.ones((N, batch_size))), requires_grad=False)
    fake = Variable(conversion(np.zeros((N, batch_size))), requires_grad=False)
    
    optimizer_D.zero_grad()
    
    # Train on Real Data
    Discriminator_tensor_real= torch.cat((SARs.float(),  SARs.float()),1)
    prediction_real = Discriminator(Discriminator_tensor_real)
    prediction_real = torch.reshape(prediction_real, (N, 16))
    # Calculate error and backpropagate
    error_real = loss_function(prediction_real, valid)
    error_real.backward(retain_graph=True)

    # Train on Fake Data
    Discriminator_tensor_fake= torch.cat((fake_SAR.float(),  SARs.float()),1)
    prediction_fake = Discriminator(Discriminator_tensor_fake)
    prediction_fake = torch.reshape(prediction_fake, (N, 16))
    # Calculate error and backpropagate
    error_fake = loss_function(prediction_fake, fake)
    error_discriminator = (error_real + error_fake)*0.5 
    error_fake.backward(retain_graph=True)
    
    # Update weights
    optimizer_D.step()
    
    # Return error and predictions for real and fake inputs
    return error_discriminator 

def validate_discriminator(optimizer_D, SARs, fake_SAR, loss_function, Discriminator, batch_size):
    N = SARs.size(0)
    conversion = T.ToTensor()
    valid = Variable(conversion(np.ones((N, batch_size))), requires_grad=False)
    fake = Variable(conversion(np.zeros((N, batch_size))), requires_grad=False)
    
    # Validate on Real Data
    Discriminator_tensor_real= torch.cat((SARs.float(),  SARs.float()),1)
    prediction_real = Discriminator(Discriminator_tensor_real)
    prediction_real = torch.reshape(prediction_real, (N, 16))
    # Calculate error 
    error_real = loss_function(prediction_real, valid)

    # Validate on Fake Data
    Discriminator_tensor_fake= torch.cat((fake_SAR.float(),  SARs.float()),1)
    prediction_fake = Discriminator(Discriminator_tensor_fake)
    prediction_fake = torch.reshape(prediction_fake, (N, 16))
    # Calculate error
    error_fake = loss_function(prediction_fake, fake)
    
    return (error_real + error_fake)*0.5 

def train_generator(optimizer_G, SARs, fake_SAR, loss_function, Discriminator, batch_size):
    N = fake_SAR.size(0)
    conversion = T.ToTensor()
    
    optimizer_G.zero_grad()
    Discriminator_tensor= torch.cat((fake_SAR.float(),  SARs.float()),1)
    prediction = Discriminator(Discriminator_tensor)
    prediction = torch.reshape(prediction, (N, 16))
    valid = Variable(conversion(np.ones(prediction.shape)), requires_grad=False)
    valid = torch.reshape(valid, (N,16))
    
    # Calculate error and backpropagate
    error = loss_function(prediction, valid)
    error.backward(retain_graph=True)
    # Update weights with gradients
    optimizer_G.step()
    # Return error
    return error

def validate_generator(optimizer_G, SARs, fake_SAR, loss_function, Discriminator, batch_size):
    N = fake_SAR.size(0)
    conversion = T.ToTensor()
    
    Discriminator_tensor= torch.cat((fake_SAR.float(),  SARs.float()),1)
    prediction = Discriminator(Discriminator_tensor)
    prediction = torch.reshape(prediction, (N, 16))
    valid = Variable(conversion(np.ones((prediction.shape))), requires_grad=False)
    valid = torch.reshape(valid, (N,16))
    # Calculate error 
    error = loss_function(prediction, valid)
    return error

def train_GAN(number_epochs, Generator, Discriminator, train_loader, 
              validate_loader, batch_size):
    
    nb_display_result = 10
    start_time = time.time()
    training_losses_G = []
    training_losses_D = []
    validation_losses_G = []
    validation_losses_D = []
    conversion = T.ToTensor()
    # Loss functions
    loss_function = torch.nn.L1Loss()
    
    # Optimizers
    optimizer_G = torch.optim.Adam(Generator.parameters(), lr=0.001)
    optimizer_D = torch.optim.Adam(Discriminator.parameters(), lr=0.001)
    
    training_loss_G = 0
    training_loss_D = 0
    
    validation_loss_G = 0
    validation_loss_D = 0
    
    for epoch in range(number_epochs):
        print("Starting epoch number " + str(epoch))
      
        Generator.train() 
        Discriminator.train()
      
        for i, imgs in enumerate(train_loader.dataset.imgs):
            SARs = conversion(train_loader.dataset.SARs[i])
            #Convert the imgs and SARs to torch Variable (will hold the computation
            #graph).
            imgs = Variable(imgs)
            SARs = Variable(SARs).type(torch.LongTensor)
            SARs = torch.reshape(SARs, (1, 1, 256, 256))
            imgs = torch.reshape(imgs, (1, 3, 256, 256))
            
            fake_SAR = Generator(imgs.float())  
            
            ##### Train Generator #########
            loss_G = train_generator(optimizer_G, SARs, fake_SAR, loss_function, Discriminator, batch_size)     
            
            ###### Train Discriminator ##########
            loss_D = train_discriminator(optimizer_D, SARs, fake_SAR, loss_function, Discriminator, batch_size)
            
            training_loss_G += loss_G.cpu().item()/nb_display_result
            training_loss_D += loss_D.cpu().item()/nb_display_result
            # Display Progress every few batches
            if (epoch*len(train_loader.dataset.imgs) +i) % nb_display_result == 0: 
                
                # Determine approximate time left
                batches_done = epoch * len(train_loader.dataset.imgs) + i
                batches_left = number_epochs * len(train_loader.dataset.imgs) - batches_done
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - start_time)/ (batches_done + 1))
        
                print("At epoch #" + str(epoch) +"/"+ str(number_epochs) +
                      " and batch " + str(i)+ "/" + str(len(train_loader.dataset.imgs)) 
                      + ", Generator training loss = " + str(training_loss_G) + 
                      ", Discriminator training loss = " + str(training_loss_D) 
                      + " " + str(time_left) + " hours left")
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
            for i, imgs in enumerate(validate_loader.dataset.imgs):
                SARs = conversion(train_loader.dataset.SARs[i])
                imgs = Variable(imgs)
                SARs = Variable(SARs).type(torch.LongTensor)
                SARs = torch.reshape(SARs, (1, 1, 256, 256))
                imgs = torch.reshape(imgs, (1, 3, 256, 256))
              
                fake_SAR = Generator(imgs.float())
                
                loss_G = validate_generator(optimizer_G, SARs, fake_SAR, loss_function, Discriminator, batch_size)     
                loss_D = validate_discriminator(optimizer_D, SARs, fake_SAR, loss_function, Discriminator, batch_size)

                
                validation_loss_G += loss_G.cpu().item() / len(validate_loader.dataset.imgs)
                validation_loss_D += loss_D.cpu().item() / len(validate_loader.dataset.imgs)
        
        print("At epoch #" + str(epoch) + ", Generator validation loss = " +
              str(validation_loss_G)  + ", Discriminator validation loss = " +
              str(validation_loss_D))
        validation_losses_G.append(validation_loss_G)
        validation_losses_D.append(validation_loss_D)
        training_loss_G = 0
        training_loss_D = 0
      
        if epoch > 0:
            
            plt.figure()
            plt.plot(np.arange(len(training_losses_G)), training_losses_G)
            plt.plot(np.arange(len(training_losses_D)), training_losses_D)
            # plt.plot(np.arange(len(validation_losses_G)), validation_losses_G)
            # plt.plot(np.arange(len(validation_losses_D)), validation_losses_D)
            plt.show()
    
    #Optional, if you want to save your model:
    #torch.save(network.state_dict(), os.path.join(base_folder, 'WeightsVaihingen/', 'Hypercolumns_' + str(number_epochs) + "epochs.pth")
    # torch.save(network.state_dict(), os.path.join(base_folder, 'WeightsVaihingen/', 'Hypercolumns_augm_weigths_' + str(number_epochs) + 'epochs.pth'))