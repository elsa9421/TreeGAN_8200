import torch
import torch.nn as nn
import torch.optim as optim

from data.dataset_benchmark import BenchmarkDataset
from model.gan_network import Generator, Discriminator
from model.gradient_penalty import GradientPenalty
from evaluation.FPD import calculate_fpd

from arguments import Arguments


import time
#import visdom
import numpy as np


## Adding import statement

import os

## Matplotlib visualization 

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import copy


def visualize_3d(data, fig, num, angles, row_no, rows=10, azim=90, dist=5.5, elev=20):

    
#     angles = [30, 60, 90, 120, 180, 210, 270]
    for idx, i in enumerate(angles):
        ax = fig.add_subplot(rows, len(angles), num + idx, projection='3d')
        ax.azim = i
        ax.dist = dist
        ax.elev = elev
        ax.scatter3D(data[:, 0], data[:, 1], data[:, 2]);
        # Turn off tick labels
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_zticklabels([])
        ax.set_title("PC={} at Angle={} degree".format(row_no, i))
 
    
      
    return fig 
    
    

class TreeGAN():
    def __init__(self, args):
        self.args = args
        # ------------------------------------------------Dataset---------------------------------------------- #
        print("Self.args.train=", self.args.train)
        
        if self.args.train:
            self.data = BenchmarkDataset(root=args.dataset_path, npoints=args.point_num, uniform=False, class_choice=args.class_choice)
            

            self.dataLoader = torch.utils.data.DataLoader(self.data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
            print("Training Dataset : {} prepared.".format(len(self.data)))
        # ----------------------------------------------------------------------------------------------------- #

        # -------------------------------------------------Module---------------------------------------------- #
        self.G = Generator(batch_size=args.batch_size, features=args.G_FEAT, degrees=args.DEGREE, support=args.support).to(args.device)
        self.D = Discriminator(batch_size=args.batch_size, features=args.D_FEAT).to(args.device)             
        
        self.optimizerG = optim.Adam(self.G.parameters(), lr=args.lr, betas=(0, 0.99))
        self.optimizerD = optim.Adam(self.D.parameters(), lr=args.lr, betas=(0, 0.99))

        self.GP = GradientPenalty(args.lambdaGP, gamma=1, device=args.device)
        print("Network prepared.")

    
    def interpolation(self, load_ckpt=None, save_images=None, save_pts_files=None, epoch=0):           
        
        if args.train:
            if not os.path.isdir(os.path.join(save_images, "Matplot_Images")):
                print("Making a directory!")
                os.mkdir(os.path.join(save_images, "Matplot_Images"))
            SAVE_IMAGES = os.path.join(save_images, "Matplot_Images")
            if not os.path.isdir(os.path.join(save_pts_files, "Points")):
                os.mkdir(os.path.join(save_pts_files, "Points"))
            SAVE_PTS_FILES =os.path.join(save_pts_files, "Points")
            epoch = str(epoch)
            args_copy = copy.deepcopy(args)
            
            args_copy.batch_size = 1

            Gen = TreeGAN(args_copy)
            
       
        if not args.train:
            SAVE_IMAGES = save_images
            SAVE_PTS_FILES = save_pts_files
            epoch = ''
            Gen = self
            
            
        if load_ckpt is not None:
            
            
            checkpoint = torch.load(load_ckpt, map_location=self.args.device)            
           # self.D.load_state_dict(checkpoint['D_state_dict'])
            Gen.G.load_state_dict(checkpoint['G_state_dict'])

            print("Checkpoint loaded in interpolation")
           
        Gen.G.zero_grad()
        with torch.no_grad():

            alpha = [0, 0.2, 0.4, 0.6, 0.8, 1]
            #seeds = [10, 40, 80, 100, 120, 140, 160]    # Make this an argument?
            seeds = self.args.seed
            print("The seed is===", seeds)
            angles = [90, 120, 210, 270]
            for s in seeds:
                np.random.seed(s)     
                z_a, z_b =  np.random.normal(size=96),  np.random.normal(size=96)

                fig_size = (30, 30)
                plt.axis('off')
                new_f = plt.figure(figsize=fig_size)
                flag = 1
                for row_no, a in enumerate(alpha):
                    z = torch.tensor((1 - a) * z_a + a * z_b , dtype=torch.float32).to(self.args.device)
                    z = z.reshape(1, 1, -1)

                    tree = [z]
                    fake_point = Gen.G(tree).detach()
                    generated_point = Gen.G.getPointcloud().cpu().detach().numpy()
                    new_f = visualize_3d(generated_point, fig=new_f, num=flag, angles=angles, row_no=row_no+1, rows=len(alpha))
                    flag += len(angles)

                    ## Creating .pts files for each z

                    list_out = generated_point.tolist()

                    if args.train:
                        f_path = os.path.join(SAVE_PTS_FILES, "Epoch_{}_Seed_{}_PC_{}.pts".format(epoch,s,row_no+1))
                    else:
                        f_path = os.path.join(SAVE_PTS_FILES, "Seed_{}_PC_{}.pts".format(s,row_no+1))
                                     
                    #f = open("/storage/TreeGAN_dataset/RS_{}_PC_{}.pts".format(s,row_no+1), "a")
                    f = open(f_path, "a")
                    for line in list_out:
                        Y= " ".join(list(map(str, line)))
                        f.write(Y + "\n")


                    f.close()
                    if args.train:
                        print("Written to Epoch_{}_Seed_{}_PC_{}.pts file".format(epoch,s,row_no+1))
                    else: 
                        print("Written to Seed_{}_PC_{}.pts file".format(s,row_no+1))




                    ####
                new_f.suptitle('Random Seed={}'.format(s), fontsize=14)
                #new_f.savefig('/storage/TreeGAN_dataset/new_to_'+str(s)+'.png')
                if args.train:
                    new_f.savefig(SAVE_IMAGES+'/Epoch_'+epoch+'_'+'Seed_'+str(s)+'.png')
                else:
                    new_f.savefig(SAVE_IMAGES+'/'+'Seed_'+str(s)+'.png')
        return
                




    def run(self, save_ckpt=None, load_ckpt=None, result_path=None):        


        epoch_log = 0
        
        loss_log = {'G_loss': [], 'D_loss': []}
        loss_legend = list(loss_log.keys())

        metric = {'FPD': []}
        if load_ckpt is not None:
            

            checkpoint = torch.load(load_ckpt, map_location=self.args.device)
            
            self.D.load_state_dict(checkpoint['D_state_dict'])
            self.G.load_state_dict(checkpoint['G_state_dict'])

            epoch_log = checkpoint['epoch']

            loss_log['G_loss'] = checkpoint['G_loss']
            loss_log['D_loss'] = checkpoint['D_loss']
            loss_legend = list(loss_log.keys())

            metric['FPD'] = checkpoint['FPD']
            
           
            
            print("Checkpoint loaded.")
            
            
            #################
            
#             self.G.zero_grad()

#             z = torch.randn(self.args.batch_size, 1, 96).to(self.args.device)
#             tree = [z]

#             fake_point = self.G(tree)   
#             generated_point = self.G.getPointcloud()
            
            
           
                    
#             out = generated_point.cpu().detach().numpy()

#             list_out = out.tolist()
#             f = open("/storage/TreeGAN_dataset/check_this.pts", "a")
#             for line in list_out:
#                 Y= " ".join(list(map(str, line)))
#                 f.write(Y + "\n")

#             f.close()

#             print("written to file")
            
            ################

        for epoch in range(epoch_log, self.args.epochs):
            for _iter, data in enumerate(self.dataLoader):
                # Start Time
                start_time = time.time()
                
                point = data
                point = point.to(self.args.device)

                # -------------------- Discriminator -------------------- #
                for d_iter in range(self.args.D_iter):
                    self.D.zero_grad()
                    
                    z = torch.randn(self.args.batch_size, 1, 96).to(self.args.device)
                    tree = [z]
                    
                    
                    with torch.no_grad():
                        fake_point = self.G(tree)    
                    
#                     print("fake_point.shape!=", fake_point.shape)
                   
                        
                        
                    D_real = self.D(point)
                    D_realm = D_real.mean()

                    D_fake = self.D(fake_point)
                    D_fakem = D_fake.mean()
                    
#                     print("checking point size", point.data.shape)
#                     print("CHECKING SIZE", fake_point.data.shape)
                    
                    gp_loss = self.GP(self.D, point.data, fake_point.data)
                    
                    d_loss = -D_realm + D_fakem
                    d_loss_gp = d_loss + gp_loss
                    d_loss_gp.backward()
                    self.optimizerD.step()
                   
                loss_log['D_loss'].append(d_loss.item())                  
                
                # ---------------------- Generator ---------------------- #
                self.G.zero_grad()
                
                z = torch.randn(self.args.batch_size, 1, 96).to(self.args.device)
                tree = [z]
                
                fake_point = self.G(tree)
                G_fake = self.D(fake_point)
                G_fakem = G_fake.mean()
                
                g_loss = -G_fakem
                g_loss.backward()
                self.optimizerG.step()

                loss_log['G_loss'].append(g_loss.item())

                # --------------------- Visualization -------------------- #

                print("[Epoch/Iter] ", "{:3} / {:3}".format(epoch, _iter),
                      "[ D_Loss ] ", "{: 7.6f}".format(d_loss), 
                      "[ G_Loss ] ", "{: 7.6f}".format(g_loss), 
                      "[ Time ] ", "{:4.2f}s".format(time.time()-start_time))


            # ---------------- Frechet Pointcloud Distance --------------- #
#             if epoch % self.args.save_at_epoch == 0 and not result_path == None:
#                  fake_pointclouds = torch.Tensor([])
#                  for i in range(10): # For 5000 samples
#                      z = torch.randn(self.args.batch_size, 1, 96).to(self.args.device)
#                      tree = [z]
#                      with torch.no_grad():
#                          sample = self.G(tree).cpu()
#                      fake_pointclouds = torch.cat((fake_pointclouds, sample), dim=0)

#                  fpd = calculate_fpd(fake_pointclouds, statistic_save_path=self.args.FPD_path, batch_size=100, dims=1808, device=self.args.device)
#                  metric['FPD'].append(fpd)
#                  print('[{:4} Epoch] Frechet Pointcloud Distance <<< {:.10f} >>>'.format(epoch, fpd))
                
#                  del fake_pointclouds
 #-------------------------------------------------------------------------------
#                 class_name = args.class_choice if args.class_choice is not None else 'all'
                
#                 torch.save(fake_pointclouds, result_path+str(epoch)+'_'+class_name+'.pt')
#                 del fake_pointclouds
    
#             if epoch % self.args.save_at_epoch == 0:
#                 generated_point = self.G.getPointcloud()

#                 out = generated_point.cpu().detach().numpy()


#                 list_out = out.tolist()
#                 f = open("/storage/TreeGAN_dataset/sample"+str(epoch+1)+".pts", "a")
#                 for line in list_out:
#                     Y= " ".join(list(map(str, line)))
#                     f.write(Y + "\n")

#                 f.close()

#                 print("written to file")

            # ---------------------- Save checkpoint --------------------- #
            if (epoch+1) % self.args.save_at_epoch == 0 and not save_ckpt == None:
                torch.save({
                        'epoch': epoch,
                        'D_state_dict': self.D.state_dict(),
                        'G_state_dict': self.G.state_dict(),
                        'D_loss': loss_log['D_loss'],
                        'G_loss': loss_log['G_loss'],
                        'FPD': metric['FPD']
                }, save_ckpt+str(epoch+1)+'.pt')

                print('Checkpoint at {} epoch is saved.'.format(epoch+1))
                
                
                # --------------Saving intermediate images and .pts files----------------------#
                
                self.interpolation(load_ckpt=save_ckpt+str(epoch+1)+'.pt', save_images=result_path, save_pts_files=result_path, epoch=epoch+1)

                
                    

if __name__ == '__main__':
    args = Arguments().parser().parse_args()

    args.device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.device)
    SAVE_IMAGES = args.save_images
    SAVE_PTS_FILES = args.save_pts_files
    
    
    
    
    
    ## Code for testing
    if args.train == "False":
         args.train = False
         if args.ckpt_load is None:
                raise Exception("No checkpoint path provided!")
         LOAD_CHECKPOINT = os.path.join(args.ckpt_path, args.ckpt_load)
         args.batch_size = 1   # for interpolation!
         model = TreeGAN(args)
         model.interpolation(load_ckpt=LOAD_CHECKPOINT, save_images=SAVE_IMAGES, save_pts_files=SAVE_PTS_FILES)
        
        
    ## Code for training
    else:
        args.train = True
        if args.ckpt_load == "None":
            args.ckpt_load = None
        SAVE_CHECKPOINT = os.path.join(args.ckpt_path, args.ckpt_save) if args.ckpt_save is not None else None
        LOAD_CHECKPOINT = os.path.join(args.ckpt_path, args.ckpt_load) if args.ckpt_load is not None else None
#         RESULT_PATH = os.path.join(args.result_path, args.result_save)
        RESULT_PATH = args.result_path  
        model = TreeGAN(args)
        model.run(save_ckpt=SAVE_CHECKPOINT, load_ckpt=LOAD_CHECKPOINT, result_path=RESULT_PATH)
        
