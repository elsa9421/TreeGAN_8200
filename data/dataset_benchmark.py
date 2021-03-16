# from __future__ import print_function
# import torch.utils.data as data
# import os
# import os.path
# import torch
# import numpy as np



# class BenchmarkDataset(data.Dataset):
#     def __init__(self, root, npoints=2500, uniform=False, classification=False, class_choice=None):
#         self.npoints = npoints
#         self.root = root
#         #self.catfile = './data/synsetoffset2category.txt' 
#         # @@@@@@
#         self.catfile = '/storage/TreeGAN_dataset/TreeGAN/data/synsetoffset2category.txt'
        
#         #@@@@@@
#         self.cat = {}
#         self.uniform = uniform
#         print("UNIFORM", self.uniform)
#         self.classification = classification
        
        
        
#         #### Error Correction ######
        
#         if class_choice == "None":
#             class_choice = None
            
#         ############################

#         with open(self.catfile, 'r') as f:
#             for line in f:
#                 ls = line.strip().split()
#                 self.cat[ls[0]] = ls[1]
            
#         if not class_choice is None:
#             self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
            
        
#         print("self.cat=", self.cat)
#         self.meta = {}
#         for item in self.cat:
#             print("item=", item)

#             self.meta[item] = []
#             dir_point = os.path.join(self.root, self.cat[item], 'points')
#             dir_seg = os.path.join(self.root, self.cat[item], 'points_label')
            
#             ##### Commenting the following as 'sampling' does not exist in dataset 
            
#             #dir_sampling = os.path.join(self.root, self.cat[item], 'sampling')
            
           

#             fns = sorted(os.listdir(dir_point))


#             for fn in fns:
#                 token = (os.path.splitext(os.path.basename(fn))[0])
#                  ##### Editing self.meta as .sam does not exist
#             #   self.meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg'), os.path.join(dir_sampling, token + '.sam')))
            
             
#                 self.meta[item].append((os.path.join(dir_point, token + '.pts')))
                
#                  #####
#         print("line 67")

#         self.datapath = []
#         for item in self.cat:
#             for fn in self.meta[item]:                    
#                 #self.datapath.append((item, fn[0], fn[1]))
# #                 print("item=", item, "fn[0]=", fn)
#                 self.datapath.append((item, fn))
               


# #         self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
# #        # print("printing self.classes", self.classes)
# #         self.num_seg_classes = 0
# #         if not self.classification:
# #             for i in range(len(self.datapath)//50):
# #                 l = len(np.unique(np.loadtxt(self.datapath[i][-1]).astype(np.uint8)))
# #                 if l > self.num_seg_classes:
# #                     self.num_seg_classes = l

#     def __getitem__(self, index):
#         fn = self.datapath[index]
# #         print("In getitem!!")
# #         print("fn=", fn)
# #         cls = self.classes[self.datapath[index][0]]
#         point_set = np.loadtxt(fn[1]).astype(np.float32)
# #         seg = np.loadtxt(fn[2]).astype(np.int64)
        
# #         print("seg===", seg)
# #         print("len(Seg)=", len(seg))

# #         print("!!!!!!!!!!!!!!!!self.uniform", self.uniform)
#         if self.uniform:
#             choice = np.loadtxt(fn[3]).astype(np.int64)
#             assert len(choice) == self.npoints, "Need to match number of choice(2048) with number of vertices."
#         else:
#             choice = np.random.randint(0, len(point_set), size=self.npoints)
#             #choice = np.random.randint(0, len(seg), size=self.npoints)

#         point_set = point_set[choice]
# #         seg = seg[choice]

#         point_set = torch.from_numpy(point_set)
# #         seg = torch.from_numpy(seg)
# #         cls = torch.from_numpy(np.array([cls]).astype(np.int64))
#         if self.classification:
#             return point_set, cls
#         else:
#            # return point_set, seg
#             return point_set

#     def __len__(self):
#         return len(self.datapath)


from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np



class BenchmarkDataset(data.Dataset):
    def __init__(self, root, npoints=2500, uniform=False, classification=False, class_choice=None):
        self.npoints = npoints
        self.root = root
       # dir_point = os.path.join(self.root, 'points')    
        dir_point = self.root
        fns = sorted(os.listdir(dir_point))
        
        
        self.meta = []
        for fn in fns:    
          token = os.path.splitext(os.path.basename(fn))[0]   
          self.meta.append((os.path.join(dir_point, token + '.pts')))
        
#         print("self.meta", self.meta)

        self.datapath = []

        for fn in self.meta:                    
            self.datapath.append(fn)
    
         
               


    def __getitem__(self, index):
        

        fn = self.datapath[index]
        
        point_set = np.loadtxt(fn).astype(np.float32)
        choice = np.random.randint(0, len(point_set), size=self.npoints)
        point_set = point_set[choice]
        point_set = torch.from_numpy(point_set)

           
        return point_set

    def __len__(self):
        return len(self.datapath)

