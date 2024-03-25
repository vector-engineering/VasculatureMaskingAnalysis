#%% Imports
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from skimage.filters import threshold_otsu, gaussian
import pandas as pd
from skimage.color import rgb2gray
from skimage.morphology import area_closing, area_opening

#%% Plotting parameters for image display
mpl.rcParams['image.interpolation'] = 'none'
mpl.rcParams['image.cmap']='gray'
mpl.rcParams['xtick.bottom']=False
mpl.rcParams['xtick.labelbottom']=False
mpl.rcParams['ytick.left']=False
mpl.rcParams['ytick.labelleft']=False
mpl.rcParams['savefig.pad_inches']=0.0
plt.rcParams['figure.dpi'] = 300

#%% Modules

## Import this instead of copying the code
class LoadImages:
    #Load MIP images exported from Zen Blue.
    def __init__(self, src):
        self.src=src
        self.IndexImages()
        
    def IndexImages(self):
        self.dict={}
        for file in os.listdir(self.src):
            self.dict[file]={}
            for img in os.listdir(self.src+"\\"+file):
                if os.path.isfile(self.src+"\\"+file+"\\"+img):
                    if "t0c0" in img:
                        self.dict[file]['LEL']=self.src+"\\"+file+"\\"+img
                    elif "t0c1" in img:
                        self.dict[file]['GBA']=self.src+"\\"+file+"\\"+img
                    elif "t0c2" in img:
                        self.dict[file]['DAPI']=self.src+"\\"+file+"\\"+img
    
    def GetImage(self, fname):
        b=rgb2gray(plt.imread(self.dict[fname]['DAPI']))
        g=rgb2gray(plt.imread(self.dict[fname]['LEL']))
        r=rgb2gray(plt.imread(self.dict[fname]['GBA']))
        return b,g,r
    
    def SaveMasks(self, fname, lectin_mask, gba_mask, inside, outside):
        mask_src=self.src+"\\"+fname+"\\Masks"
        if not os.path.exists(mask_src): 
            os.makedirs(mask_src) 
        plt.imsave(mask_src+"\\LEL.mask.png", lectin_mask, cmap='gray')
        plt.imsave(mask_src+"\\GBA.mask.png", gba_mask, cmap='gray')
        plt.imsave(mask_src+"\\GBA_inside.mask.png", inside, cmap='gray')
        plt.imsave(mask_src+"\\GBA_outside.mask.png", outside, cmap='gray')
    
    def LoadMasks(self, fname):
        mask_src=self.src+"\\"+fname+"\\Masks"
        LEL_mask=(plt.imread(mask_src+"\\LEL.mask.png")>0)[:,:,0]
        GBA_mask=(plt.imread(mask_src+"\\GBA.mask.png")>0)[:,:,0]
        return LEL_mask, GBA_mask
    
class Analysis:
    # Name the class more specifically, like GBAAnalysis
    # Could identifyVascularNuclei and this class be combined? I'm noticing some shared features. Probably not worth the effort, but food for thought.
    def __init__(self, vessles, gba):
        self.vessle_int=vessles
        self.vessle_mask=self.get_vessle_mask()
        self.gba_int=gba
    
    def get_vessle_mask(self, blur_sigma=8, thr_offset=0.8, opening_area=499, closing_area=199):
        blur=gaussian(self.vessle_int, sigma=blur_sigma)
        mask=blur>threshold_otsu(blur)*thr_offset
        rm_small=area_opening(mask, area_threshold=opening_area)
        return area_closing(rm_small, area_threshold=closing_area)
    
    def get_gba_mask(self, thr):
        self.gba_mask=self.gba_int>thr

    def area_inside_vessles(self):
        if np.count_nonzero(self.gba_mask) > 0:
            setattr(self, "area_inside_vessles", (np.count_nonzero(self.gba_mask[self.vessle_mask])/np.count_nonzero(self.gba_mask))*100)
        else:
            setattr(self, "area_inside_vessles", 0)
            
    def area_outside_vessles(self):
        if np.count_nonzero(self.gba_mask) > 0:
            setattr(self, "area_outside_vessles", (np.count_nonzero(self.gba_mask*np.invert(self.vessle_mask))/np.count_nonzero(self.gba_mask))*100)
        else:
            setattr(self, "area_outside_vessles", 0)
        
    def tot_int_inside_vessles(self):
        setattr(self, "tot_int_inside_vessles", np.sum(self.gba_int[self.vessle_mask]))
    
    def tot_int_outside_vessles(self):
        setattr(self, "tot_int_outside_vessles", np.sum(self.gba_int[~self.vessle_mask]))

    def mean_int_inside_vessles(self):
        setattr(self, "mean_int_inside_vessles", np.mean(self.gba_int[self.vessle_mask]))
    
    def mean_int_outside_vessles(self):
        setattr(self, "mean_int_outside_vessles", np.mean(self.gba_int[~self.vessle_mask]))
                
    def measure(self):
        self.area_inside_vessles()
        self.area_outside_vessles()
        self.tot_int_inside_vessles()
        self.tot_int_outside_vessles()
        self.mean_int_inside_vessles()
        self.mean_int_outside_vessles()
        
    def data_to_df(self):
        return pd.Series({'area_inside_vessles':self.area_inside_vessles,
                          'area_outside_vessles':self.area_outside_vessles,
                          'tot_int_inside_vessles' : self.tot_int_inside_vessles,
                          'tot_int_outside_vessles' : self.tot_int_outside_vessles,
                          'mean_int_inside_vessles' : self.mean_int_inside_vessles,
                          'mean_int_outside_vessles' : self.mean_int_outside_vessles})
    
    def export_images(self, fname):
        os.makedirs('./masks', mode=0o777, exist_ok=True)
        
        fig, ((ax1,ax2,ax3),(ax4,ax5,ax6))=plt.subplots(2,3, dpi=600, figsize=(12,8))
        ax1.set_title('GBA')
        ax1.imshow(self.gba_int)
        ax4.set_title('Vessles')
        ax4.imshow(self.vessle_int)
        ax2.set_title('Vessle Mask')
        ax2.imshow(self.vessle_mask)
        ax5.set_title('Masked Vessles')
        ax5.imshow(self.vessle_int*np.invert(self.vessle_mask))
        ax3.set_title('GBA in Vessles')
        ax3.imshow(self.gba_int*self.vessle_mask)
        ax6.set_title('GBA outside Vessles')
        ax6.imshow(self.gba_int*np.invert(self.vessle_mask))
        plt.tight_layout()
        plt.savefig(f'./masks/{fname}.png')
        
# All of this to go under a if __name__ == "__main__": block
src=r'E:\Imaging\20240301 TifGBA GBA stacks\MIPs'
os.chdir(src)

images=LoadImages(src+"\SingleChannel")
results={}
len_samples=len(list(images.dict.keys()))
for nr, image in enumerate(list(images.dict.keys())):
    image=list(images.dict.keys())[4]
    print(nr, "/", len_samples)
    b,g,r=images.GetImage(image)
           
    vessle_analysis=Analysis(g,r)
    vessle_analysis.get_gba_mask(0.25)
    vessle_analysis.measure()
    vessle_analysis.export_images(image)
    results[image]=vessle_analysis.data_to_df()
    images.SaveMasks(image, vessle_analysis.vessle_mask, 
                       vessle_analysis.gba_mask,
                       vessle_analysis.gba_mask*vessle_analysis.vessle_mask, 
                       vessle_analysis.gba_mask*np.invert(vessle_analysis.vessle_mask))


df=pd.DataFrame(results).T

df2=df.copy()
df2[['Sample', 'Region']] = df2[df2.columns[0]].str.split('_', expand=True)
df2=df2[df2.columns[1:]]
df2.to_csv('../20240321_GBA-RNA_localization.csv', index=None)
