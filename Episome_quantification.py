import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import math
from skimage import io, img_as_ubyte
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import area_closing, area_opening, binary_dilation, diamond
from skimage.measure import regionprops
from skimage.filters import threshold_otsu, gaussian
from cellpose import models
from scipy import ndimage as ndi
from scipy.signal import medfilt

mpl.rcParams['image.interpolation'] = 'none'
mpl.rcParams['image.cmap']='gray'
mpl.rcParams['xtick.bottom']=False
mpl.rcParams['xtick.labelbottom']=False
mpl.rcParams['ytick.left']=False
mpl.rcParams['ytick.labelleft']=False
mpl.rcParams['savefig.pad_inches']=0.0
plt.rcParams['figure.dpi'] = 300

class LoadImages:
    
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
                        self.dict[file]['CMV']=self.src+"\\"+file+"\\"+img
                    elif "t0c2" in img:
                        self.dict[file]['DAPI']=self.src+"\\"+file+"\\"+img
    
    def GetImage(self, fname):
        b=io.imread(self.dict[fname]['DAPI'])
        g=io.imread(self.dict[fname]['LEL'])
        r=io.imread(self.dict[fname]['CMV'])
        return b,g,r
    
    def SaveMasks(self, fname, lectin_mask=None, nuclei_mask=None, cmv_mask=None, vas=None, ext=None):
        mask_src=self.src+"\\"+fname+"\\Masks"
        if not os.path.exists(mask_src): 
            os.makedirs(mask_src) 
        if lectin_mask is not None:
            io.imsave(mask_src+"\\LEL.mask.png", img_as_ubyte(lectin_mask), cmap='gray')
        if nuclei_mask is not None:
            io.imsave(mask_src+"\\nuclei.mask.png", img_as_ubyte(nuclei_mask), cmap='gray')
        if cmv_mask is not None:
            io.imsave(mask_src+"\\CMV.mask.png", img_as_ubyte(cmv_mask), cmap='gray')
        if vas is not None:
            io.imsave(mask_src+"\\vascular_nuclei.mask.png", img_as_ubyte(vas), cmap='gray')
        if ext is not None:
            io.imsave(mask_src+"\\extra_nuclei.mask.png", img_as_ubyte(ext), cmap='gray')
        
    def LoadMasks(self, fname):
        mask_src=self.src+"\\"+fname+"\\Masks"
        LEL_mask=(io.imread(mask_src+"\\LEL.mask.png")>0)
        CMV_mask=(io.imread(mask_src+"\\CMV.mask.png")>0)
        nuclei_mask=(io.imread(mask_src+"\\nuclei.mask.png")>0)
        return LEL_mask, CMV_mask, nuclei_mask
    
class IdentifyVascularNuclei:
    
    def __init__(self, vessles, nuclei, vessle_mask=None, nuclei_mask=None):
        self.vessle_int=vessles
        self.nuclei_int=nuclei
        if vessle_mask is not None:
            self.vessle_mask=vessle_mask
        else:
            self.get_vessle_mask()
        if nuclei_mask is not None:
            self.nuclei_mask=nuclei_mask
        else:
            self.get_nuclei_mask(self.nuclei_int)
        self.find_vascular_nuclei()
    
    def get_vessle_mask(self, blur_sigma=8, thr_offset=0.8, opening_area=499, closing_area=499):
        blur=gaussian(self.vessle_int, sigma=blur_sigma)
        mask=blur>threshold_otsu(blur)*thr_offset
        rm_small=area_opening(mask, area_threshold=opening_area)
        self.vessle_mask = area_closing(rm_small, area_threshold=closing_area)
    
    def get_nuclei_mask(self, img):
        nuc_model = models.Cellpose(gpu=False, model_type='nuclei')
        nuc_mask, flows, styles, diams = nuc_model.eval([img], diameter=60, channels=[0,0],
                                                 flow_threshold=0.4, do_3D=False)
        self.nuclei_mask = nuc_mask[0]>0
        
    def find_vascular_nuclei(self):
        objs=regionprops(ndi.label(self.nuclei_mask)[0], self.vessle_mask.astype(int))
        self.vascular_nuclei=[]
        self.extravascular_nuclei=[]
        for obj in objs:
            if np.count_nonzero(obj.image_intensity)/obj.area >0.9:
                self.vascular_nuclei.append(obj)
            else:
                self.extravascular_nuclei.append(obj)
                
    def get_sorted_masks(self):
        vasular=np.isin(ndi.label(self.nuclei_mask)[0],[obj.label for obj in self.vascular_nuclei])
        extravasular=np.isin(ndi.label(self.nuclei_mask)[0],[obj.label for obj in self.extravascular_nuclei])
        return vasular, extravasular
          
#%% Segmentation of nuclear mask

class NucleiSegmentation:
    
    def __init__(self, nuc_mask):
        self.ori_nuc=nuc_mask
        self.objs=regionprops(ndi.label(self.ori_nuc)[0])
        
    def SegmentRoundness(self):
        seg_out=np.zeros_like(self.ori_nuc).astype(int)
        for obj in self.objs:
                roundness=(obj.perimeter*obj.perimeter)/(math.pi*4*obj.area)
                if roundness > 1.3:
                    segmented=self.WatershedSegmentation(obj)
                    labels=ndi.label(segmented)[0]
                    sl=obj.slice
                    seg_out[sl[0].start:sl[0].stop,sl[1].start:sl[1].stop]+=labels
                else:
                    sl=obj.slice
                    seg_out[sl[0].start:sl[0].stop,sl[1].start:sl[1].stop]+=obj.image_filled
        return seg_out>0
    
    def FilterArea(self, thr, mode='above'):
        if mode=='above':
            self.objs=[obj for obj in self.objs if obj.area>thr]
        elif mode == 'below':
            self.objs=[obj for obj in self.objs if obj.area<thr]
    
    def WatershedSegmentation(self, obj):
        distance = ndi.distance_transform_edt(obj.image_filled)
        coords = peak_local_max(distance, footprint=np.ones((20, 20)), labels=obj.image_filled, exclude_border=False, min_distance=15)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labels = watershed(-distance, markers, mask=obj.image_filled, watershed_line=True)
        return labels
    
def IdentifyAdditionalNuc(nuc_int, nuc_mask, thr=6000):
    remnant_mask=(nuc_int*np.invert(binary_dilation(nuc_mask)))>6000
    smoothed_remnant=medfilt(remnant_mask.astype(int),kernel_size=15)
    
    seg_remnant=NucleiSegmentation(smoothed_remnant)
    seg_remnant.FilterArea(500)
    add_nuc=seg_remnant.SegmentRoundness()
    return nuc_mask+add_nuc

#%% Generate all masks
src=r'E:\Tiffany GBA CMV\MIPs'
os.chdir(src)
images=LoadImages(src+"\SingleChannel")
len_images=len(list(images.dict.keys()))

for num, i in enumerate(list(images.dict.keys())):
    print(num,"/",len_images)
    
    b,g,r=images.GetImage(i)
        
    vessle_analysis=IdentifyVascularNuclei(g,b)
    vessle_analysis.get_vessle_mask(blur_sigma=14)
    nuc_seg=NucleiSegmentation(vessle_analysis.nuclei_mask).SegmentRoundness()
    nuc_seg=IdentifyAdditionalNuc(b, nuc_seg)
    
    vessle_analysis=IdentifyVascularNuclei(g,b, vessle_mask=vessle_analysis.vessle_mask, nuclei_mask=nuc_seg)
    vascular, extra = vessle_analysis.get_sorted_masks()
    images.SaveMasks(i,
                     lectin_mask=vessle_analysis.vessle_mask, 
                     nuclei_mask=nuc_seg, 
                     cmv_mask=r>5750, 
                     vas=vascular, 
                     ext=extra)    

#%% Analysis on generated masks

src=r'E:\Tiffany GBA CMV\MIPs'
os.chdir(src)
images=LoadImages(src+"\SingleChannel")

results={}

for num, i in enumerate(list(images.dict.keys())):
    results[i]={}
    print("sample nr: ", num+1)
    print("processing: ", i)

    b,g,r=images.GetImage(i)
    vas, cmv, nuc = images.LoadMasks(i)
    vessle_analysis=IdentifyVascularNuclei(g,b, vessle_mask=vas, nuclei_mask=nuc)
    vascular, extra = vessle_analysis.get_sorted_masks()
    
    results[i]['pct_extravascular_cmv']=np.count_nonzero(cmv*extra)/(np.count_nonzero(cmv*(vascular+extra)))*100
    results[i]['pct_vascular_cmv']=np.count_nonzero(cmv*vascular)/(np.count_nonzero(cmv*(vascular+extra)))*100
    results[i]['pct_nuclear_cmv']=(np.count_nonzero(cmv*(binary_dilation(vascular+extra, footprint=diamond(3))))/np.count_nonzero(cmv))*100
    results[i]['total_cmv_area']=np.count_nonzero(cmv)
    vascular_count=ndi.label(vascular)[1]
    extra_count=ndi.label(extra)[1]
    results[i]['pct endothelial cells']=(vascular_count/(vascular_count+extra_count))*100
    
df=pd.DataFrame(results).T

df['treatment']=df.index.str.split("_").str[1]
df['region']=df.index.str.split("_").str[7]
df['animal']=df.index.str.split("_").str[2]
df.to_csv('../20240323_cmv_quantification.csv', index=False)

#%% QC masks

src=r'E:\Tiffany GBA CMV\MIPs'
os.chdir(src)
images=LoadImages(src+"\SingleChannel")

if not os.path.exists(r"../QC_masks"): 
    os.makedirs(r"../QC_masks") 

for num, i in enumerate(list(images.dict.keys())):
    print("sample nr: ", num+1)
    print("processing: ", i)

    b,g,r=images.GetImage(i)
    ves, cmv, nuc = images.LoadMasks(i)

    title = " ".join(i.split("_")[1:3] +[i.split("_")[-3]])
    
    fig_vas, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
    ax1.set_title('Vasculature')
    ax1.imshow(g)
    ax2.set_title('Mask')
    ax2.imshow(ves)
    ax3.set_title('Masked Vasculature')
    ax3.imshow(g*np.invert(ves), vmax=np.amax(g))
    plt.suptitle(f"{title}", fontsize=15)
    plt.tight_layout()
    plt.savefig(f"../QC_masks/{title}.vasculature.png", dpi=300)
    plt.show()
    
    
    fig_nuc, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
    ax1.set_title('Nuclei')
    ax1.imshow(b)
    ax2.set_title('Mask')
    ax2.imshow(nuc)
    ax3.set_title('Masked Nuclei')
    ax3.imshow(b*np.invert(nuc), vmax=np.amax(b))
    plt.suptitle(f"{title}", fontsize=15)
    plt.tight_layout()
    plt.savefig(f"../QC_masks/{title}.nuclei.png", dpi=300)
    plt.show()
    
    fig_cmv, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
    ax1.set_title('Genomes')
    ax1.imshow(r)
    ax2.set_title('Mask')
    ax2.imshow(cmv)
    ax3.set_title('Masked Genomes')
    ax3.imshow(r*np.invert(cmv), vmax=np.amax(r))
    plt.suptitle(f"{title}", fontsize=15)
    plt.savefig(f"../QC_masks/{title}.genomes.png", dpi=300)
    plt.tight_layout()
    plt.show()
