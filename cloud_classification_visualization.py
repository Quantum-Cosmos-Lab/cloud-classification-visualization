from xml.etree.ElementInclude import include
import numpy as np # linear algebra


import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm

import seaborn as sns
from scipy.stats import pearsonr  

BARdict = {'red':   ((0.0,  0.0, 0.0),
                   (0.5, 0.0, 0.0),
                   (1.0,  1.0, 1.0)),

         'green': ((0.0,  0.0, 0.0),
                   (1.0, 0.0, 0.0),),

         'blue':  ((0.0,  1.0, 1.0),
                   (0.5,  0.0, 0.0),
                   (1.0,  0.0, 0.0)),

         'alpha':  ((0.0,  1.0, 1.0),
                    (0.25, 1.0, 1.0),
                    (0.5,  0.0, 0.0),
                    (0.75, 1.0, 1.0),
                    (1.0,  1.0, 1.0))
                   }

ARdict = {'red':   ((0.0,  0.0, 0.0),
                   (0.5, 0.0, 0.0),
                   (1.0,  1.0, 1.0)),

         'green': ((0.0,  0.0, 0.0),
                   (1.0, 0.0, 0.0),),

         'blue':  ((0.0,  0.0, 0.0),
                   (0.5,  0.0, 0.0),
                   (1.0,  0.0, 0.0)),

        'alpha':    ((0.0,0.0,0.0),
                    (0.75,0.0,0.0),
                    (1.0,1.0,1.0))

                   }

BlueAlphaRed = LinearSegmentedColormap('BlueAlphaRed', BARdict)
AlphaRed = LinearSegmentedColormap('BlueRed', ARdict)

def plot_scene_mask(scene, mask, figsize = (10,9)):
    fig, ax = plt.subplots(1,2, figsize=figsize)
    ax[0].imshow(scene)
    ax[1].imshow(mask)
    plt.show()

def plot_scene_mask_difference(scene, mask, difference, cmap = BlueAlphaRed, figsize = (24,8)):
    fig, ax = plt.subplots(1,3, figsize=figsize)
    ax[0].imshow(scene)
    ax[1].imshow(mask) #, cmap = 'gray'
    ax[2].imshow(mask) #, cmap = 'gray'
    ax[2].imshow(difference, cmap = cmap)
    plt.show()

def show_bands_summary(scene_data, patch_id, cmap = 'viridis', bands = ['red', 'green', 'blue', 'nir'], figsize = (20,20), fontsize = 20, include_nir=True):
    cmap = matplotlib.cm.get_cmap(cmap)
    viridis_purple = cmap(0.0)
    viridis_yellow = cmap(1.0)

    n = len(bands)
    fig, ax = plt.subplots(n, n, figsize = figsize)

    #X labels on top
    for i in range(n):
        ax[0,i].set_xlabel(bands[i], fontsize=fontsize)
        ax[0,i].xaxis.set_label_position('top') 

    #Y labels on the left
    for i in range(n):
        ax[i,0].set_ylabel(bands[i], fontsize=fontsize)
        ax[i,0].yaxis.set_label_position('left') 

    #Scatter plots above diagonal
    for i in range(n):
        for j in range(i):
            ax[j,i].scatter(scene_data.open_as_points(patch_id, include_nir=include_nir)[:,j], scene_data.open_as_points(patch_id, include_nir=include_nir)[:,i], c=scene_data.open_mask_as_points(patch_id), alpha=0.3)
    
    #Density plots on the diagonal
    for i in range(n):
        scene_points = scene_data.open_as_points(patch_id, include_nir=include_nir)[:,i]
        scene_mask = scene_data.open_mask_as_points(patch_id)
        scene_points_cloud = scene_points[scene_mask != 0]
        scene_points_non_cloud = scene_points[scene_mask == 0]
        sns.kdeplot(scene_points_non_cloud, color=viridis_purple, ax=ax[i,i])
        sns.kdeplot(scene_points_cloud, color=viridis_yellow, ax=ax[i,i])
        if(i!=0):
            ax[i,i].set_xlabel('')
            ax[i,i].set_ylabel('')
    
    #Below diagonal Pearson correlation
    for i in range(n):
        for j in range(i):
            corr_coef = pearsonr(scene_data.open_as_points(patch_id, include_nir=include_nir)[:,i],scene_data.open_as_points(patch_id, include_nir=include_nir)[:,j])[0]   
            corr_coef = np.round(corr_coef, decimals=4)
            ax[i,j].text(0.5,0.5, 'Correlation\n'+str(corr_coef), fontsize = fontsize, horizontalalignment='center', verticalalignment='center')
    plt.show()