
import numpy as np

import glob

import pandas as pd

from random import shuffle

from skimage.morphology import binary_erosion

import utils



import time



def get_triplet_imgs(img_dir, img_ext='.svs', n_triplets=1000, sampling="SAME"):
    img_names = [file.replace("\\", "/") for file in glob.glob(img_dir+"**\**\**\**\**"+img_ext)]
    type_dict = {
        "LUAD":"Lung",
        "LUSC":"Lung",
        "MESO":"Lung",
        "ESCA":"Gastrointestinal",
        "STAD":"Gastrointestinal",
        "COAD":"Gastrointestinal",
        "READ":"Gastrointestinal",
        "PRAD":"Prostate",
        "TGCT":"Prostate"

    }
    df = pd.DataFrame({"path":img_names})
    type_list = [image_name.split("/")[-4] for image_name in img_names]
    subtype_list_interm = [image_name.split("_")[-1] for image_name in img_names]
    subtype_list = [type_name.split(".")[0] for type_name in subtype_list_interm]
    test_or_train = [image_name.split("/")[-3] for image_name in img_names]

    df.insert(1, "subtype", subtype_list, True)
    df.insert(2, "type", type_list, True)
    df.insert(3, "test_or_train", test_or_train, True)

    df.query('test_or_train == "Train"', inplace=True)

    del df['test_or_train']

    if sampling == "SAME":
        sample_1 = df["path"].sample(n=n_triplets, replace=True, random_state=42).tolist()
        img_triplets = [sample_1, sample_1]
    
    elif sampling == "SUBTYPE":
        sample_1 = []
        sample_2 = []
        
        for subtype in type_dict.keys():
            df_subtype = df[df["subtype"]==subtype]
            sample_1.extend(df_subtype["path"].sample(n=n_triplets, replace=True).tolist())
            sample_2.extend(df_subtype["path"].sample(n=n_triplets, replace=True).tolist())
        
        img_triplets = [sample_1, sample_2]
    
    elif sampling == "TYPE":
        sample_1 = []
        sample_2 = []
        
        for type_ in set(type_dict.values()):
            df_type = df[df["type"]==type_]
            sample_1.extend(df_type["path"].sample(n=n_triplets, replace=True).tolist())
            sample_2.extend(df_type["path"].sample(n=n_triplets, replace=True).tolist())
        
        c = list(zip(sample_1, sample_2))
        shuffle(c)
        sample_1, sample_2 = zip(*c)
        img_triplets = [sample_1[:n_triplets], sample_2[:n_triplets]]
    
    elif sampling == "ALL":
        sample_1 = df["path"].sample(n=n_triplets, replace=True).tolist()
        sample_2 = df["path"].sample(n=n_triplets, replace=True).tolist()
        
        c = list(zip(sample_1, sample_2))
        shuffle(c)
        sample_1, sample_2 = zip(*c)
        img_triplets = [sample_1[:n_triplets], sample_2[:n_triplets]]
        
        img_triplets = [sample_1, sample_2]
        
    img_triplets = np.array(img_triplets)
    
    return  img_triplets.T

def get_triplet_tiles(msk_triplets, tile_size=8, neighborhood=1024, verbose=True):
    n_triplets = msk_triplets.shape[0]
    unique_msks = np.unique(msk_triplets)
    
    img_1 = [(elem.replace("Mask", "Thumbnail")).replace(".npy",".png") for elem in msk_triplets[:,0]]
    img_2 = [(elem.replace("Mask", "Thumbnail")).replace(".npy",".png") for elem in msk_triplets[:,1]]
    tiles = pd.DataFrame({"img_1":img_1,
                          "img_2":img_2,
                          "anchor":[(0, 0)]*n_triplets,
                          "neighbor":[(0, 0)]*n_triplets,
                          "distant":[(0, 0)]*n_triplets})
    
    selem = np.ones((tile_size, tile_size))

    for msk_name in unique_msks:
        print("Sampling mask {}".format(msk_name))
        msk = np.load(msk_name)
        msk_eroded = binary_erosion(msk, selem)
        msk_shape = msk_eroded.shape
        for idx, row in enumerate(msk_triplets):
            if row[0] == msk_name:
                y_anchor, x_anchor = utils.sample_anchor(msk_eroded, tile_size)
                y_neighbor, x_neighbor = utils.sample_neighbor(msk_shape, x_anchor, y_anchor, tile_size)
                if verbose:
                    print("    Saving anchor and neighbor tile #{}".format(idx))
                    print("    Anchor tile center:{}".format((y_anchor, x_anchor)))
                    print("    Neighbor tile center:{}".format((y_neighbor, x_neighbor)))
                tiles["anchor"].iloc[idx] = (y_anchor, x_anchor)
                tiles["neighbor"].iloc[idx] = (y_neighbor, x_neighbor)
                if row[1] == msk_name:
                    y_distant, x_distant = utils.sample_distant_same(msk_eroded, x_anchor, y_anchor, neighborhood)
                    if verbose:
                        print("    Saving distant tile #{}".format(idx))
                        print("    Distant tile center:{}".format((y_distant, x_distant)))
                    tiles["distant"].iloc[idx] = (y_distant, x_distant)
            elif row[1] == msk_name:
                y_distant, x_distant = utils.sample_distant_diff(msk_eroded,tile_size)
                if verbose:
                    print("    Saving distant tile #{}".format(idx))
                    print("    Distant tile center:{}".format((y_distant, x_distant)))
                tiles["distant"].iloc[idx] = (y_distant, x_distant)
    return tiles

def main():
    with utils.Timer('saving_stuff'):
        masks = get_triplet_imgs("D://Datasets//TCGA", img_ext='.npy', sampling="ALL", n_triplets=75000)
        df = get_triplet_tiles(msk_triplets=masks, tile_size=64)
        file_name = str('dataframe-'+ time.strftime("%Y%m%d-%H%M%S")+'.csv')
        df.to_csv(file_name)
main()

