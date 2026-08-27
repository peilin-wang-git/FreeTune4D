"""
data generators for VoxelMorph

for the CVPR and MICCAI papers, we have data arranged in train/validate/test folders
inside each folder is a /vols/ and a /asegs/ folder with the volumes
and segmentations. All of our papers use npz formated data.
"""

import os, sys
import numpy as np
import scipy.io as sio


def cvpr2018_gen(gen, atlas_vol_bs, batch_size=1):
    """ generator used for cvpr 2018 model """

    volshape = atlas_vol_bs.shape[1:-1]
    zeros = np.zeros((batch_size, *volshape, len(volshape)))
    while True:
        X = next(gen)[0]
        yield ([X, atlas_vol_bs], [atlas_vol_bs, zeros])


def cvpr2018_gen_s2s(gen, batch_size=1):
    """ generator used for cvpr 2018 model for subject 2 subject registration """
    zeros = None
    while True:
        X1 = next(gen)[0]
        X2 = next(gen)[0]

        if zeros is None:
            volshape = X1.shape[1:-1]
            zeros = np.zeros((batch_size, *volshape, len(volshape)))
        yield ([X1, X2], [X2, zeros])


def miccai2018_gen(gen, atlas_vol_bs, batch_size=1, bidir=False):
    """ generator used for miccai 2018 model """
    volshape = atlas_vol_bs.shape[1:-1]
    zeros = np.zeros((batch_size, *volshape, len(volshape)))
    while True:
        # batch-size number of training examples.
        X = next(gen)[0]
        if bidir:
            yield ([X, atlas_vol_bs], [atlas_vol_bs, X, zeros])
        else:
            yield ([X, atlas_vol_bs], [atlas_vol_bs, zeros])


def miccai2018_gen_s2s(gen, batch_size=1, bidir=False):
    """ generator used for miccai 2018 model """
    zeros = None
    while True:
        X = next(gen)[0]
        Y = next(gen)[0]
        if zeros is None:
            volshape = X.shape[1:-1]
            zeros = np.zeros((batch_size, *volshape, len(volshape)))
        if bidir:
            yield ([X, Y], [Y, X, zeros])
        else:
            yield ([X, Y], [Y, zeros])
            
def gen_trainHX(gen, atlas_vol_bs, batch_size=1):
    """
    Generator used for 4D Lung training.
    """
    volshape = atlas_vol_bs.shape[1:-1]
    zeros = np.zeros((batch_size, *volshape, len(volshape)))
    while True:
        X = next(gen)[0]
        yield ([X, atlas_vol_bs], [atlas_vol_bs, zeros])
    

def example_gen_HX(cwd, studies, volshape = (256,256,64), batch_size=1, np_var='volume'):
    """
    Gerenrate examples, randomly pair fixed and moving volume from the 10 phases
    of 20 patients. Theratically, there is supposed to be A(10,2) = 90 pairs for
    every study and 1800 pairs in total. (Assuming 1 4D scan for 1 patient)
    Options in the future is to randomly pair all the volumes. Much more samples.
    Every study should contain 10 .npy files for 10 volumes.
    
    Parameters:
        cwd: root folder (abs path)
        studies: folders containing all 4D scans.
        batch_size: the size of mini-batch. Depending the # of GPUs. 
        return_segs: whether to return segmentation. Useless.
        seg_dir: useless.
        np_var: specify the name of the variable in numpy files, if your data is stored in 
            npz files. default to 'vol_data'
    """
    zeros = np.zeros((batch_size, *volshape, len(volshape)))
    while True:
        # Randomly choose a study
        idx_study = np.random.randint(len(studies), size=batch_size)
        X_data = []
        Y_data = []
        # Load number of batch_size volumes
        for idx_s in idx_study:
            # Choose 2 phase volumes from the study, one for moving and
            # one for fixed
            idx_phase = np.random.randint(10, size=2)
            phases = os.listdir(os.path.join(cwd, '4D-Lung-mask-npy', studies[idx_s]))
            volume_X_path = os.path.join(cwd, '4D-Lung-mask-npy', studies[idx_s], phases[idx_phase[0]])
            # X = sio.loadmat(volume_X_path)[np_var]
            # To fit the 5D input, add two dimensions which are 1s
            X = np.load(volume_X_path)
            X = X
            X = X[np.newaxis, ..., np.newaxis]
            X_data.append(X)
            
            volume_Y_path = os.path.join(cwd, '4D-Lung-mask-npy', studies[idx_s], phases[idx_phase[1]])
            # Y = sio.loadmat(volume_Y_path)[np_var]
            Y = np.load(volume_Y_path)
            Y = Y
            Y = Y[np.newaxis, ..., np.newaxis]
            Y_data.append(Y)

                
        if batch_size > 1:
            return_X = np.concatenate(X_data, 0)
            return_Y = np.concatenate(Y_data, 0)
        else:
            return_X = X_data[0]
            return_Y = Y_data[0]
        
            
        yield ([return_X, return_Y], [return_Y, zeros])
        
def example_genMRI_HX(cwd, studies, volshape = (256,256,64), batch_size=1, np_var='volume'):
    """
    Gerenrate examples, randomly pair fixed and moving volume from the 10 phases
    of 20 patients. Theratically, there is supposed to be A(10,2) = 90 pairs for
    every study and 1800 pairs in total. (Assuming 1 4D scan for 1 patient)
    Options in the future is to randomly pair all the volumes. Much more samples.
    Every study should contain 10 .npy files for 10 volumes.
    
    Parameters:
        cwd: root folder (abs path)
        studies: folders containing all 4D scans.
        batch_size: the size of mini-batch. Depending the # of GPUs. 
        return_segs: whether to return segmentation. Useless.
        seg_dir: useless.
        np_var: specify the name of the variable in numpy files, if your data is stored in 
            npz files. default to 'vol_data'
    """
    zeros = np.zeros((batch_size, *volshape, len(volshape)))
    while True:
        # Randomly choose a study
        idx_study = np.random.randint(len(studies), size=batch_size)
        X_data = []
        Y_data = []
        # Load number of batch_size volumes
        for idx_s in idx_study:
            # Choose 2 phase volumes from the study, one for moving and
            # one for fixed
            phases = os.listdir(os.path.join(cwd, '4D-MRI-N4-mask-nor-mat', studies[idx_s]))
            idx_phase = np.random.randint(len(phases), size=2)
            volume_X_path = os.path.join(cwd, '4D-MRI-N4-mask-nor-mat', studies[idx_s], phases[idx_phase[0]])
            X = sio.loadmat(volume_X_path)[np_var]
            # To fit the 5D input, add two dimensions which are 1s
            X = X[np.newaxis, ..., np.newaxis]
            X_data.append(X)
            
            volume_Y_path = os.path.join(cwd, '4D-MRI-N4-mask-nor-mat', studies[idx_s], phases[idx_phase[1]])
            Y = sio.loadmat(volume_Y_path)[np_var]
            Y = Y[np.newaxis, ..., np.newaxis]
            Y_data.append(Y)

                
        if batch_size > 1:
            return_X = np.concatenate(X_data, 0)
            return_Y = np.concatenate(Y_data, 0)
        else:
            return_X = X_data[0]
            return_Y = Y_data[0]
        
        yield ([return_X, return_Y], [return_Y, zeros])
        
def example_gen_cheat(cwd, studies, volshape = (256,256,64), batch_size=1, np_var='volume'):
    """
    Gerenrate examples, randomly pair fixed and moving volume from the 10 phases
    of 20 patients. Theratically, there is supposed to be A(10,2) = 90 pairs for
    every study and 1800 pairs in total. (Assuming 1 4D scan for 1 patient)
    Options in the future is to randomly pair all the volumes. Much more samples.
    Every study should contain 10 .npy files for 10 volumes.
    
    Parameters:
        cwd: root folder (abs path)
        studies: folders containing all 4D scans.
        batch_size: the size of mini-batch. Depending the # of GPUs. 
        return_segs: whether to return segmentation. Useless.
        seg_dir: useless.
        np_var: specify the name of the variable in numpy files, if your data is stored in 
            npz files. default to 'vol_data'
    """
    zeros = np.zeros((batch_size, *volshape, len(volshape)))
    while True:
        
        X_data = []
        Y_data = []
        # Load number of batch_size volumes
        X = sio.loadmat('CT2_Train.mat')['CT1']
        # To fit the 5D input, add two dimensions which are 1s
        X = X[np.newaxis, ..., np.newaxis]
        X_data.append(X)
            
        Y = sio.loadmat('CT2_Train.mat')['CT2']
        Y = Y[np.newaxis, ..., np.newaxis]
        Y_data.append(Y)
                
        if batch_size > 1:
            return_X = np.concatenate(X_data, 0)
            return_Y = np.concatenate(Y_data, 0)
        else:
            return_X = X_data[0]
            return_Y = Y_data[0]
        
        yield ([return_X, return_Y], [return_Y, zeros])

def example_gen(vol_names, batch_size=1, return_segs=False, seg_dir=None, np_var='vol_data'):
    """
    generate examples

    Parameters:
        vol_names: a list or tuple of filenames
        batch_size: the size of the batch (default: 1)

        The following are fairly specific to our data structure, please change to your own
        return_segs: logical on whether to return segmentations
        seg_dir: the segmentations directory.
        np_var: specify the name of the variable in numpy files, if your data is stored in 
            npz files. default to 'vol_data'
    """

    while True:
        # get a random batch size from the data source
        idxes = np.random.randint(len(vol_names), size=batch_size)

        X_data = []
        for idx in idxes:
            X = load_volfile(vol_names[idx], np_var=np_var)
            # To fit the 5D input, add two dimensions which are 1
            X = X[np.newaxis, ..., np.newaxis]
            X_data.append(X)

        if batch_size > 1:
            return_vals = [np.concatenate(X_data, 0)]
        else:
            return_vals = [X_data[0]]

        # also return segmentations
        if return_segs:
            X_data = []
            for idx in idxes:
                X_seg = load_volfile(vol_names[idx].replace('norm', 'aseg'), np_var=np_var)
                X_seg = X_seg[np.newaxis, ..., np.newaxis]
                X_data.append(X_seg)
            
            if batch_size > 1:
                return_vals.append(np.concatenate(X_data, 0))
            else:
                return_vals.append(X_data[0])

        yield tuple(return_vals)


def load_example_by_name(vol_name, seg_name, np_var='vol_data'):
    """
    load a specific volume and segmentation

    np_var: specify the name of the variable in numpy files, if your data is stored in 
        npz files. default to 'vol_data'
    """
    X = load_volfile(vol_name, np_var)
    X = X[np.newaxis, ..., np.newaxis]

    return_vals = [X]

    X_seg = load_volfile(seg_name, np_var)
    X_seg = X_seg[np.newaxis, ..., np.newaxis]

    return_vals.append(X_seg)

    return tuple(return_vals)


def load_volfile(datafile, np_var='vol_data'):
    """
    load volume file
    formats: nii, nii.gz, mgz, npz
    if it's a npz (compressed numpy), variable names innp_var (default: 'vol_data')
    """
    assert datafile.endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file'

    if datafile.endswith(('.nii', '.nii.gz', '.mgz')):
        # import nibabel
        if 'nibabel' not in sys.modules:
            try :
                import nibabel as nib  
            except:
                print('Failed to import nibabel. need nibabel library for these data file types.')

        X = nib.load(datafile).get_data()
        
    else: # npz
        if np_var is None:
            np_var = 'vol_data'
        X = np.load(datafile)[np_var]

    return X
