# some training parameters
import numpy as np

EPOCHS = 50
BATCH_SIZE = 40
NUM_CLASSES = 5
DROPOUT = 0.5
n = 1
res = 224
combination = '0-1-2-3-4'
image_size = (res,res,3)


# trainy = np.load('../cut_trainy_random_augmentation.npy')
# trainy = np.array(to_categorical(trainy), dtype = 'uint8')

# valy = np.load('../cut_valy_random.npy')
# valy = np.array(to_categorical(valy), dtype = 'uint8')

# valx = np.memmap('../cut_valx_COLOR_NORMALIZATION_NORMALIZED_random.dat',
#                 dtype='float32',
#                 mode='r',
#                 shape=(valy.shape[0],224,224,3))

# #valx = preprocess_input(valx)

# trainx = np.memmap('../cut_trainx_COLOR_NORMALIZATION_NORMALIZED_random_augmentation.dat',
#                 dtype='float32',
#                 mode='r',
#                 shape=(trainy.shape[0],224,224,3))
