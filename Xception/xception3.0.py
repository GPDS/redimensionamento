import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf
import gc
from keras import layers, models
from keras.api.applications.xception import Xception, preprocess_input
from keras.api.applications.vgg16 import VGG16
from keras.api.utils import to_categorical
from keras.api.callbacks import CSVLogger
from keras.api.optimizers import Adam
from keras.api.models import Model
from keras.api.layers import Dense, GlobalAveragePooling2D, Input, Conv2D, MaxPooling2D, Flatten, BatchNormalization
import keras.api as keras
import config
from datetime import datetime
import matplotlib.pyplot as plt
import cv2
import copy

#weights = {
#  0: (2288)/1376,
#  1: (2288)/912,
#}

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Aloca toda a memória da GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, False)
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=11400)]
            )
    except RuntimeError as e:
        print(e)


def normalize_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # Converte a imagem para float32 e normaliza
    return image, label

code = config.combination
path_files = f'../new_redim_{config.res}x{config.res}/'

train_path = path_files + f'train_{code}/'
dataset = tf.keras.preprocessing.image_dataset_from_directory(
   train_path,
   #'../new_redim/train_categorical/',
   labels='inferred',
   label_mode = "int",
   #class_names = ['0','1','2','3','4'],
   color_mode='rgb',
   batch_size=config.BATCH_SIZE,
   image_size=(config.res,config.res),
   shuffle = True,
   #validation_split=None,
   seed = 2357,
   #subset='training'
)

#dataset = dataset.map(lambda x, y: (tf.keras.applications.xception.preprocess_input(x), y))
#dataset = dataset.map(lambda x, y: tf.py_function(func=gauss, inp=[x, y], Tout=[tf.float32, tf.int32]))
#dataset = tf.convert_to_tensor(dataset, tf.float32)
dataset = dataset.map(normalize_image)
dataset = dataset.map(lambda x, y: (x, tf.one_hot(y, depth=5)))


val_path = path_files + f'val_{code}/'
val = keras.preprocessing.image_dataset_from_directory(
   val_path,
   #'../new_redim/val_categorical/',
   labels='inferred',
   label_mode = "int",
   #class_names = ['0','1','2','3','4'],
   color_mode='rgb',
   batch_size=config.BATCH_SIZE,
   image_size=(config.res,config.res),
   shuffle = False,
   #validation_split=None,
   seed = 7,
   #subset='validation'
)
#val = val.map(lambda x, y: (tf.keras.applications.xception.preprocess_input(x), y))
val = val.map(normalize_image)
val = val.map(lambda x, y: (x, tf.one_hot(y, depth=5)))


# def normalize(x,y):
#   #image = x/255.
#   image = preprocess_input(x)
#   return image, y

# dataset = dataset.map(normalize)
# val = val.map(normalize)

for i in range(0,config.n):

  def clear_ram():
    gc.collect()
    tf.keras.backend.clear_session()

  clear_ram()

  #trainx = preprocess_input(trainx)

  ##################### 
  # ESTRUTURA DA REDE #
  #####################

  base_model = Xception(
      include_top=False, weights='imagenet',input_shape=(config.res,config.res,3),
  )

  x = base_model.output
  x = BatchNormalization()(x)
  x = keras.layers.GlobalAveragePooling2D()(x)
  output = keras.layers.Dense(config.NUM_CLASSES, activation='softmax')(x)

  model = keras.models.Model(inputs = base_model.input, outputs = output)
  
  # s1 = base_model.layers[-2].output
  # s1 = keras.layers.Dense(5, activation='softmax', name='predictions')(s1)


  # model = keras.models.Model(
  #     inputs = base_model.output,
  #     outputs = s1
  # )

  model.summary()

  #x = BatchNormalization()(x)
  #x = keras.layers.LayerNormalization()(x)
  #x = keras.layers.Flatten()(x)
  #x = keras.layers.GlobalAveragePooling2D()(base_model.output)
  #x = keras.layers.Flatten()(x)
  # x = Dense(1056, activation="relu")(x)
  # x = Dense(352, activation="relu")(x)
  # x = Dense(96, activation="relu")(x)
  #x = keras.layers.Dropout(config.DROPOUT)(x)
  #output = keras.layers.Dense(config.NUM_CLASSES, activation = 'softmax')(x)

  #for layer in model.layers:
  #  if isinstance(layer, BatchNormalization):
  #      layer.trainable = True

  #exit()

  model.compile(optimizer=keras.optimizers.Adam(),
                loss=keras.losses.BinaryCrossentropy(),  # Adjust loss function based on your task
                #loss=keras.losses.CategoricalCrossentropy(),  # Adjust loss function based on your task
                metrics = [keras.metrics.BinaryAccuracy(name = 'Acc')]
                #metrics = keras.metrics.CategoricalAccuracy(name = 'acc')
  )

  date = datetime.now().strftime("%Y%m%d%H%M%S")
  print(f'Código da Rede = {date}')

  csv_log_file = f'model/{date}.csv'
  #csv_log_file = 'training_log'+str(i+1)+'_XCEPTION_full_trainable_softmax_batchsize_'+str(config.BATCH_SIZE)+'_dropout_' + str(config.DROPOUT) + '_COLOR_NORMALIZATION_'+'selected_images'+'_' + '.csv'
  
  csv_logger = CSVLogger(csv_log_file)

  #######################
  # TREINAMENTO DA REDE #
  #######################

  history = model.fit(dataset,
                    epochs=config.EPOCHS,
                    validation_data=val,
                    callbacks=[csv_logger],
                    #steps_per_epoch = len(trainy)/24)
                    #class_weight=weights
                    )
  
  plt.figure(figsize=(10, 5))

# Gráfico de Acurácia
  plt.subplot(1, 2, 1)
  plt.plot(history.history['Acc'], label='Treinamento')
  plt.plot(history.history['val_Acc'], label='Validação')
  plt.title(f'Acurácia por Época')
  plt.xlabel('Épocas')
  plt.ylabel('Acurácia')
  plt.legend()

  # Gráfico de Perda
  plt.subplot(1, 2, 2)
  plt.plot(history.history['loss'], label='Treinamento')
  plt.plot(history.history['val_loss'], label='Validação')
  plt.title('Perda por Época')
  plt.xlabel('Épocas')
  plt.ylabel('Perda')
  plt.legend()

  # Salvando o gráfico
  plt.tight_layout()  # Ajusta o layout para evitar sobreposições
  plt.savefig(f'model/{date}_graph.png')
  
  model.save(f'model/{date}.keras')
  #model.save("modelo_half_trainable_selected_images_batchsize_"+str(config.BATCH_SIZE)+"_epoch_"+str(config.EPOCHS)+".keras")

  #y_test = model.predict(dataset)
  #loss, acc = model.evaluate(dataset)
  #print(acc)
  clear_ram()