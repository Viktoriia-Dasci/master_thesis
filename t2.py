import numpy as np
import os
import random
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import EfficientNetB0
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.models import Sequential
import tensorflow_addons as tfa
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow_addons import metrics
import keras_tuner
from kerastuner.tuners import Hyperband
from kerastuner.engine.hyperparameters import HyperParameters
from sklearn.metrics import roc_curve, auc
import cv2

#custom functions
from Model_functions import *

home_dir = '/home/viktoriia.trokhova/'

base_dir = '/home/viktoriia.trokhova/Split_data/'

modality = 't2'

#load data
HGG_list_train = load_from_dir(f'/content/drive/MyDrive/Split_data/{modality}_mri_slices/train/HGG_{modality}')
LGG_list_train = load_from_dir(f'/content/drive/MyDrive/Split_data/{modality}_mri_slices/train/LGG_{modality}')
HGG_list_val = load_from_dir(f'/content/drive/MyDrive/Split_data/{modality}_mri_slices/val/HGG_{modality}')
LGG_list_val = load_from_dir(f'/content/drive/MyDrive/Split_data/{modality}_mri_slices/val/LGG_{modality}')
HGG_list_test = load_from_dir(f'/content/drive/MyDrive/Split_data/{modality}_mri_slices/test/HGG_{modality}')
LGG_list_test = load_from_dir(f'/content/drive/MyDrive/Split_data/{modality}_mri_slices/test/LGG_{modality}')


#preprocessing data
HGG_list_new_train = preprocess(HGG_list_train)
LGG_list_new_train = preprocess(LGG_list_train)

HGG_list_new_val = preprocess(HGG_list_val)
LGG_list_new_val = preprocess(LGG_list_val)

HGG_list_new_test = preprocess(HGG_list_test)
LGG_list_new_test = preprocess(LGG_list_test)

# Combining the HGG and LGG lists
X_train, y_train = add_labels([], [], HGG_list_new_train, label='HGG')
X_train, y_train = add_labels(X_train, y_train, LGG_list_new_train, label='LGG')

X_val, y_val = add_labels([], [], HGG_list_new_val, label='HGG')
X_val, y_val = add_labels(X_val, y_val, LGG_list_new_val, label='LGG')

X_test, y_test = add_labels([], [], HGG_list_new_test, label='HGG')
X_test, y_test = add_labels(X_test, y_test, LGG_list_new_test, label='LGG')

# Converting labels to numerical values and one-hot encoding
labels = {'HGG': 0, 'LGG': 1}
y_train = tf.keras.utils.to_categorical([labels[y] for y in y_train])
y_val = tf.keras.utils.to_categorical([labels[y] for y in y_val])
y_test = tf.keras.utils.to_categorical([labels[y] for y in y_test])

# Converting data to arrays and shuffle
X_val, y_val = shuffle(np.array(X_val), y_val, random_state=101)
X_train, y_train = shuffle(np.array(X_train), y_train, random_state=101)
X_test, y_test = shuffle(np.array(X_test), y_test, random_state=101)

#Calculating class_weights
class_weights = generate_class_weights(y_train, multi_class=False, one_hot_encoded=True)
print(class_weights)

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)


#Data augmentation
datagen = ImageDataGenerator(
    rotation_range=90,
    vertical_flip=True,
    horizontal_flip=True,
    fill_mode='nearest')
    
train_generator = datagen.flow(
    X_train, y_train,
    shuffle=True)

#Hyperparameter tuning
hp = HyperParameters()

tuner_effnet = Hyperband(
    model_effnet,
    objective=keras_tuner.Objective("val_f1_score", direction="max"),
    overwrite=True,
    max_epochs=30,
    factor=3,
    hyperband_iterations=5
)

tuner_densenet = Hyperband(
    model_densenet,
    objective=keras_tuner.Objective("val_f1_score", direction="max"),
    overwrite=True,
    max_epochs=30,
    factor=3,
    hyperband_iterations=5
)

tuner_inception = Hyperband(
    model_inception,
    objective=keras_tuner.Objective("val_f1_score", direction="max"),
    overwrite=True,
    max_epochs=30,
    factor=3,
    hyperband_iterations=5
)

# Searching for best hyperparameters and models for each tuner
best_hyperparameters = {}
best_models = {}
tuners = {'EffNet': tuner_effnet, 'DenseNet': tuner_densenet, 'Inception': tuner_inception}

for name, tuner in tuners.items():
    tuner.search(train_generator,
                 validation_data=(X_val, y_val),
                 steps_per_epoch=len(train_generator),
                 epochs=50,
                 verbose=1
                 )
    
    best_hyperparameters[name] = tuner.get_best_hyperparameters(1)[0]
    best_models[name] = tuner.get_best_models(1)[0]

# Define callbacks
early_stop = EarlyStopping(monitor='val_f1_score', mode='max', patience=10, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor = 'val_f1_score', factor = 0.3, patience = 5, min_delta = 0.001, mode='max',verbose=1)

# Define the path for saving the plots
plot_folder_path = os.path.join(home_dir, f"model_plots/{modality}") 

# Fit the best model from each tuner to the training data for 50 epochs using the best hyperparameters
for name, model in best_models.items():

    checkpoint = ModelCheckpoint(f"{home_dir}/model_weights/model_tuned_{name}.h5", monitor='val_f1_score',save_best_only=True, mode="max",verbose=1)

    print(f'Training {name}...')
    history = model.fit(
        train_generator,
        epochs=50,
        validation_data=(X_val, y_val),
        verbose=1,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )

    # After training, plot the accuracy, loss, and f1 score
    plot_acc_loss_f1(history, plot_folder_path, name)

#Training models with the best hyperparameters inputted manually
# history_inception_weights = model_train(model_name = tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=(224,224,3), classes=2), save_name = f"inception_{modality}", image_size = 224, dropout=0.5, optimizer = tf.keras.optimizers.SGD(learning_rate=0.01), dense_0_units=80, dense_1_units=48, batch_size=16)
# history_effnet = model_train(model_name = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224,3)), save_name = f"effnet_{modality}", image_size = 224, dropout=0.7, optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), dense_0_units=96, batch_size=32)
# history_densenet_weights = model_train(model_name = tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_shape=(224,224,3), classes=2), save_name = f"densenet_{modality}", image_size = 224, dropout=0.5, optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), dense_0_units=32, batch_size=32)
# plot_acc_loss_f1(history_inception_weights,  plot_folder_path, 'inception')  
# plot_acc_loss_f1(history_densenet_weights,  plot_folder_path, 'densenet') 
# plot_acc_loss_f1(history_effnet,  plot_folder_path, 'effnet')


#Explainability analysis
pretrained_model = load_model('/content/drive/MyDrive/final_models/densenet_t2.h5', custom_objects={'f1_score': f1_score, 'focal_loss': focal_loss})

modality = 'T2'

pred_res = pretrained_model.predict(X_test)
pred_ready_res = np.argmax(pred_res,axis=1)
y_test_new_res = np.argmax(y_test,axis=1)

# Load the models
densenet = load_model('/content/drive/MyDrive/final_models/densenet_t2.h5', custom_objects={'f1_score': f1_score, 'focal_loss': focal_loss})
effnet = load_model('/content/drive/MyDrive/final_models/effnet_t2.h5', custom_objects={'f1_score': f1_score, 'focal_loss': focal_loss})
inception = load_model('/content/drive/MyDrive/final_models/inception_t2.h5', custom_objects={'f1_score': f1_score, 'focal_loss': focal_loss})

models = {'DenseNet-121': densenet, 'EfficientNetB0': effnet, 'Inception-V3': inception}

plt.figure(figsize=[5,5])

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12

for name, model in models.items():
    pred = model.predict(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test[:, 1], pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (name, roc_auc))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

print(classification_report(y_test_new_res, pred_ready_res, target_names=['HGG', 'LGG']))

colors_dark = ["#1F1F1F", "#313131", '#636363', '#AEAEAE', '#DADADA']
colors_red = ["#331313", "#582626", '#9E1717', '#D35151', '#E9B4B4']
colors_green = ['#01411C','#4B6F44','#4F7942','#74C365','#D0F0C0']

fig,ax=plt.subplots(1,1,figsize=(14,7))
sns.heatmap(confusion_matrix(y_test_new_res,pred_ready_res),ax=ax,xticklabels=labels,yticklabels=labels,annot=True, fmt='g',
           cmap=colors_green[::-1],alpha=0.7,linewidths=2,linecolor=colors_dark[3])
fig.text(s='Heatmap of the Confusion Matrix',size=18,fontweight='bold',
             fontname='monospace',color=colors_dark[1],y=0.92,x=0.28,alpha=0.8)

plt.show()


@tf.custom_gradient
def guidedRelu(x):
    def grad(dy):
        return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy
    return tf.nn.relu(x), grad

# Reference: https://github.com/eclique/keras-gradcam with adaption to tensorflow 2.0
class GuidedBackprop:
    def __init__(self,model, layerName=None):
        self.model = model
        self.layerName = layerName
        if self.layerName == None:
            self.layerName = self.find_target_layer()

        self.gbModel = self.build_guided_model()

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply Guided Backpropagation")

    def build_guided_model(self):
        gbModel = Model(
            inputs = [self.model.inputs],
            outputs = [self.model.get_layer(self.layerName).output]
        )
        layer_dict = [layer for layer in gbModel.layers[1:] if hasattr(layer,"activation")]
        for layer in layer_dict:
            if layer.activation == tf.keras.activations.relu:
                layer.activation = guidedRelu

        return gbModel

    def guided_backprop(self, images, upsample_size):
        """Guided Backpropagation method for visualizing input saliency."""
        with tf.GradientTape() as tape:
            inputs = tf.cast(images, tf.float32)
            tape.watch(inputs)
            outputs = self.gbModel(inputs)

        grads = tape.gradient(outputs, inputs)[0]

        saliency = cv2.resize(np.asarray(grads), upsample_size)

        return saliency


def deprocess_image(x):
    """Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    """
    # normalize tensor: center on 0., ensure std is 0.25
    x = x.copy()
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

class GradCAM:
    def __init__(self, model, layerName=None):
        """
        model: pre-softmax layer (logit layer)
        """
        self.model = model
        self.layerName = layerName

        if self.layerName == None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM")

    def compute_heatmap(self, image, upsample_size, classIdx=None, eps=1e-5):
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output, self.model.output]
        )
        # record operations for automatic differentiation

        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOuts, preds) = gradModel(inputs)  # preds after softmax
            if classIdx is None:
                classIdx = np.argmax(preds)
            loss = preds[:, classIdx]

        # compute gradients with automatic differentiation
        grads = tape.gradient(loss, convOuts)
        # discard batch
        convOuts = convOuts[0]
        grads = grads[0]
        norm_grads = tf.divide(grads, tf.reduce_mean(tf.square(grads)) + tf.constant(eps))

        # compute weights
        weights = tf.reduce_mean(norm_grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOuts), axis=-1)

        # Apply reLU
        cam = np.maximum(cam, 0)
        cam = cam / np.max(cam)
        cam = cv2.resize(cam, upsample_size,cv2.INTER_LINEAR)

        # convert to 3D
        cam3 = np.expand_dims(cam, axis=2)
        cam3 = np.tile(cam3, [1, 1, 3])

        return cam3


def overlay_gradCAM(img, cam3):
    cam3 = np.uint8(255 * cam3)
    cam3 = cv2.applyColorMap(cam3, cv2.COLORMAP_HOT)

    new_img = 0.6 * cam3 + 0.2 * img

    return (new_img * 255.0 / new_img.max()).astype("uint8")
