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
from lime import lime_image
from skimage.segmentation import mark_boundaries
import PIL.Image
from matplotlib import pylab as P
import saliency.core as saliency
from skimage import img_as_float
import matplotlib.pyplot as plt
import pandas as pd
import re


#custom functions
from Model_functions import *
from explainabilty_functions import *

home_dir = '/home/viktoriia.trokhova/'

base_dir = '/home/viktoriia.trokhova/Split_data/'

modality = 't2'

#load data
HGG_list_train = load_from_dir(f'{base_dir}/{modality}_mri_slices/train/HGG_{modality}')
LGG_list_train = load_from_dir(f'{base_dir}/{modality}_mri_slices/train/LGG_{modality}')
HGG_list_val = load_from_dir(f'{base_dir}/{modality}_mri_slices/val/HGG_{modality}')
LGG_list_val = load_from_dir(f'{base_dir}/{modality}_mri_slices/val/LGG_{modality}')
HGG_list_test = load_from_dir(f'{base_dir}/{modality}_mri_slices/test/HGG_{modality}')
LGG_list_test = load_from_dir(f'{base_dir}/{modality}_mri_slices/test/LGG_{modality}')


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

drive = '/content/drive/MyDrive'
#change for other modalities
modality = 't2'

#change when analysing exolainability methods for effnet and inception
model = 'densenet'

pretrained_model = load_model(f'{drive}/final_models/{model}_{modality}.h5', custom_objects={'f1_score': f1_score, 'focal_loss': focal_loss})

# Load the models
densenet = load_model(f'{drive}/final_models/densenet_{modality}.h5', custom_objects={'f1_score': f1_score, 'focal_loss': focal_loss})
effnet = load_model(f'{drive}/final_models/effnet_{modality}.h5', custom_objects={'f1_score': f1_score, 'focal_loss': focal_loss})
inception = load_model(f'{drive}/final_models/inception_{modality}.h5', custom_objects={'f1_score': f1_score, 'focal_loss': focal_loss})

#Plotting ROC curves
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

pred_res = pretrained_model.predict(X_test)
pred_ready_res = np.argmax(pred_res,axis=1)
y_test_new_res = np.argmax(y_test,axis=1)

print(classification_report(y_test_new_res, pred_ready_res, target_names=['HGG', 'LGG']))

#Confusion Matrix
colors_dark = ["#1F1F1F", "#313131", '#636363', '#AEAEAE', '#DADADA']
colors_red = ["#331313", "#582626", '#9E1717', '#D35151', '#E9B4B4']
colors_green = ['#01411C','#4B6F44','#4F7942','#74C365','#D0F0C0']

fig,ax=plt.subplots(1,1,figsize=(14,7))
sns.heatmap(confusion_matrix(y_test_new_res,pred_ready_res),ax=ax,xticklabels=labels,yticklabels=labels,annot=True, fmt='g',
           cmap=colors_green[::-1],alpha=0.7,linewidths=2,linecolor=colors_dark[3])
fig.text(s='Heatmap of the Confusion Matrix',size=18,fontweight='bold',
             fontname='monospace',color=colors_dark[1],y=0.92,x=0.28,alpha=0.8)

plt.show()


#Grad-CAM & Guided Grad-CAM
# Setting image, mask, and patient numbers
img_num = 100
msk_num = 100
pat_num = 17

# Setting the image class
img_class = 'HGG'

# Loading image data from a numpy file
img_arr = np.load(f"{drive}/{modality}_mri_slices/test/{img_class}_{modality}/{img_num}_{pat_num}.npy")

# Setting the mask class
img_msk_class = 'HGG'

# Loading mask data from a numpy file
img_msk = np.load(f"{drive}/{modality}_mri_slices/test/{img_msk_class}_masks/{msk_num}_{pat_num}.npy")

# Converting the grayscale image to RGB and cropping to desired size
img_rgb = cv2.cvtColor(img_arr.astype('float32'), cv2.COLOR_GRAY2RGB)
img_rgb = tf.image.crop_to_bounding_box(img_rgb, 8, 8, 224, 224)
img_rgb = img_rgb.numpy()

# Clipping and casting the mask values
msk_rgb = np.clip(img_msk, 0, 255).astype('float32')

# Displaying and saving the mask image
msk=plt.imshow(msk_rgb)
plt.axis("off")
save_results_to = '/content/' + str(msk_num)
plt.savefig(save_results_to, bbox_inches='tight', pad_inches=0)
msk=cv2.imread('/content/' + str(msk_num) + '.png')
msk_rgb=cv2.resize(msk, (224, 224))

# Printing the maximum pixel value of the mask
print(msk_rgb.max())

# Setting the target size for the GradCAM computations
target_size = (224,224)

# Computing the GradCAM heatmaps
gradCAM = GradCAM(model=pretrained_model)
cam3 = gradCAM.compute_heatmap(image=np.expand_dims(img_rgb,axis=0),classIdx=0,upsample_size=target_size)

# Normalizing and scaling the image data
img_rgb_plot = cv2.normalize(img_rgb, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
img_rgb_plot = img_rgb_plot *255

# Overlaying the GradCAM heatmap onto the image
heatmap = overlay_gradCAM(img_rgb_plot,cam3)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

# Computing the Guided Backpropagation
guidedBP = GuidedBackprop(model=pretrained_model)
gb_cam = guidedBP.guided_backprop(np.expand_dims(img_rgb,axis=0),target_size)
guided_gradcam = deprocess_image(gb_cam*cam3)

# Overlaying the GradCAM heatmap onto the mask
heatmap_mask = overlay_gradCAM(msk_rgb,cam3)

# Predicting the class of the test image
test_img_input = np.expand_dims(np.float32(img_rgb), axis=0)
test_prediction = pretrained_model.predict(test_img_input)
test_prediction_argmax=np.argmax(test_prediction, axis=1)

# Decoding the prediction to a label name
labels = ['HGG', 'LGG']
name = labels[int(test_prediction_argmax)]

# Printing the predicted and actual labels
print('predicted label:' + ' ' + name)
print('actual label:' + ' ' + img_class)

# Displaying the original image, mask, GradCAM, GradCAM Mask, and Guided GradCAM
fig, ax = plt.subplots(1,5,figsize=(16,32))
ax[0].imshow(rgb2gray(img_rgb_plot))
ax[0].axis("off")
ax[0].set_title(img_class + ' ' +  modality + ' ' + "Original Image ")
ax[1].imshow(rgb2gray(msk_rgb))
ax[1].axis("off")
ax[1].set_title(img_class + " Mask")
ax[2].imshow(rgb2gray(heatmap))
ax[2].axis("off")
ax[2].set_title("GradCAM")
ax[3].imshow(rgb2gray(heatmap_mask))
ax[3].axis("off")
ax[3].set_title("GradCAM Mask")
ax[4].imshow(guided_gradcam)
ax[4].axis("off")
ax[4].set_title("Guided GradCAM")
plt.show()

#Implementing LIME

# Convert the grayscale image to RGB and crop to desired size
img_col = cv2.cvtColor(img_arr.astype('float32'), cv2.COLOR_GRAY2RGB)
img_rgb = tf.image.crop_to_bounding_box(img_col, 8, 8, 224, 224)
img_rgb = img_rgb.numpy()

# Preprocess the image
img_rgb = tf.keras.applications.imagenet_utils.preprocess_input(img_rgb, mode='tf')

# Initialize LIME Image Explainer
explainer = lime_image.LimeImageExplainer()

# Explain the image prediction
explanation = explainer.explain_instance(img_rgb,
                                         pretrained_model.predict,
                                         top_labels=1,
                                         hide_color=0,
                                         num_samples=3000)

# Generate heatmap from explanation
heatmap = explanation_heatmap(explanation, explanation.top_labels[0])

# Get the image and mask for the top labels
temp_1, mask_1 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=1, hide_rest=True)
temp_2, mask_2 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=1, hide_rest=False)

# Prepare images for plotting
img_hide = mark_boundaries(temp_1, mask_1)
img_show = mark_boundaries(temp_2, mask_2)
img_hide = img_hide/np.amax(img_hide)
img_hide = np.clip(img_hide, 0, 1)
img_show = img_show/np.amax(img_show)
img_show = np.clip(img_show, 0, 1)

# Plot the images
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15,15))
ax1.imshow(rgb2gray(img_show))
ax2.imshow(rgb2gray(msk_rgb))
ax3.imshow(rgb2gray(img_hide))
img4 = ax4.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
ax1.axis('off')
ax1.set_title(str(img_class) + ' ' + modality + ' ' + 'LIME Image Overlap')
ax2.axis('off')
ax2.set_title(str(img_class) + ' ' + "Mask")
ax3.axis('off')
ax3.set_title("Top LIME Superpixel")
ax4.axis('off')
ax4.set_title("LIME Heatmap")
colorbar(img4)


# XRAI

conv_layer = pretrained_model.get_layer('conv5_block16_2_conv')

model = tf.keras.models.Model([pretrained_model.inputs], [conv_layer.output, pretrained_model.output])

class_idx_str = 'class_idx_str'

_, predictions = model(np.array([img_rgb]))
prediction_class = np.argmax(predictions[0])
call_model_args = {class_idx_str: prediction_class}

print("Prediction class: " + str(prediction_class))

# Convert grayscale image to RGB and resize to desired size
img_arr = cv2.cvtColor(img_arr.astype('float32'), cv2.COLOR_GRAY2RGB)
img_rgb = cv2.resize(img_arr, (224, 224))

# Convert image to floating point numbers in range [0,1]
img_rgb = img_as_float(img_rgb)

# Construct the saliency object. This alone doesn't do anything.
xrai_object = saliency.XRAI()

# Compute XRAI attributions with default parameters
xrai_attributions = xrai_object.GetMask(img_rgb, call_model_function, call_model_args, batch_size=20)

# Set up matplot lib figures.
ROWS = 1
COLS = 4
UPSCALE_FACTOR = 20
P.figure(figsize=(ROWS * UPSCALE_FACTOR, COLS * UPSCALE_FACTOR))

# Show original image
ShowImage(rgb2gray(img_rgb), title=img_class + ' ' +  modality + ' ' + "Original Image ", ax=P.subplot(ROWS, COLS, 1))

# Show image mask
ShowImage(rgb2gray(msk_rgb), title=str(img_class) + ' ' + "Mask", ax=P.subplot(ROWS, COLS, 2))

# Show most salient 7% of the image
mask = xrai_attributions >= np.percentile(xrai_attributions, 93)
im_mask = np.array(img_rgb)
im_mask[~mask] = 0
ShowImage(rgb2gray(im_mask), title='Top 7% Superpixels', ax=P.subplot(ROWS, COLS, 3))

# Show XRAI attributions heatmap
ShowHeatMap(xrai_attributions, title='XRAI Heatmap', ax=P.subplot(ROWS, COLS, 4))

#creating lists to store dice and iou coefficeints
rows_list = []
ggcam_list = []
xrai_list = []
lime_list = []

df_grad_LGG = gradcam_coef('LGG', drive, modality, pretrained_model, stacked=False)
df_grad_HGG = gradcam_coef('HGG', drive, modality, pretrained_model, stacked=False)
df_grad_HGG.to_csv(f"{home_dir}/explain_datasets/grad_coef_{model}_{modality}.csv", index=False)

df_ggrad_LGG = guided_grad_coef('LGG', drive, modality, pretrained_model, stacked=False)
df_ggrad_HGG = guided_grad_coef('HGG', drive, modality, pretrained_model, stacked=False)
df_ggrad_HGG.to_csv(f"{home_dir}/explain_datasets/guidgrad_coef_{model}_{modality}.csv", index=False)

df_xrai_LGG = xrai_coef('LGG', drive, modality, pretrained_model, stacked=False)
df_xrai_HGG = xrai_coef('HGG', drive, modality, pretrained_model, stacked=False)
df_xrai_HGG.to_csv(f"{home_dir}/explain_datasets/xrai_coef_{model}_{modality}.csv", index=False)

df_lime_LGG = lime_coef('LGG', drive, modality, pretrained_model, stacked=False)
df_lime_HGG = lime_coef('HGG', drive, modality, pretrained_model, stacked=False)
df_lime_HGG.to_csv(f"{home_dir}/explain_datasets/lime_coef_{model}_{modality}.csv", index=False)

