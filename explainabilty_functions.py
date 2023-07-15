import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import seaborn as sns
from scipy.ndimage import zoom
from skimage.color import rgb2gray
from lime import lime_image
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.util import img_as_float
import PIL
import saliency


#Grad-CAM & Guided Grad-CAM
# Reference: https://github.com/eclique/keras-gradcam with adaption to tensorflow 2.0
@tf.custom_gradient
def guidedRelu(x):
    def grad(dy):
        return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy
    return tf.nn.relu(x), grad


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
        gbModel = Model(inputs=[self.model.inputs], outputs=[self.model.get_layer(self.layerName).output])
        layer_dict = [layer for layer in gbModel.layers[1:] if hasattr(layer,"activation")]
        for layer in layer_dict:
            if layer.activation == tf.keras.activations.relu:
                layer.activation = guidedRelu
        return gbModel

    def guided_backprop(self, images, upsample_size):
        with tf.GradientTape() as tape:
            inputs = tf.cast(images, tf.float32)
            tape.watch(inputs)
            outputs = self.gbModel(inputs)
        grads = tape.gradient(outputs, inputs)[0]
        saliency = cv2.resize(np.asarray(grads), upsample_size)
        return saliency


def deprocess_image(x):
    x = x.copy()
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


class GradCAM:
    def __init__(self, model, layerName=None):
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
        gradModel = Model(inputs=[self.model.inputs], outputs=[self.model.get_layer(self.layerName).output, self.model.output])
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOuts, preds) = gradModel(inputs)
            if classIdx is None:
                classIdx = np.argmax(preds)
            loss = preds[:, classIdx]
        grads = tape.gradient(loss, convOuts)
        convOuts = convOuts[0]
        grads = grads[0]
        norm_grads = tf.divide(grads, tf.reduce_mean(tf.square(grads)) + tf.constant(eps))
        weights = tf.reduce_mean(norm_grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOuts), axis=-1)
        cam = np.maximum(cam, 0)
        cam = cam / np.max(cam)
        cam = cv2.resize(cam, upsample_size,cv2.INTER_LINEAR)
        cam3 = np.expand_dims(cam, axis=2)
        cam3 = np.tile(cam3, [1, 1, 3])
        return cam3


def overlay_gradCAM(img, cam3):
    cam3 = np.uint8(255 * cam3)
    cam3 = cv2.applyColorMap(cam3, cv2.COLORMAP_HOT)
    new_img = 0.6 * cam3 + 0.2 * img
    return (new_img * 255.0 / new_img.max()).astype("uint8")

#LIME

def explanation_heatmap(exp, exp_class):
    dict_heatmap = dict(exp.local_exp[exp_class])
    heatmap = np.vectorize(dict_heatmap.get)(exp.segments)
    return heatmap


def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

#XRAI method
#Reference: adapted from 'https://github.com/PAIR-code/saliency'

def ShowImage(im, title='', ax=None):
    if ax is None:
        P.figure()
    P.axis('off')
    P.imshow(im)
    P.title(title)

def ShowGrayscaleImage(im, title='', ax=None):
    if ax is None:
        P.figure()
    P.axis('off')
    P.imshow(im, cmap=P.cm.gray, vmin=0, vmax=1)
    P.title(title)

def ShowHeatMap(im, title, ax=None):
    if ax is None:
        P.figure()
    P.axis('off')
    P.imshow(im, cmap='inferno')
    P.title(title)

def LoadImage(file_path):
    im = PIL.Image.open(file_path)
    im = im.resize((224,224))
    im = np.asarray(im)
    return im

def call_model_function(images, call_model_args=None, expected_keys=None):
    target_class_idx =  call_model_args[class_idx_str]
    images = tf.convert_to_tensor(images)
    with tf.GradientTape() as tape:
        if expected_keys==[saliency.base.INPUT_OUTPUT_GRADIENTS]:
            tape.watch(images)
            _, output_layer = model(images)
            output_layer = output_layer[:,target_class_idx]
            gradients = np.array(tape.gradient(output_layer, images))
            return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
        else:
            conv_layer, output_layer = model(images)
            gradients = np.array(tape.gradient(output_layer, conv_layer))
            return {saliency.base.CONVOLUTION_LAYER_VALUES: conv_layer,
                    saliency.base.CONVOLUTION_OUTPUT_GRADIENTS: gradients}


#Computing dice and iou coefficients (for the sake of sparsity only dice coefficients are then reported in the thesis as they produced similar results)

def dice_coefficient(img1, img2):
    intersection = np.logical_and(img1, img2)
    return 2.0 * intersection.sum() / (img1.sum() + img2.sum())

def iou(img1, img2):
    intersection = np.logical_and(img1, img2).sum()
    union = np.logical_or(img1, img2).sum()
    iou_score = intersection / union
    return iou_score

"""
The following functions processes the images and computes dice coefficient and IOU.

Args:
img_class: class of the image
pretrained_model: pre-trained model to use for image processing
drive: the drive where the images are located
modality: modality of the MRI
stacked: boolean flag to decide whether to convert img_arr to RGB or not

Returns:
df: dataframe with image details and dice coefficient and IOU
"""
#creating lists to store dice and iou coefficeints
rows_list = []
ggcam_list = []
xrai_list = []
lime_list = []

def gradcam_coef(img_class, drive, modality, pretrained_model, stacked=False):
    img_dir = f"{drive}/{modality}_new_MRI_slices/test/" + img_class + f"_{modality}"
    img_files = os.listdir(img_dir)
    msk_dir = f"{drive}/{modality}_new_Msk_slices/test/" + img_class + "_masks"
    msk_files = os.listdir(msk_dir)

    for i, img_file in enumerate(img_files):
        img_name = img_file.split(".")[0]
        img_num = int(img_name.split("_")[0])
        pat_num = int(img_name.split("_")[1])

        # Load and preprocess the image array
        img_arr = np.load(os.path.join(img_dir, img_file))
        if not stacked:
            img_arr = cv2.cvtColor(img_arr.astype('float32'), cv2.COLOR_GRAY2RGB)
        img_rgb = tf.image.crop_to_bounding_box(img_arr, 8, 8, 224, 224)
        img_rgb = img_rgb.numpy()

        # Load and preprocess the mask array
        img_msk = np.load(os.path.join(msk_dir, img_file))
        msk_rgb = np.clip(img_msk, 0, 255).astype('float32')
        msk_rgb = cv2.cvtColor(msk_rgb.astype('float32'), cv2.COLOR_GRAY2RGB)
        msk_rgb = tf.image.crop_to_bounding_box(msk_rgb, 8, 8, 224, 224)
        msk_rgb = msk_rgb.numpy()

        # Calculate GradCam
        target_size = (224,224)
        gradCAM = GradCAM(model=pretrained_model)
        cam3 = gradCAM.compute_heatmap(image=np.expand_dims(img_rgb,axis=0),classIdx=0,upsample_size=target_size)
        msk_rgb = cv2.normalize(msk_rgb, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

        # Convert GradCam and mask to binary
        cam3 = cv2.normalize(cam3, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        msk_rgb[np.mean(msk_rgb, axis=-1)<0.2] = 0
        msk_rgb[np.mean(msk_rgb, axis=-1)>=0.2] = 1

        cam3[np.mean(cam3, axis=-1)<0.5] = 0
        cam3[np.mean(cam3, axis=-1)>=0.5] = 1

        dice_coef = dice_coefficient(msk_rgb, cam3)
        iou_coef = iou(msk_rgb, cam3)

        # Append a new row to the dataframe with the data
        new_row = {"img_class": img_class, "img_num": img_num, "pat_num": pat_num, "dice_coef": dice_coef, "iou_coef": iou_coef}
        rows_list.append(new_row)

    df = pd.DataFrame(rows_list)
    return df

def guided_grad_coef(img_class, drive, modality, pretrained_model, stacked=False):
    
    img_dir = f"{drive}/{modality}_new_MRI_slices/test/{img_class}_{modality}"
    img_files = os.listdir(img_dir)
    msk_dir = f"{drive}/{modality}_new_Msk_slices/test/{img_class}_masks"
    msk_files = os.listdir(msk_dir)

    for i, img_file in enumerate(img_files):
        img_name = img_file.split(".")[0]
        img_num = int(img_name.split("_")[0])
        pat_num = int(img_name.split("_")[1])

        img_arr = np.load(os.path.join(img_dir, img_file))
        if not stacked:
            img_arr = cv2.cvtColor(img_arr.astype('float32'), cv2.COLOR_GRAY2RGB)
        img_rgb = tf.image.crop_to_bounding_box(img_arr, 8, 8, 224, 224)
        img_rgb = img_rgb.numpy()

        img_msk = np.load(os.path.join(msk_dir, img_file))
        msk_rgb = np.clip(img_msk, 0, 255).astype('float32')
        msk_rgb = cv2.cvtColor(msk_rgb.astype('float32'), cv2.COLOR_GRAY2RGB)
        msk_rgb = tf.image.crop_to_bounding_box(msk_rgb, 8, 8, 224, 224)
        msk_rgb = msk_rgb.numpy()

        target_size = (224,224)
        gradCAM = GradCAM(model=pretrained_model)
        cam3 = gradCAM.compute_heatmap(image=np.expand_dims(img_rgb,axis=0),classIdx=0,upsample_size=target_size)

        guidedBP = GuidedBackprop(model=pretrained_model)
        gb_cam = guidedBP.guided_backprop(np.expand_dims(img_rgb,axis=0),target_size)
        guided_gradcam = deprocess_image(gb_cam*cam3)

        cam3 = gradCAM.compute_heatmap(image=np.expand_dims(img_rgb,axis=0),classIdx=0,upsample_size=target_size)
        msk_rgb = cv2.normalize(msk_rgb, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

        msk_rgb[np.mean(msk_rgb, axis=-1)<0.2] = 0
        msk_rgb[np.mean(msk_rgb, axis=-1)>=0.2] = 1

        guided_gradcam = cv2.normalize(guided_gradcam, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        guided_gradcam[np.mean(guided_gradcam, axis=-1)<0.6] = 0
        guided_gradcam[np.mean(guided_gradcam, axis=-1)>=0.6] = 1

        dice_coef = dice_coefficient(msk_rgb, guided_gradcam)
        iou_coef = iou(msk_rgb, guided_gradcam)

        new_row = {"img_class": img_class, "img_num": img_num, "pat_num": pat_num, "dice_coef": dice_coef, "iou_coef": iou_coef}
        ggcam_list.append(new_row)

    df = pd.DataFrame(ggcam_list)

    return df


def xrai_coef(img_class, drive, modality, pretrained_model, stacked=False):
    img_dir = f"{drive}/{modality}_MRI_slices/test/" + img_class + modality
    img_files = os.listdir(img_dir)
    msk_dir = f"{drive}/{modality}_Msk_slices/test/" + img_class + modality
    msk_files = os.listdir(msk_dir)

    for i, img_file in enumerate(img_files):
        img_name = img_file.split(".")[0]
        img_num = int(img_name.split("_")[0])
        pat_num = int(img_name.split("_")[1])

        img_arr = np.load(os.path.join(img_dir, img_file))
        img_arr = cv2.resize(img_arr, (224, 224))
        img_float32 = np.float32(img_arr)
        if not stacked:
            img_rgb = cv2.cvtColor(img_float32, cv2.COLOR_GRAY2RGB)

        img_msk = np.load(os.path.join(msk_dir, img_file))
        img_msk = cv2.resize(img_msk, (224, 224))
        msk_float32 = np.float32(img_msk)
        msk_rgb = cv2.cvtColor(msk_float32, cv2.COLOR_GRAY2RGB)
        
        msk_rgb = cv2.normalize(msk_rgb, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        msk_rgb[np.mean(msk_rgb, axis=-1)<0.2] = 0
        msk_rgb[np.mean(msk_rgb, axis=-1)>=0.2] = 1

        img_rgb = img_as_float(img_rgb)
        xrai_object = saliency.XRAI()
        xrai_attributions = xrai_object.GetMask(img_rgb, call_model_function, call_model_args, batch_size=20)
        mask = xrai_attributions >= np.percentile(xrai_attributions, 97)
        im_mask = np.array(img_rgb)
        im_mask[~mask] = 0

        im_mask = cv2.normalize(im_mask, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        im_mask[np.mean(im_mask, axis=-1)<0.2] = 0
        im_mask[np.mean(im_mask, axis=-1)>=0.2] = 1

        dice_coef = dice_coefficient(msk_rgb, im_mask)
        iou_coef = iou(msk_rgb, im_mask)

        new_row = {"img_class": img_class, "img_num": img_num, "pat_num": pat_num, "dice_coef": dice_coef, "iou_coef": iou_coef}
        xrai_list.append(new_row)

    df = pd.DataFrame(xrai_list)
    return df



def lime_coef(img_class, drive, modality, pretrained_model, stacked=False):
    img_dir = f"{drive}/{modality}_new_MRI_slices/test/{img_class}_{modality}"
    img_files = os.listdir(img_dir)
    msk_dir = f"{drive}/{modality}_new_Msk_slices/test/{img_class}_masks"
    msk_files = os.listdir(msk_dir)

    for i, img_file in enumerate(img_files):
        img_name = img_file.split(".")[0]
        img_num = int(img_name.split("_")[0])
        pat_num = int(img_name.split("_")[1])

        img_arr = np.load(os.path.join(img_dir, img_file))
        if not stacked:
            img_col = cv2.cvtColor(img_arr.astype('float32'), cv2.COLOR_GRAY2RGB)
        img_rgb = tf.image.crop_to_bounding_box(img_col, 8, 8, 224, 224)
        img_rgb = img_rgb.numpy()
        img_rgb = tf.keras.applications.imagenet_utils.preprocess_input(img_rgb, mode='tf')

        img_msk = np.load(os.path.join(msk_dir, img_file))
        msk_rgb = np.clip(img_msk, 0, 255).astype('float32')
        msk_rgb = cv2.cvtColor(msk_rgb.astype('float32'), cv2.COLOR_GRAY2RGB)
        msk_rgb = tf.image.crop_to_bounding_box(msk_rgb, 8, 8, 224, 224)
        msk_rgb = msk_rgb.numpy()

        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(img_rgb,
                                                pretrained_model.predict,
                                                top_labels=1,
                                                hide_color=0,
                                                num_samples=1000)

        temp_1, mask_1 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=1, hide_rest=True)
        img_hide = mark_boundaries(temp_1, mask_1)
        img_hide = img_hide/np.amax(img_hide)
        img_hide = np.clip(img_hide, 0, 1)

        msk_rgb = cv2.normalize(msk_rgb, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

        msk_rgb[np.mean(msk_rgb, axis=-1)<0.2] = 0
        msk_rgb[np.mean(msk_rgb, axis=-1)>=0.2] = 1

        img_hide = cv2.normalize(img_hide, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        img_hide[np.mean(img_hide, axis=-1)<0.2] = 0
        img_hide[np.mean(img_hide, axis=-1)>=0.2] = 1

        dice_coef = dice_coefficient(msk_rgb, img_hide)
        iou_coef = iou(msk_rgb, img_hide)

        new_row = {"img_class": img_class, "img_num": img_num, "pat_num": pat_num, "dice_coef": dice_coef, "iou_coef": iou_coef}
        lime_list.append(new_row)

    df = pd.DataFrame(lime_list)

    return df
