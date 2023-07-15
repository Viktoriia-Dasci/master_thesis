# master_thesis


# MRI Modality Training and Evaluation

This repository contains code for training and evaluating models for different MRI modalities (t1, t1ce, t2, flair, and stack) and performing explainability analysis using Grad-CAM, Guided Grad-CAM, LIME, and XRAI methods.

## Data Preprocessing and MRI Slice Extraction

1. Import the necessary libraries, including os, glob, numpy, nibabel, splitfolders, and pathlib.
2. Define the `create_directory` function to create directories if they don't already exist.
3. Specify the `input_folder` and `output_folder` paths for the dataset.
4. Use the `split_folders_with_ratio` function to split the dataset into train, validation, and test folders based on the specified ratio.
5. Define a list of file paths for training and testing, and create necessary directories for saving the MRI slices.
6. Use the `extract_slices` function to extract and save individual MRI slices for each modality and data type.
7. Use the `stack_3_slices` function to extract, save, and stack three slices from the t2, t1ce, and flair modalities.
8. Use the `stack_4_slices` function to extract, save, and stack four slices from the t2, t1ce, flair, and t1 modalities.

## Preparation

1. Import the necessary custom functions from the `Model_functions` and `explainabilty_functions` modules.
2. Set the `home_dir` and `base_dir` variables to the corresponding directory paths.
3. Specify the `modality` variable for the desired MRI modality.
4. Load the data for training, validation, and testing using the `load_from_dir` function.
5. Preprocess and combine the HGG and LGG data.
6. Convert labels to numerical values and perform one-hot encoding.
7. Convert data to arrays and shuffle the datasets.

## Model Training

1. Generate class weights for imbalanced datasets.
2. Perform data augmentation using the ImageDataGenerator.
3. Set up hyperparameter tuning using the Hyperband tuner.
4. Search for the best hyperparameters and models for each tuner.
5. Define callbacks for early stopping, learning rate reduction, and model checkpointing.
6. Specify the folder path for saving the training plots.
7. Fit the best models from each tuner to the training data using the best hyperparameters.

## Explainability Analysis

1. Load the pretrained models for explainability analysis.
2. Plot ROC curves for the loaded models.
3. Generate predictions and compute classification metrics.
4. Plot a confusion matrix.
5. Perform Grad-CAM and Guided Grad-CAM analysis.
6. Perform LIME analysis.
7. Perform XRAI analysis.
8. Save the computed coefficients to CSV files for further analysis.

**Note**: Make sure to have the required dependencies installed before running the code.
