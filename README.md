**What The project does:
. Chromagram mainly used for music application. 
. Here, we are demonstrating to apply on non-musical application like sound detection of drone vs. noise.

**Why the project is useful:
. It's a simple binary classification using SVM and it is computationally effective.**

****How users can get started with the project:
. Use Public or Private data and run Chroma_Features_Creation.py. It will create 12 chromagram features for each .wav file. 
. To find top features by running through Top_Features_Selection.py file. This will give the score and which will help to select top N features.
. To build and test the model using either top features or using the all features and pass into the Binary_Classification_Models.py. 
. Hypothesis testing using Hypothesis_Testing.py to confirm why chromagram works with non-musical application. For that created two csv files drone_music_data.csv and noise_music_data.csv, 
  passed one by one through the Hypothesis_Testing.py file. From AUC result, it shows that why Chromagram works for non-musical application as well.
. To work with CNN, we need image files. convert_audio_to_image.py file willconvert audio file to image.
. To build and test the model with the generated images using image_processing_using_CNN.py file.
. To compare the performance of GNN against SVM, gnn_classification.py file helps to train and test the data. To build GNN, we use the same chromagram features as we used for binary_classification.

