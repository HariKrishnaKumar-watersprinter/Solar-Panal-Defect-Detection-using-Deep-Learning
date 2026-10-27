Solar Panel Defect Detection using Deep Learning
Automated classification of solar panel conditions from images using transfer learning with EfficientNetB0 — served through an interactive Streamlit web app.

PythonTensorFlowKerasStreamlitJupyter

Overview
Solar panels lose significant power output when their surface is obstructed by dust, bird droppings, or snow, or when they suffer electrical and physical faults. Detecting these issues early is critical for maintaining plant efficiency, yet manual inspection of large solar farms is slow, expensive, and inconsistent. This project builds an image-classification pipeline that inspects a photo of a panel and automatically determines its condition, enabling fast, repeatable, and scalable quality checks.

The system was developed experimentally in a Jupyter/Colab notebook, starting from a CNN built from scratch and progressively moving to transfer learning with MobileNetV2 and finally EfficientNetB0, which was then optimized with KerasTuner. The best configuration achieved 83.05% validation accuracy across six condition classes and is packaged as the deployed model in this repository. A lightweight Streamlit application wraps the model so that anyone can upload a panel image and receive an instant diagnosis with per-class confidence scores.

Table of Contents
Defect Classes
Dataset
Model Development Journey
Results
Final Model Architecture
Hyperparameter Optimization
Tech Stack
Getting Started
Project Structure
Roadmap
Author
Defect Classes
The classifier distinguishes six operational conditions of a solar panel:

Class	Description
Clean	Panel surface free of soiling or defects — the healthy reference condition
Bird-drop	Bird droppings accumulated on the panel, blocking irradiance and risking hotspots
Dusty	Dust and soiling settled on the surface, reducing power output
Electrical-damage	Electrical faults such as burn marks, wiring or junction-box failures
Physical-damage	Structural damage such as cracks, broken glass or deformed frames
Snow-Covered	Panel obstructed by snow deposition, temporarily out of service
Dataset
Property	Value
Total images	885
Classes	6 (one folder per class)
Split	80% train / 20% validation (708 / 177 images)
Image size	224 × 224, resized at load time
Batch size	32
Split seed	42 (reproducible)
The dataset is organized as class-named subfolders and loaded with tf.keras.utils.image_dataset_from_directory.
The class distribution is imbalanced — Physical-Damage and Electrical-damage have fewer samples than Clean or Dusty — so class weights are computed and passed to model.fit() to prevent the minority defect classes from being ignored during training.
Data augmentation (horizontal flip, rotation, zoom) is applied on-the-fly to improve generalization on this small dataset.
The dataset itself is not bundled in this repository; point DATASET_DIR in the notebook to your own copy structured as DATASET_DIR/<class_name>/*.jpg.
Model Development Journey
The notebook documents the full experimentation path rather than jumping straight to the final model, which makes it a useful reference for anyone facing a similar small-dataset classification problem:

Baseline CNN (from scratch) — a three-block Conv2D network (32 → 64 → 128 filters) with a 128-unit dense head (~11.2M parameters). It memorized the training set (99.45% train accuracy) but generalized poorly (~62% validation accuracy), a textbook overfitting case on 708 training images.
Regularization attempts — BatchNormalization, Dropout, EarlyStopping and a learning-rate schedule were added. Training became unstable on this small dataset and validation performance degraded, showing that regularization alone could not compensate for the lack of data.
Data-augmented deeper CNN — on-the-fly augmentation plus a wider convolutional trunk was tried next, but results confirmed that a from-scratch CNN was not the right tool for 885 images.
Transfer learning with MobileNetV2 — an ImageNet-pretrained MobileNetV2 with a frozen base and a small classification head immediately lifted best validation accuracy to ~76%, validating the transfer-learning approach.
Transfer learning with EfficientNetB0 — switching to EfficientNetB0 (with its native preprocess_input, augmentation and class weights) pushed the best validation accuracy to ~81%.
Hyperparameter optimization with KerasTuner — a 20-trial random search over augmentation strength, dropout, head size and learning rate produced the final configuration and the best overall validation accuracy of 83.05%.
Results
Best validation accuracy observed for each approach (177-image validation set):

#	Model	Best Val Accuracy	Remarks
1	Custom CNN (from scratch)	62.2%	Severely overfit — train accuracy 99.5%
2	Regularized CNN (BatchNorm + Dropout + LR schedule)	28.8%	Training collapsed on the small dataset
3	MobileNetV2, frozen base	76.3%	First big jump via transfer learning
4	EfficientNetB0, frozen base + augmentation + class weights	81.4%	Strong and stable
5	EfficientNetB0 + KerasTuner (final)	83.1%	Deployed as model/trained_effnet_finetune.h5
Key takeaway: on a small custom dataset, a frozen pretrained backbone with a lightweight, well-tuned classification head outperforms a from-scratch CNN by more than 20 accuracy points while training only ~165K parameters.

Final Model Architecture
Input (224 × 224 × 3)└── Data Augmentation: RandomFlip(horizontal) + RandomRotation(0.05) + RandomZoom(0.05)└── EfficientNetB0 base (ImageNet weights, frozen)└── GlobalAveragePooling2D└── Dropout (rate = 0.4)└── Dense (128 units, ReLU)└── Dense (6 units, Softmax)
Total parameters: ~4.2M — trainable parameters: ~165K
Preprocessing: EfficientNet preprocess_input; the identical transform is applied in the Streamlit app
Saved format: HDF5 (.h5) at model/trained_effnet_finetune.h5 (~40 MB)
Hyperparameter Optimization
A KerasTuner RandomSearch (20 trials, up to 20 epochs per trial, EarlyStopping with restore_best_weights=True, objective val_accuracy) explored the following space:

Hyperparameter	Search Space	Best Value
Rotation (augmentation)	0.05 – 0.30, step 0.05	0.05
Zoom (augmentation)	0.05 – 0.30, step 0.05	0.05
Dropout rate	0.00 – 0.50, step 0.10	0.40
Dense units (head)	64 – 512, step 64	128
Learning rate (Adam)	1e-4 – 1e-2, log-sampled	~2.9e-3
Total search time was ≈ 6 h 49 min on a GPU runtime, and class weights were passed into the search so the minority defect classes stayed represented.

Tech Stack
Layer	Tools
Deep learning	TensorFlow 2.20 / Keras 3, EfficientNetB0 (ImageNet weights)
Tuning	KerasTuner (RandomSearch)
App	Streamlit
Data / imaging	NumPy, Pillow
Experimentation	Jupyter Notebook / Google Colab, Matplotlib
Getting Started
Prerequisites
Python 3.10 or newer (developed on Python 3.12)
pip package manager
No GPU is required for inference in the Streamlit app; training was performed on Google Colab
Installation
# 1. Clone the repositorygit clone https://github.com/HariKrishnaKumar-watersprinter/Solar-Panal-Defect-Detection-using-Deep-Learning.gitcd Solar-Panal-Defect-Detection-using-Deep-Learning# 2. Create and activate a virtual environmentpython -m venv venv# Windowsvenv\Scripts\activate# macOS / Linuxsource venv/bin/activate# 3. Install dependenciespip install -r requirements.txt
Run the Web App
streamlit run app.py
Then open the page Streamlit starts (usually http://localhost:8501) and:

Upload a solar panel photo (JPG / JPEG / PNG)
The image is preprocessed exactly as during training (RGB, 224 × 224, EfficientNet preprocess_input)
Read the predicted condition with its confidence score
Inspect the color-coded Top-3 predictions and expand the full class-probability breakdown
A clean panel shows a success message; a detected defect raises a warning
Training Notebook
Solar_Panel_Classification.ipynb contains the complete pipeline: dataset loading and splitting, the baseline CNN, regularization experiments, MobileNetV2 and EfficientNetB0 transfer learning, KerasTuner optimization, and model export.

Designed for Google Colab: it mounts Google Drive and reads the dataset from a Drive folder
Update DATASET_DIR to point to your dataset organized as <DATASET_DIR>/<class_name>/<image>.jpg
Run the cells top-to-bottom; the tuned model is exported as an .h5 file that you can drop into model/
Project Structure
Solar-Panal-Defect-Detection-using-Deep-Learning/├── model/│   └── trained_effnet_finetune.h5      # Final KerasTuner-tuned EfficientNetB0 (~40 MB)├── Solar_Panel_Classification.ipynb    # Full training & experimentation pipeline├── app.py                              # Streamlit inference application├── requirements.txt                    # Python dependencies└── .gitignore
Roadmap
 Fine-tune the upper EfficientNetB0 blocks (unfreeze the last few layers with a low learning rate) to push accuracy beyond 83%
 Expand the dataset, especially for the minority Electrical-damage and Physical-damage classes
 Add Grad-CAM heatmaps so predictions are explainable — showing where the model "looked"
 Move from classification to defect localization (bounding boxes) for actionable inspection reports
 Deploy the Streamlit app to Streamlit Community Cloud / Hugging Face Spaces for one-click public access
 Export to TensorFlow Lite / ONNX for edge deployment on drones and embedded inspection devices
 Evaluate with a confusion matrix and per-class precision/recall to quantify class-level behavior
Author
Hari Krishna Kumar — GitHub: @HariKrishnaKumar-watersprinter

Project Link: Solar-Panal-Defect-Detection-using-Deep-Learning
