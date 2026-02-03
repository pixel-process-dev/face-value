# Face-Value

## Overview
Face Value (FV) is a hosted affective signal model that estimates facial expression distributions from images and video. Rather than optimizing for benchmark accuracy alone, the system is evaluated using face-valid external criteria, including generalization to established emotion datasets and aggregate behavioral alignment on culturally stable stimuli such as film.

## Motivation & Goals
Face Value aims to demonstrate an end-to-pipeline for collecting images with minimal manual curation for classification with ecological validity. Classification of emotional expressions often struggles to bridge the gap between controlled datasets and "in the wild" expressions. Reported metrics here inform how this approach performs in multiple scenarios. 

- **Multiple Validation**: FV assesses validity on multiple levels for both internal consistency and generalization.
    - **Ecological**: facial expressions from well known movies are extracted and classified. Although accuracy on individual faces is not assessed here, clear patterns support that the approach generalizes well. 
    - **Standardized**: evaluation against two established datasets of emotional facial expressions (FER 2013, CK+) are used to provide comparison with established metrics. 
    - **Performance**: classification metrics, confusion matrices, and performance metrics demonstrate that this process yields stable signal (with minimal curation) for images collected. 
- **Automated Pipeline**: FV builds on established models (Resnet, MediaPipe face detector) and open data sources for an automated approach for end-to-end classification. Emotional faces are used here, but the process which relies on keyword searchers grouped by categories is extensible. Of note, use of a face detection model resulted in many initial images dropped for not meeting requirements reducing noise. 


## Data Sources

### Open-Source Image Datasets
Data sources used are freely available with licensing permissive of model and project testing for non-commercial uses. The raw images remain the property of the source and are therefore not shared or distributed within this project.

- [Pixabay](https://pixabay.com/): Stunning royalty-free images & royalty-free stock
- [Pexels](https://www.pexels.com/): The best free stock photos, royalty free images & videos shared by creators.
- [Unsplash](https://unsplash.com/): Visuals for everyone

For each source two versions of querying were conducted.
- **Version 1** used the format of **[emotion] face**
- **Version 2** used 3 keywords for each emotion category:
    - **angry**: "angry", "mad", "irate"
    - **disgust**: "disgusted", "gross", "repulsed"
    - **fear**: "fear", "afraid", "scared"
    - **happy**: "happy", "smiling", "joyful"
    - **sad**: "sad", "crying", "unhappy"
    - **surprise**: "surprised", "shocked", "astonished"
    - **neutral**: "neutral", "calm", "expressionless"
- **Version 3**: indicates combined results from Versions 1 & 2

### Curated / External Media (Films)

## Target Labels & Emotion Taxonomy
Emotion classes included 5 expressions: Angry, Fear, Happy, Sad, Surprise.

Emotion categories followed those used in [FER 2013]: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral. 
Disgust was dropped due to a low volumn of images returned and faces extracted.
Neutral was dropped as too ambiguous. A future version might use low probability of model classification as "neutral".

**Note**: Adding Contempt would allow better comparison with CK+. 

## Dataset Construction
### Inclusion Criteria
### Preprocessing & Normalization
### Class Balance & Sampling Strategy

## Validation Strategy
### Train / Validation / Test Splits
### Benchmark Datasets (FER2013, CK+)
### Out-of-Distribution Evaluation (Films)

## Modeling Approach
### Baseline Models
### Advanced Models
### Training Strategy

## Evaluation Metrics
### Dataset-Specific Metrics
### Cross-Domain Generalization

## Results & Observations
### Performance on Benchmark Datasets
### Performance on Film Data

## Workflow & Reproducibility
### Project Structure

```text

├── configs
│   ├── data_pull
│   └── training
├── data
│   ├── archive
│   ├── ck-faces
│   ├── fer-2013
│   ├── processed
│   └── raw
├── environment.yml
├── evaluation
│   ├── combined_v1
│   ├── pexels_v1
│   ├── pexels_v1_r2
│   ├── pixabay_comb_v1
│   ├── pixabay_v1
│   ├── pixabay_v1_lr001
│   └── pixabay_v1_r2
├── models
│   ├── mediapipe_face_detector
│   ├── pexels_v1
│   ├── pexels_v1_r2
│   ├── pexels_v2
│   ├── pixabay_combined_mlflow1
│   ├── pixabay_comb_v1
│   ├── pixabay_light_aug_comb
│   ├── pixabay_light_aug_v1
│   ├── pixabay_light_aug_v2
│   ├── pixabay_v1
│   ├── pixabay_v1_lr001
│   ├── pixabay_v1_mlflow1
│   ├── pixabay_v1_r2
│   └── resnet_trained_weights
├── notebooks
│   ├── archive
│   ├── ConfusionMatrixPlottingPlotly.ipynb
│   ├── data_source_mixing.ipynb
│   ├── FACE_EDA_MERGE.ipynb
│   ├── fer2013.ipynb
│   ├── movie_analyzer.ipynb
│   └── movie_comparisons.ipynb
├── README.md
└── src
    ├── api_pulls
    ├── archive
    ├── batch_runner.py
    ├── face_extraction.py
    ├── mlflow.db
    ├── mlruns
    ├── model_config.py
    ├── movie_analyzer.py
    ├── __pycache__
    └── train_from_config.py
```

### Experiment Tracking
### Configuration & Parameters

## Tools & Technologies

- **[ResNet](https://pytorch.org/hub/pytorch_vision_resnet/)** from PyTorch is base model for fine-tuning.
- **[MediaPipe face detector](https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector)**: is used for face extraction from images.
- **[ML Flow](https://mlflow.org/)**: for model and metric tracking
- PyTorch, polars, plotly, PIL, cv2, scikit-learn, Jupyter, conda

## Known Limitations

## Ethical Considerations & Bias

## Future Work
- Add unsplash data
- Additional disgust pulls/generation
- Add Contempt

## How to Run
### Environment Setup
- `environment.yml` has package details
- Use [conda](https://anaconda.org/) to build with `conda env create -f environment.yml`
- Additional face detector model file may be downloaded from [MediaPipe](https://mediapipe.readthedocs.io/en/latest/solutions/face_detection.html). Save to `nodels/mediapipe_face_detector/detector.tflite`.
- Accounts will need to be set up on data sources for API access and keys

### Data Pulls
A script for each datasource is in `src/api_pull` and take a json configuration file that determines keywords and output locations. Each API has different parameters and limits, so these are unique per source currently.

#TODO: Create unified request script by moving api settings to config files.

Example usage:
`python unsplash_pull.py ../../configs/data_pull/unsplash_v1_2.2.26.json`

Outputs (located set in config) include:
- Images stored in "[emotion]/[keyword]" directory structure
- **manifest.parquet**: details on each image result
- **run_log.parquet**: details on each request
- **seen_ids.parquet**: helps detect if duplicate images are being returned

### Face Extraction
Script iterates over subdirectories from raw data images to detect and extract faces into their own images. Also creates summary datasets and EDA.  

Example usage:
`python face_extraction.py --raw_dir ../data/raw/unsplash_v1/ --out_dir ../data/processed/unsplash_v1`

Outputs include:
- Cropped images of faces
- Summary Files:
    - aggregate_stats.parquet: summary stats from face extraction
    - face_level.parquet: 1 row/face image with details on path, size, etc.
    - image_level.parquet: 1 row/original image with details on if face found, size
    - training_data.parquet: combined image and face level details with images that did not have a face dropped
- EDA
    - face_area_box.html: total pixel count of face
    - face_area_relative_box.html: proportion of image covered by face
    - face_yield.html: proportion of images with faes
    - image_counts.html: total counts

### Data Preparation
#### 

### Training
### Evaluation

## References & Credits
