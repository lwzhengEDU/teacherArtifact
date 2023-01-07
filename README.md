# Teacher Artifact
> Supplementary materials of the article *Inspecting Technology-related Quality of Teaching  Artifacts to Understand Teachers’ Technology Adoption (under revision)*

Teaching artifacts encapsulate the attributes that can reflect teacher expertise in the authentic context. The paper inspects the implicit quality of teaching artifacts to examine teachers’ technology adoption. This repository contains the core code of training machine annotators,  samples of trainning data, and runtime environments that are addressed in the article.

## Code
The code of training machine learning annotators are presented within the Jupyter notebooks. Each notebook demonstrates how we trained an annotator of with the artifact samples that we collected in the study using  deep learning libraries.  In each note book, the  data pre-processing, processes performance metrics, and optimisation processes are detailed in full.

**Trained  Annotators (Machine Learning Techniques):**

- Classification of self-created artifact category ([a joint deep learning model using integrated data](https://arxiv.org/abs/1910.03910))

- Segmentation of design elements in artifacts ([UNet](https://en.wikipedia.org/wiki/U-Net))

- Detection of design elements in artifacts ([YOLOv3](https://pjreddie.com/darknet/yolo/))

- Classification of material structuring principle ([CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network))

- Classification of instruction events ([a joint deep learning model using integrated data](https://arxiv.org/abs/1910.03910))

## Libraries and Environments
> Runing the code in the notebooks requires the dependent libraries or a corresponding environment. The environment config files can be found in [the envs folder.](https://github.com/lwzhengEDU/teacherArtifact/tree/main/envs)
> 
The joint deep leaning model that we trained relies on the [fantatic Fastai  **v1**](https://github.com/fastai/fastai1) and [image_tabular library](https://github.com/naity/image_tabular). Before runing the notebooks ([Artifact_technology_classification.ipynb](https://github.com/lwzhengEDU/teacherArtifact/blob/main/Artifact_technology_classification.ipynb "Artifact_technology_classification.ipynb") and [Artifact_instruction_event_classification_integrated_model.ipynb](https://github.com/lwzhengEDU/teacherArtifact/blob/main/Artifact_instruction_event_classification_integrated_model.ipynb "Artifact_instruction_event_classification_integrated_model.ipynb") ), please install two libraries. Alternatively, download the environment file  [env.yml](https://github.com/lwzhengEDU/teacherArtifact/blob/main/envs/environment_jointDeepLearning.yml)  and  run 
```bash
coda env create -f environment_jointDeepLearning.yml
```
 in the terminal or Anaconda Prompt to create a new environment.

The UNet model that we trained for artficat segmentation relies on the [Fastai **v2**](https://github.com/fastai/fastai) . Before runing [the notebook](https://github.com/lwzhengEDU/teacherArtifact/blob/main/Artifact_segmentation_text.ipynb), please install Fastai v2 library. Alternatively, run 
```bash
coda env create -f environment_segmentation.yml
```
 in the terminal or Anaconda Prompt

The Yolo that we trained for the detection of design elements relies on [iceVision library](https://github.com/airctic/icevision). Before runing [the notebook](https://github.com/lwzhengEDU/teacherArtifact/blob/main/Artifact_objectDetection.ipynb), please run 
```bash
coda env create -f environment_icevision.yml
```
to install the icevision framework and the pretrained models of object detection.

The CNN classifier that we trained to code the material structuring principles of artifact relies on the [Fastai **v2** library](https://github.com/fastai/fastai) . before running [the notebook](https://github.com/lwzhengEDU/teacherArtifact/blob/main/Artifact_Signaling_Classification.ipynb), please install fastai v2 library. Alternatively,  run 
```bash
coda env create -f environment_classification.yml
```
 in the terminal or Anaconda Prompt to create a new environment.



## Data

We provided a few data samples for each training task in the folder [data](https://github.com/lwzhengEDU/teacherArtifact/tree/main/data). In the tasks of the design element analysis (segmentation & detection), we utilized  [labelme](https://github.com/wkentaro/labelme), a graphical image annotation tool, to collect the regions of the deign elements.
