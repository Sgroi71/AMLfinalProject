# Natural Language Queries in Egocentric Videos  
*Pretrain on LLM-Based Generated Queries*

## Overview  
This repository contains the implementation of AML final project, which addresses the challenge of answering natural language queries (NLQs) in egocentric videos. By leveraging large language models (LLMs) to generate synthetic NLQs from video narrations, we propose a novel two-step training pipeline that improves query understanding and localization performance.  

The core workflow is presented in a Google Colab notebook for ease of use and reproducibility. You can access it [here](https://colab.research.google.com/drive/1_d9-IKHrs8dwXlaRAZOznhUVYrj3-jH-?usp=sharing).

---

## Contributors 
- **Leonardo Sgroi** - Politecnico di Torino - [s331398@studenti.polito.it](mailto:s331398@studenti.polito.it)  
- **Matteo Bonsignore** - Politecnico di Torino - [s329653@studenti.polito.it](mailto:s329653@studenti.polito.it)  
- **Alessandro Romeo** - Politecnico di Torino - [s324247@studenti.polito.it](mailto:s324247@studenti.polito.it)  

---

## Features  
- Leverages LLMs to generate synthetic natural language queries (NLQs) from video narrations.  
- Implements a two-phase training process: pretraining on synthetic queries and fine-tuning on original NLQ datasets.  
- Validates the methodology on the Ego4D dataset using state-of-the-art models and configurations.  

---

## Requirements  
The code relies on Python and popular deep learning frameworks such as PyTorch. Detailed requirements and dependencies are provided in the `requirements.txt` file.  

---
## How to Use  
1. Open the Google Colab notebook [here](https://colab.research.google.com/drive/1_d9-IKHrs8dwXlaRAZOznhUVYrj3-jH-?usp=sharing).

2. Insert your credentials in the first cell to access to the EGO4D dataset. Sign the license as described [here](https://ego4d-data.org/docs/start-here/#license-agreement).

3. If you want to use the EGOVLP features, download them from [here](https://drive.google.com/file/d/1U318S34jw3uNnsURJ1T40YwsSuK5_-RJ/view?usp=share_link), creating a copy in your Google Drive in the directory Colab Notebook.

4. Choose your configuration by setting the enviroment variable in the fifth code cell.

5. Make sure you are using a GPU runtime.
