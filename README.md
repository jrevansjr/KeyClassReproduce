<h1 align="center">KeyClass: Text Classification with Label-Descriptions Only</h1>

<p align="center">
<img alt="License" src="https://img.shields.io/badge/License-MIT-yellow.svg">
<img alt="Visitors" src="https://visitor-badge.glitch.me/badge?page_id=autonlab/KeyClass">
</p>

`KeyClass` is a general weakly-supervised text classification framework that learns from *class-label descriptions only*, without the need to use any human-labeled documents. It leverages the linguistic domain knowledge stored within pre-trained language models and the data programming framework to assign labels to documents. We demonstrate the efficacy and flexibility of our method by comparing it to state-of-the-art weak text classifiers across four real-world text classification datasets.

## Contents

1. [Overview of Methodology](#methodology) 
2. [KeyClass Outperforms Advanced Weakly Supervised Models](#results) 
3. [Datasets](#datasets)
4. [Instructions](#instructions)
5. [Citation](#citation)
6. [Contributing](#contrib)
7. [License](#license)

<a id="methodology"></a>
## Overview of Methodology 

<p align="center">
<img height ="300px" src="assets/KeyClass.png">
</p>

**Figure.1** From class descriptions only, KeyClass classifies documents without access to any labeled data. It automatically creates interpretable labeling functions (LFs) by extracting frequent keywords and phrases that are highly indicative of a particular class from the unlabeled text using a pre-trained language model. It then uses these LFs along with Data Programming (DP) to generate probabilistic labels for training data, which are used to train a downstream classifier [(Ratner et al., 2016)](https://arxiv.org/abs/1605.07723)

<a id="results"></a>
## `KeyClass` Outperforms Advanced Weakly Supervised Models

<p align="center">
<img height ="120px" src="assets/result_table.png">
</p>

**Table 1.    Classification Accuracy.** `KeyClass` outperforms state-of-the-art weakly supervised methods on 4 real-world text classification datasets. We report our model’s accuracy with a 95% bootstrap confidence intervals. Results for Dataless, WeSTClass,
LOTClass, and BERT are reported from [(Meng et al., 2020)](https://arxiv.org/abs/2010.07245).


----
<a id="datasets"></a>
## Datasets

The datasets for this model are private, please see original KeyClass repository for baseline models. In addition, for MIMIC-III dataset, please visit Physionet.org.

<a id="instructions"></a>
## Installation

All models were built and trained using Google Colab. The notebook is designed to run end to end. It is recommended to put your hugging face token in the colab secrets book if you want to pull a fresh model from the website. However, if this is left blank, the program will just retrieve previous models downloaded from the google drive. After collecting the data and n-grams from the CDC catagory definitions, the three step weak supervision process starts in section 3.

Via the variable config_file_path at the top of section 3.2.1, is how the dataset and model are selected. The default configuration file is a slimmed down version of the imdb dataset to ensure that the notebook will run under 8 minutes in Colab on a CPU.

Important Scripts:
1. keyclass/config_files - Configuration files, information pertaining to the model architecture, training hyperparameters, and definitions mined from the CDC.
2. keyclass/create_lfs - get_vocabulary function gathers the n-grams from the training corpus and the assign_categories_to_keywords function assigns these to labeling functions based on their embedding similarity to the categories. The remaining functions leverage the snorkel library to assign probabilistic labels to each of the files.
3. keyclass/models - contains the encoder, downstream model torch abstractions.
4. keyclass/train_classifiers - contain the training loops for executing the finetuning of the base hugging face model.

<a id="citation"></a>
## Citation
If you use our code please cite the following paper. 
```
@article{gao2022classifying,
  title={Classifying Unstructured Clinical Notes via Automatic Weak Supervision},
  author={Gao, Chufan and Goswami, Mononito and Chen, Jieshi and Dubrawski, Artur},
  journal={Machine Learning for Healthcare Conference},
  year={2022},
  organization={PMLR}
}
```
<a id="contrib"></a>
## Contributing

`KeyClass` is [on GitHub]. We welcome bug reports and pull requests.

[on GitHub]: https://github.com/autonlab/KeyClass.git

<a id="license"></a>
## License

MIT License

Copyright (c) 2022 Carnegie Mellon University, Auton Lab.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

See [MIT LICENSE](https://github.com/autonlab/KeyClass/blob/main/LICENSE) for details.

<img align="right" height ="120px" src="assets/cmu_logo.png">
<img align="right" height ="110px" src="assets/autonlab_logo.png"> 
