# Show and Tell: A Neural Image Caption Generator

[Link to paper](https://arxiv.org/pdf/1411.4555.pdf)

Our results can be found in [report.pdf](report.pdf).

### `model.py`
This contains our show-and-tell model definition. It requires the pretrained Resnet34 model.

### `dataset.py`
This contains a custom dataset that wraps the PyTorch COCO dataset.

### `train.py`
This file is used to train only the last layer of the CNN, the LSTM, and the decoder network.

Usage: `python train.py`

### `train_entire_network.py`
This file is used to train the entire network. Gradients are required for every layer.

Usage: `python train_entire_network.py`

### `test.py`
This file is used to generate `results.json`. This `results.json` is compared against the validation data to compute our metrics for the BLEU, CIDEr, and ROUGE_L scores.

### `preprocess_data.py`
This file is used to generate our vocabulary from the training data. It requires that you have the training data downloaded.

### `get_scores.py`
This file computes the BLEU, CIDEr, and ROUGE_L scores. It requires that you download this [repo](https://github.com/tylin/coco-caption) and place it inside.

### Jobs
This folder contains all of our Blue Waters' jobs for running these scripts
