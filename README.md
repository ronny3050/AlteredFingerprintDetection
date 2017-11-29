# Altered Fingerprint Detection

This project uses the TF-slim library from Tensorflow to learn altered fingerprint detection.

## Requirements
* Tensorflow
* Altered Fingerprint Dataset (train and test folder)

## Table of contents

<a href='#Data'>Preparing the datasets</a><br>
<a href='#Training'>Training</a><br>
<a href='#Testing'>Testing</a><br>
<a href='#Troubleshooting'>Troubleshooting</a><br>

# Preparing the datasets
<a id='Data'></a>

The altered fingerprint dataset must be converted into TF-records which is used by the TF-slim library for very fast training.

## Converting dataset to TFRecord format

Assuming, `train` and `test` folders are present in the root folder and both folders contain samples for `live` and `altered` fingerprint images in subfolders, run the following command:

```shell
bash convertImageToTFRecord.sh
```

When the script finishes you will find several TFRecord files created:

```shell
$ ls TF
fingerprint_train_00000-of-00004.tfrecord
fingerprint_train_00001-of-00004.tfrecord
fingerprint_train_00002-of-00004.tfrecord
fingerprint_train_00003-of-00004.tfrecord
```

These represent the training data, sharded over 4 files.
We keep the testing data as raw images and do not convert them into tfrecords.


# Training a model from scratch.
<a id='Training'></a>

We provide an easy way to train a model from scratch using the altered TF-Slim dataset.
Simply run the following command:

```shell
bash train.sh
```

### TensorBoard

To visualize the losses and other metrics during training, you can use
[TensorBoard](https://github.com/tensorflow/tensorboard)
by running the command below.

```shell
tensorboard --logdir=train_logs
```

Once TensorBoard is running, navigate your web browser to http://localhost:6006.


# Evaluating performance of a model
<a id='Testing'></a>

W.I.P.

# Troubleshooting
<a id='Troubleshooting'></a>

#### The model runs out of CPU memory.

See
[Model Runs out of CPU memory](https://github.com/tensorflow/models/tree/master/research/inception#the-model-runs-out-of-cpu-memory).

#### The model runs out of GPU memory.

See
[Adjusting Memory Demands](https://github.com/tensorflow/models/tree/master/research/inception#adjusting-memory-demands).

#### The model training results in NaN's.

See
[Model Resulting in NaNs](https://github.com/tensorflow/models/tree/master/research/inception#the-model-training-results-in-nans).

#### The ResNet and VGG Models have 1000 classes but the ImageNet dataset has 1001

The ImageNet dataset provided has an empty background class which can be used
to fine-tune the model to other tasks. If you try training or fine-tuning the
VGG or ResNet models using the ImageNet dataset, you might encounter the
following error:

```bash
InvalidArgumentError: Assign requires shapes of both tensors to match. lhs shape= [1001] rhs shape= [1000]
```
This is due to the fact that the VGG and ResNet V1 final layers have only 1000
outputs rather than 1001.

To fix this issue, you can set the `--labels_offset=1` flag. This results in
the ImageNet labels being shifted down by one:


#### I wish to train a model with a different image size.

The preprocessing functions all take `height` and `width` as parameters. You
can change the default values using the following snippet:

```python
image_preprocessing_fn = preprocessing_factory.get_preprocessing(
    preprocessing_name,
    height=MY_NEW_HEIGHT,
    width=MY_NEW_WIDTH,
    is_training=True)
```

#### What hardware specification are these hyper-parameters targeted for?

See
[Hardware Specifications](https://github.com/tensorflow/models/tree/master/research/inception#what-hardware-specification-are-these-hyper-parameters-targeted-for).
