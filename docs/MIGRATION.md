# Migration notes: notebooks to the `dlresnet` package

## 1. Overview

The repository originally contained two notebooks: `ant-bee-classification.ipynb` (binary ant/bee classification with scratch, transfer and "revamp" ResNet-50 variants) and `bird-species-classification.ipynb` (525-way bird species classification from scratch). Both were written for TensorFlow 2.13 `tf.keras`; they are replaced by the `dlresnet` package (TensorFlow 2.21, Keras 3.15, NumPy 2), driven by a CLI and YAML configurations. This document records the API changes, the defects found in the original code, and the results the notebooks reported; the notebooks themselves are kept unmodified as the historical record. The package description follows the specification and may be incomplete while implementation is in progress.

## 2. API migration table

| Old (notebook, tf.keras 2.13) | New (Keras 3.15) | Why |
|---|---|---|
| `from tensorflow.keras import Model`; `from tensorflow.keras.layers import ...` | `import keras`; `keras.layers.*`, `keras.Model` | Keras 3 is the reference API; `tf.keras` is only an alias of it in TF 2.16 and later, and the explicit namespace removes that dependency. |
| `ImageDataGenerator(...).flow_from_directory(directory=..., target_size=(224,224), class_mode='binary')` | `keras.utils.image_dataset_from_directory(directory, image_size=(224,224), label_mode='binary')` composed with `tf.data` (`cache`, `shuffle`, `prefetch`) | `ImageDataGenerator` is a legacy Python iterator, not a `tf.data` pipeline, and is not graph-compatible; the `label_mode` argument replaces `class_mode`. Class names are exposed as `.class_names`, which the new code saves as `class_names.json`. |
| `ImageDataGenerator(rotation_range=20)` | `keras.layers.RandomRotation(factor=20/360)` | Keras augmentation layers run inside the model graph and are active only in training. `factor` is a fraction of a full turn, so 20 degrees corresponds to 0.0556. |
| `width_shift_range=0.2, height_shift_range=0.2` | `keras.layers.RandomTranslation(height_factor=0.2, width_factor=0.2)` | Same semantics as fractions of the image size. |
| `horizontal_flip=True` | `keras.layers.RandomFlip("horizontal")` | Direct replacement. |
| `zoom_range=0.05` | `keras.layers.RandomZoom(0.05)` | Direct replacement. |
| `shear_range=0.05` (bird notebook) | No Keras preprocessing layer | Shear has no built-in layer; it is dropped or reimplemented explicitly, and the difference should be noted when comparing results. |
| `preprocessing_function=preprocess_input` (`tensorflow.keras.applications.resnet50`) | In-model `CaffePreprocess` layer | Preprocessing is saved with the model, so inference accepts raw images. The notebooks applied it in the generator, which is outside the model and easy to mismatch at prediction time (bug 6). |
| `rescale=1/255.0` (bird notebook) | `CaffePreprocess` (no rescaling) | A single preprocessing path is used for all runs; the bird pipeline previously differed from the ImageNet convention used by the transfer path. |
| `SGD(lr=0.001, momentum=0.9)` | `keras.optimizers.SGD(learning_rate=..., momentum=0.9)` | `lr` is not honored in the old code: the history shows 0.01, the optimizer default (bug 3). The learning rate now comes from the configuration. |
| `tf.keras.optimizers.legacy.Adam(learning_rate=lr, decay=lr/(epochs/0.5))` | `keras.optimizers.Adam(learning_rate=<LearningRateSchedule>)` | Legacy optimizers from the Keras 2 namespace and the `decay` argument are removed in Keras 3; decay is expressed as a schedule object. |
| `keras.callbacks.LearningRateScheduler(scheduler)` with `return lr * 0.1` | `StepDecay` (`keras.optimizers.schedules.LearningRateSchedule` subclass, stateless) | The callback reads and overwrites mutable optimizer state, which compounded across `fit()` calls (bug 4). A stateless schedule is a function of the global step, so its value is reproducible and cannot be decayed twice. |
| No schedule (constant rate) | `WarmupCosine` (`LearningRateSchedule`) | New option; linear warm-up followed by cosine decay, in the spirit of Goyal et al. (2017). |
| `model.save('ant-bee-25epochs.h5')` (HDF5, legacy) | `model.save('model.keras')` | The `.keras` format is native to Keras 3, stores architecture, configuration and optimizer state, and resolves registered custom layers. The notebooks produced the legacy-format warning shown in their outputs. |
| `load_model(path, custom_objects={'Block': Block})` | `keras.models.load_model(path)` with `BottleneckBlock` registered via `keras.saving.register_keras_serializable` | Custom classes are resolved from the serialization registry, so callers no longer pass a mapping of names. |
| `class Block(Model)` with `__init__` and `call` | `class BottleneckBlock(keras.layers.Layer)` with `get_config` | A layer composes inside the functional graph and serializes its configuration; a nested `Model` subclass does not serialize cleanly with the `.keras` format. |
| `Input(tensor=input_tensor, shape=input_shape)` and `backend.is_keras_tensor(input_tensor)` | Removed; `keras.Input(shape=...)` only | Both are TensorFlow 1 idioms. `backend` is never imported in `ant-bee-classification.ipynb`, so the `input_tensor` branch would raise a `NameError`; the default path never reaches it. |
| `strategy = tf.distribute.MirroredStrategy()` with `with strategy.scope():` | Kept, optional, enabled through the configuration | Single-GPU runs do not require it; it is a deployment choice, not part of the model definition. |
| `from keras.utils import load_img, img_to_array` | Unchanged | Still valid in Keras 3. |
| `history.history['acc']`, `['val_acc']` | Keys produced by the new `compile` metrics (`accuracy`) | Metric naming follows the Keras 3 metric names; plotting code must use the keys reported by the new `History`. |

## 3. Bug fixes

1. **(a) Bottleneck widths.** The `Block` class in the ant/bee notebook takes `input_channels` as the width of its first two convolutions:

    ```python
    self.conv1=Conv2D(input_channels,kernel_size=1,strides=identity_strides)
    self.conv2=Conv2D(input_channels,kernel_size=3,strides=1,padding='same')
    self.conv3=Conv2D(output_channels,kernel_size=1,strides=1,padding='same')
    ```

    and the stage builders pass the previous stage's output as `input_channels`, for example `X = Block(256, 512, use_conv=True, identity_strides=2)(X)` in stage 2 (cell 9). The bottleneck width is therefore the output of the previous stage, not one quarter of the current output: stage 2 uses 256 instead of 128, stage 3 uses 512 instead of 256, and stage 4 uses 1024 instead of 512. Only the first block of stage 1, `Block(64, 256)`, is correct; its identity blocks `Block(256, 256)` already use width 256 instead of 64. The notebook's ResNet-50 with 64x64 input and 100 classes (cell 10) reports `Total params: 179820772`. The canonical ResNet-50 (torchvision convention, bias-free convolutions) has 25,557,032 parameters at 1000 classes, equivalent to 23,712,932 at 100 classes and 23,512,130 at 2 classes; the difference from the notebook is an order of magnitude larger than the bias terms can explain. The revamp model (`ResNet50_2`, cell 28) uses correct widths, but its stage 3 contains three blocks instead of four and it ends with `AveragePooling2D((2, 2))` and `Flatten` instead of global average pooling. Its logged summary reports 23,342,210 parameters at 2 classes.

2. **(b) ReLU before the residual addition.** The residual branch ends with a ReLU, so the sum is taken with a non-negative residual:

    ```python
    Y=self.conv3(Y)
    Y = self.bn3(Y)
    Y= relu(Y)
    ...
    Y+=X
    Y=relu(Y)
    ```

    In the ResNet formulation, the residual branch ends with batch normalization and the activation is applied once, after the addition, as `relu(F(x) + x)`. The extra activation clips the residual branch and removes the identity-mapping property that the architecture relies on. The same code appears in `Block_2` (cell 26).

3. **(c) Learning-rate keyword ignored.** The transfer and revamp cells compile with:

    ```python
    model.compile(optimizer=SGD(lr=0.001, momentum=0.9), ...)
    ```

    The recorded history reports `lr: 0.0100` from the first epoch onward, which is the `SGD` default. The intended rate of 0.001 was never applied. The new configuration sets the rate explicitly as `learning_rate`.

4. **(d) Scheduler compounding across a second `fit()`.** The scheduler multiplies the optimizer's current rate:

    ```python
    def scheduler(epoch, lr):
        if epoch % 7 == 0 and epoch:
            return lr * 0.1
        return lr
    lr_callback = keras.callbacks.LearningRateScheduler(scheduler)
    ```

    The revamp notebook calls `model.fit` twice with the same callback (cells 32 and 34). The optimizer keeps its state between calls and the callback's epoch counter restarts at zero, so the second run starts at the rate left by the first (`1.0000e-05`, cell 34) and decays again at epochs 7, 14 and 21. The recorded rate reaches `1.0000e-08` by epoch 24. `StepDecay` removes the mutable state: the rate at a given global step is always the same value.

5. **(e) Model and data mismatch.** The scratch model is built as:

    ```python
    model = ResNet50(input_shape=(64,64, 3), classes=100)
    ```

    (cell 10), while the data generators produce 224x224 images with two classes in binary mode (cells 6 and 14). The transfer and revamp models use 224x224 inputs with two outputs. Keras 2.13 functional models only log a warning when called on an input shape that differs from the declared one, and every layer in this network (convolutions, pooling, global average pooling) accepts any spatial size, so training still ran: on 224x224 images, with a 100-way softmax of which only indices 0 and 1 are ever targeted. The declared shape and class count are simply wrong for the data. The new package derives both from the dataset.

6. **(f) Prediction cell.** The prediction code is:

    ```python
    img = load_img('test/WILLOW PTARMIGAN/5.jpg', target_size=(224, 224))
    ...
    img = img/255.0
    ...
    model = load_model('models/ant-bee-25epochs.h5', custom_objects={'Block': Block})
    ```

    It has three faults. The image belongs to the bird dataset but is passed to the two-class ant/bee model, which can only return ant or bee. The preprocessing divides by 255, whereas the training path used `preprocess_input` (Caffe-style BGR mean subtraction). The model file is the 252 MB legacy file discussed in (j), which is probably not the ant/bee architecture.

7. **(g) Bird source does not match its logged summary.** The bird notebook defines `ResNet50(input_shape=(224,224,3), classes=525)` (cell 9) with the same `Block` definition as the ant/bee notebook. Its logged summary reports `Total params: 21015693` with a final `Dense (None, 525)` layer of 1,075,725 parameters. Using the source's backbone (179,820,772 parameters at 100 classes, from bug (a)) with a 525-way head gives 179,820,772 - 204,900 + 1,075,725 = 180,691,597 parameters. The logged run was therefore not produced by the source as saved.

8. **(h) Test generator never evaluated.** The bird notebook creates a test generator over 2,625 images and 525 classes (cell 5):

    ```python
    test_generator = test_datagen.flow_from_directory(
      directory=test_directory, ..., class_mode=None, shuffle=False, seed=42)
    ```

    Training (cell 11) is followed only by a commented-out save (cell 12). No cell calls `evaluate` or `predict` on the test split, so no test-set accuracy exists for the bird model. The new `evaluate` command reports test metrics explicitly.

9. **(i) Stale `.gitattributes` entry.** `.gitattributes` contains `models/bird_classification.keras filter=lfs diff=lfs merge=lfs -text`, but no such file is present under `models/` or in the Git index. The bird model is tracked only as `models/bird_classification.h5`. The entry should be removed when the new model files are added.

10. **(j) Legacy model files.** `models/ant-bee-25epochs.h5` and `models/bird_classification.h5` are tracked with Git LFS. The checkout holds 134-byte pointer files, each recording `size 252477020`. That size is consistent with about 21.0 M parameters at 4 bytes per parameter times 3 (weights plus the two Adam moment slots that the legacy optimizer stores in HDF5). The two files have the same size but different object IDs (`f987544b...` and `069b6b1c...`), so their contents differ. None of the ant/bee architectures in the notebooks has about 21.0 M parameters (179.8 M for the scratch model, 23.3 M for the revamp, and about 23.5 M for the transfer model with two outputs). The count matches the logged bird summary (21,015,693), so `models/ant-bee-25epochs.h5` is likely a bird-architecture model, which would also explain the failure described in (f).

## 4. Results reported by the notebooks

Values are taken from the recorded outputs of the notebooks. They were obtained under the defects listed above and on the original splits, and are not reproduced by the `dlresnet` package.

| Run (notebook, cell) | Data | Epochs | Final train acc | Final val acc | Peak val acc | Notes |
|---|---|---|---|---|---|---|
| Ant/bee scratch (ant-bee, cell 14) | 244 train / 153 val, 2 classes | 25 | 0.660 | 0.608 (about 61%) | 0.647 | Input mismatch and bottleneck defects (bugs 1, 5). |
| Ant/bee transfer (ant-bee, cell 19) | 244 train / 153 val, 2 classes | 25 | 0.992 | 0.941 (about 94%) | 0.954 | ImageNet backbone frozen; effective rate 0.01 (bug 3). |
| Ant/bee revamp (ant-bee, cells 32 and 34) | 244 train / 153 val, 2 classes | 25 + 25 | 0.557 (cell 32); 0.521 (cell 34) | 0.523 (cell 32); 0.536 (cell 34) (about 52-54%) | 0.569 | Scheduler compounding in the second run (bug 4). |
| Birds scratch (bird, cell 11) | 84,635 train / 2,625 val, 525 classes | 30 configured and logged | 0.980 (about 98%) | 0.801 (80.1%) | 0.842 (84.2%, epoch 25) | Test split never evaluated (bug 8); logged parameter count does not match source (bug 7). |

## 5. References

- He, K., Zhang, X., Ren, S., and Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR. https://arxiv.org/abs/1512.03385
- He, K., Zhang, X., Ren, S., and Sun, J. (2016). Identity Mappings in Deep Residual Networks. ECCV. https://arxiv.org/abs/1603.05027
- Goyal, P., Dollar, P., Girshick, R., Noordhuis, P., Wesolowski, L., Kyrola, A., Tulloch, A., Jia, Y., and He, K. (2017). Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour. https://arxiv.org/abs/1706.02677
- Keras 3 migration guide. https://keras.io/guides/migrating_to_keras_3/
- Keras ResNet implementation (source). https://github.com/keras-team/keras/blob/master/keras/src/applications/resnet.py
- torchvision ResNet implementation (source). https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
- Dive into Deep Learning, chapter on residual networks. https://d2l.ai/chapter_convolutional-modern/resnet.html
