## imagenet-multiGPU.torchnet

Basic Torchnet code for Imagenet Classification

* datagen.lua : create t7 file from imagenet like directory
* train.lua : Main training file

Has:

* Class Balancing
* Training and testing
* Multithreaded dataloading
* MultiGPU support
* Fine tuning and training from scratch
* Data Augmentation wrapper

Doesn't have

* Code for resuming training from optimstate (but easy to add)

#### How to use:

1. Prepare imagenet type folder structure. (e.g. https://drive.google.com/file/d/0B7ZgIaKJsQhbU1c3OHNxNk8wRXc/view?usp=sharing)
2. Prepare its mean and std in a table like {mean = {} , std = {}} or use ILSVRC meanstd given.
3. Run `th utils/datagen.lua -s path_to_dataset -n name_of_dataset -m path_to_meanstd_file`
4. Prepare the model file.
5. Run `th train.lua -d path_to_data -c path_to_t7_cache -m path_to_model -s folder_to_save_logs_in`
6. Run `th train.lua -h` for more options.
