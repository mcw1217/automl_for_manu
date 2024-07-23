# AutoML for manufacturing
- This Experiment is a model that detects anomlies in untrained manufacturing data and is optimized through the AutoML framework

# Start Code
- First, download the MVTecAD dataset
- Second, Through the all_make_dataset.py code, the MVTecAD dataset is created in the learning format we propose ( Change it to suit your dataset path )
- Third, Start new_main.py

- Warning! This experiment consumes a lot of GPU memory, so use a graphics card of at least 24GB or more
 

# Use code
- new_main.py sets one class as test and uses the remaining classes for learning.
- main_third.py uses 15 normal and defective images from one class for training, and the remaining images are used for testing.
- Model is randomly selected through AutoML
- main_third_resnet.py is not random, but only selects hyperparameters randomly from the Resnet model.