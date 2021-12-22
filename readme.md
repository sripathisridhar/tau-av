This repository is the final project in the CS677: Deep Learning class at NJIT, titled "A study in audio-visual scene 
classification".

The model is trained on the TAU Audio-Visual Urban Scenes 2021 dataset as per the DCASE 2021 Task 1B instructions.

### Environment setup

To setup the environment, do the following in order for a MacOS machine:
- `conda install -c conda-forge ffmpeg`
- `conda install pytorch torchvision torchaudio -c pytorch`
- `pip install pandas tqdm h5py sklearn seaborn tabulate soundfile opencv-python`
- `pip install mir_eval`
- `pip install pyyaml`

The ordering more easily resolves conflicts from the ffmpeg installation.\
For a linux or windows machine, install the pytorch libraries with your cuda version from [here](https://pytorch.org/get-started/locally/).

### Training the model

To train a model, run \
`srun python main.py --features_dir <path_to_features_directory>
--config <path_to_config_file>` \
Alternatively, you can modify the `gpu-train.sh` script to run the program on a cluster.

The program expects the features_directory to contain
audio_features and video_features sub-directories. 
This will automatically happen if you download the features from [this link](https://drive.google.com/file/d/1-LrwHwUBG8Rq1THJtRlyZcGQMSVsEqUo/view).

By default, the config file trains an audio+video model.
To train an audio only or video only model, modify `MODE` to `audio` or `video` in the config file. 

### Evaluation

For evaluation, run \
`srun python evaluate.py --features_path <path_to_features_directory>
--model_type audio_video`

Change the `--model_type` argument to `audio` or `video` to evaluate
an audio only or video only model.

The evaluate script expects the trained model weights to be in the models/ directory.


