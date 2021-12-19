To setup the environment, do the following in order for a MacOS machine:
- `conda install -c conda-forge ffmpeg`
- `conda install pytorch torchvision torchaudio -c pytorch`
- `pip install pandas tqdm h5py sklearn seaborn tabulate soundfile opencv-python`
- `pip install mir_eval`
- `pip install pyyaml`

The ordering can resolve conflicts from the ffmpeg installation.

Loading the train dataset features into memory from the h5py file takes about 5 minutes.
Loading the validation features into memory takes about 30s.


