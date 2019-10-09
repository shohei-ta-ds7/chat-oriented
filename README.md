# ncm
Neural Conversational Models Repository

## Requirements
- pipenv
- CUDA 9.0
- cuDNN 7.5.1

## Usage

- Launching pipenv shell.

```sh
pipenv shell
```

### Preprocessing

- Downloading, unzipping, and preprocessing DailyDialog (http://yanran.li/dailydialog)

```sh
./dl_dailydialog.sh
```

### Training

- Training model

```sh
python ./main.py --model_arc MODEL_ARCHITECTURE --model_pre MODEL_PREFIX
```

"MODEL_ARCHITECTURE" is an NCM architecture such as HRED.

"MODEL_PREFIX" is a prefix of checkpoints such as "./pkl/ncm".

To print arguments, type as follows.

```sh
python ./main.py --help
```

### Evaluation

- Beam Search Decoding

```sh
python ./main.py --mode inference -c CHECKPOINT_PATH
```

"CHECKPOINT_PATH" is a checkpoint path such as "./pkl/ncm.tar".

An inference pickle file, such as "./pkl/inf.ncm.tar", will be outputted.

- Automatic Evaluation

```sh
python ./auto_eval.py -i INFERENCE_PICKLE_FILE_LIST
```

"INFERENCE_PICKLE_FILE_LIST" is a file list such as "./pkl/inf.encdec.tar ./pkl/inf.hred.tar".

- Chatting with NCM

```sh
python ./ncm/main.py --mode chat -c CHECKPOINT_PATH
```

To quit, input ":q" or ":quit".

## Licence

[MIT](https://github.com/tcnksm/tool/blob/master/LICENCE)

## Author

[Shohei Tanaka](https://github.com/Tanasho0928)
