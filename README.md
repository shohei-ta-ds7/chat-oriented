# ncm
Neural Conversational Models Repository

## Requirements
- pipenv
- CUDA 9.0
- cuDNN 7.5.1
- MTEval Toolkit

## Usage

- Launching pipenv shell.

```sh
pipenv shell
```

### Preprocessing

- Downloading, unzipping, and preprocessing DailyDialog (http://yanran.li/dailydialog)

```sh
dl_dailydialog.sh
```

### Training neural conversational models

- Training model

```sh
python ./ncm/main.py --model_arc MODEL_ARCHITECTURE --model_pre MODEL_PREFIX
```

"MODEL_ARCHITECTURE" is an NCM architecture such as HRED.

"MODEL_PREFIX" is a prefix of checkpoints such as "./pkl/ncm".

To print arguments, type as follows.

```sh
python ./ncm/main.py --help
```

- Beam Search Decoding

```sh
python ./ncm/main.py --mode inference -c CHECKPOINT_PATH
```

"CHECKPOINT_PATH" is a checkpoint path such as "./pkl/ncm.tar".

An inference pickle file, such as "./pkl/inf.ncm.tar", will be outputted.

- Chatting with NCM

```sh
python ./ncm/main.py --mode chat -c CHECKPOINT_PATH
```

To quit, input ":q" or ":quit".

## Licence

[MIT](https://github.com/tcnksm/tool/blob/master/LICENCE)

## Author

[Shohei Tanaka](https://github.com/Tanasho0928)
