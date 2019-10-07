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

- Downloading & unzipping DailyDialog (http://yanran.li/dailydialog)

```sh
dl_dailydialog.sh
```

- Loading data

```sh
python load_dailydialog.py -i ijcnlp_dailydialog/train/ -o data
python load_dailydialog.py -i ijcnlp_dailydialog/validation/ -o data
python load_dailydialog.py -i ijcnlp_dailydialog/validation/ -o data
```

The first command outputs a vocabulary file.

### Training neural conversational models

- Training model

```sh
python ./ncm/main.py --model_arc MODEL_ARCHITECTURE --model_pre MODEL_PREFIX
```

"MODEL_ARCHITECTURE" is an NCM architecture such as HRED.

"MODEL_PREFIX" is a prefix of checkpoints such as "./pkl/ncm."

To print arguments, type as follows.

```sh
python ./ncm/main.py --help
```

- Beam Search Decoding

```sh
python ./ncm/main.py --inference -c CHECKPOINT_PATH -i INF_PICKLE
```

"--inference" means inference mode.

"CHECKPOINT_PATH" is a checkpoint path such as "./pkl/ncm_1.tar."

"INF_PICKLE" is a inference pickle path such as "./pkl/inf.pkl."

## Licence

[MIT](https://github.com/tcnksm/tool/blob/master/LICENCE)

## Author

[Shohei Tanaka](https://github.com/Tanasho0928)
