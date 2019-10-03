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

MODEL_ARCHITECTURE: NCM architecture such as HRED.
MODEL_PREFIX: Output model prefix such as "model_log/hred."
To print arguments, type as follows.

```sh
python ./ncm/main.py --help
```

- Beam Search Decoding

```sh
python ./ncm/main.py --inference -c CHECKPOINT_PATH -o OUTPUT_PICKLE
```

## Licence

[MIT](https://github.com/tcnksm/tool/blob/master/LICENCE)

## Author

[Shohei Tanaka](https://github.com/Tanasho0928)
