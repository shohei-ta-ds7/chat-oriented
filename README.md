# ncm
Neural Conversational Models Repository

## Requirements
- pipenv
- CUDA 9.0
- cuDNN 7.5.1

## Usage

#### Launching pipenv shell.

```sh
pipenv shell
```

### Preprocessing

#### Downloading, unzipping, and preprocessing DailyDialog (http://yanran.li/dailydialog)

```sh
./dl_dailydialog.sh
```

### Training

#### Training model

```sh
python ./main.py --model_arc MODEL_ARCHITECTURE --model_pre MODEL_PREFIX
```

"MODEL_ARCHITECTURE" is an NCM architecture such as "hred".

Available models are as follows.

- encdec [Luong et al., 2015]
https://www.aclweb.org/anthology/D15-1166/

- hred [Serban et al., 2016]
https://arxiv.org/abs/1507.04808

- vhred [Serban et al., 2017]
https://arxiv.org/abs/1605.06069

- vhcr [Park et al., 2018]
https://arxiv.org/abs/1804.03424

"MODEL_PREFIX" is a prefix of checkpoints such as "./pkl/hred".

To print arguments, type as follows.

```sh
python ./main.py --help
```

### Evaluation

#### Beam Search Decoding

```sh
python ./main.py --mode inference -c CHECKPOINT_PATH
```

"CHECKPOINT_PATH" is a checkpoint path such as "./pkl/hred.tar".

An inference pickle file, such as "./pkl/inf.hred.tar", will be outputted.

#### Automatic Evaluation

```sh
python ./auto_eval.py -i INFERENCE_PICKLE_FILE_LIST
```

"INFERENCE_PICKLE_FILE_LIST" is a file list such as "./pkl/inf.encdec.tar ./pkl/inf.hred.tar".

#### Chatting with NCM

```sh
python ./main.py --mode chat -c CHECKPOINT_PATH
```

To quit, input ":q" or ":quit".

## Licence

[MIT](https://github.com/tcnksm/tool/blob/master/LICENCE)

## Author

[Shohei Tanaka](https://github.com/Tanasho0928)
