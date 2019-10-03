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

## Licence

[MIT](https://github.com/tcnksm/tool/blob/master/LICENCE)

## Author

[Shohei Tanaka](https://github.com/Tanasho0928)
