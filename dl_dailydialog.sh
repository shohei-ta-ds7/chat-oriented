#!/usr/bin/zsh

wget "http://yanran.li/files/ijcnlp_dailydialog.zip"
unzip ijcnlp_dailydialog.zip
cd ijcnlp_dailydialog
unzip train.zip
unzip validation.zip
unzip test.zip
cd ..
python load_dailydialog.py -i ijcnlp_dailydialog/train/ -o data
python load_dailydialog.py -i ijcnlp_dailydialog/validation/ -o data
python load_dailydialog.py -i ijcnlp_dailydialog/test/ -o data
