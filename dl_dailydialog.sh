#!/usr/bin/zsh

wget "http://yanran.li/files/ijcnlp_dailydialog.zip"
unzip ijcnlp_dailydialog.zip
cd ijcnlp_dailydialog
unzip train.zip
unzip validation.zip
unzip test.zip
cd ..
