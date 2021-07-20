!/bin/bash

# script created by Professor Bruno Martins

for i in `seq 1 11`;
do
  wget --header="Accept: text/html" --user-agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10.8; rv:21.0) Gecko/20100101 Firefox/21.0" "https://www.dafont.com/bitmap.php?page=${i}&fpp=100"
done
cat bitmap.* | grep -a --text "dl.dafont.com/dl/" | sed s/"\(dl\.dafont\.com[^\"]*\)"/"\1\n"/g | sed s/".*\(dl\.dafont\.com[^\"]*\)"/"\1"/ | sed s/".*\(dl\.dafont\.com[^\"]*\)"/"http:\/\/\1"/ >aux.sh
rm -rf bitmap*
iconv -f latin1 -t utf-8 aux.sh | sed s/".*http\(.*\)"/"wget --header=\"Accept: text\/html\" --user-agent=\"Mozilla\/5.0 (Macintosh; Intel Mac OS X 10.8; rv:21.0) Gecko\/20100101 Firefox\/21.0\" \"http\1\""/ | head -n -1 | grep "^wget" | grep -v "paypal" >get-data.sh
sh get-data.sh
rm aux.sh
rename 's/index.html.f=(.*)/$1.zip/' *

