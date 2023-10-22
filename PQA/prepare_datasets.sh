mkdir data

cd data

#downloads the path query dataset
wget --no-check-certificate https://worksheets.codalab.org/rest/bundles/0xdb6b691c2907435b974850e8eb9a5fc2/contents/blob/ -O freebase_paths.tar.gz
wget --no-check-certificate https://worksheets.codalab.org/rest/bundles/0xf91669f6c6d74987808aeb79bf716bd0/contents/blob/ -O wordnet_paths.tar.gz

mkdir pathqueryWN && tar -zxvf wordnet_paths.tar.gz -C pathqueryWN
mkdir pathqueryFB && tar -zxvf freebase_paths.tar.gz -C pathqueryFB

cd ..

#Preprocess the dataset
python data_preprocess.py --task pathqueryFB --dir data/pathqueryFB 
python data_preprocess.py --task pathqueryWN --dir data/pathqueryWN