#!/bin/sh

python3 run.py --config=configs/mnist_epoch1.json
python3 run.py --config=configs/mnist_epoch10.json
python3 run.py --config=configs/mnist_client5.json
python3 run.py --config=configs/mnist_client15.json




#python3 run.py --config=configs/fashion_ethernet.json
#python3 run.py --config=configs/fashion_ethernet2.json
#python3 run.py --config=configs/fashion_ethernet3.json
#python3 run.py --config=configs/fashion_ethernet4.json

#python3 run.py --config=configs/fashion_wifi.json
#python3 run.py --config=configs/fashion_wifi2.json
#python3 run.py --config=configs/fashion_wifi3.json
#python3 run.py --config=configs/fashion_wifi4.json
