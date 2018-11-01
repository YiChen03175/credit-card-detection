# Introduction

This is a project for detecting the shape of credit card in natural images.

# Requirement

- Python >= 3.5.2
- [opencv-python](https://docs.opencv.org/3.0-beta/index.html#) 3.4.3.18
- [numpy](https://docs.scipy.org/doc/numpy/) 1.15.2
- [argparse](https://docs.python.org/3/howto/argparse.html) 1.4

# Environment

operating system: ubuntu 16.04 LTS  

Set a virtual environment using virtualenv

```sh
$ cd ~
$ virtualenv env_name
```

 Activate the virtualenv

```sh
$ source env_name/bin/activate
```
Go to the ImgurUpload folders

```sh
$ cd [ImgurUpload FOLDER PATH]
```

Install require package using [requirements.txt](./requirements.txt)

```sh
$ pip3 install -r requirements.txt
```
# How to Run

Run [CreditCardDetection.py](./CreditCardDetection.py) in this folder

```sh
$ python3 CreditCardDetetion.py
```
Default path is *./images/simple/simple1.jpg*, you can add your own path by *-input* as below

```sh
$ python3 CreditCardDetection.py -input ./images/contrast/contrast0.jpg
```
Run [ShowSamplePath.py](./ShowSamplePath.py), you can get all sample images path in [test folder](./images/test).

```sh
$ python3 ShowSamplePath.py
 
./images/test/position position2.jpg
./images/test/position position0.jpg
./images/test/position position3.jpg
./images/test/position position1.jpg
./images/test/simple simple0.jpg
./images/test/simple simple2.jpg
./images/test/simple simple1.jpg
./images/test/distance distance6.jpg
./images/test/distance distance3.jpg
./images/test/distance distance0.jpg
./images/test/distance distance2.jpg
./images/test/distance distance4.jpg
./images/test/distance distance1.jpg
./images/test/distance distance5.jpg
./images/test/light light3.jpg
./images/test/light light1.jpg
./images/test/light light2.jpg
./images/test/light light0.jpg
./images/test/contrast contrast3.jpg
./images/test/contrast contrast2.jpg
./images/test/contrast contrast1.jpg
./images/test/contrast contrast0.jpg
```