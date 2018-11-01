# Introduction

This is a project for detecting the shape of credit card in natural images.

# Requirement

- Python >= 3.5.2
- [opencv-python](https://docs.opencv.org/3.0-beta/index.html#) 3.4.3.18
- [numpy](https://docs.scipy.org/doc/numpy/) 1.15.2
- [argparse](https://docs.python.org/3/howto/argparse.html) 1.4

# Environment

operating system: ubuntu 16.04 LTS  

If you don't have virtualenv, [here](https://linuxhostsupport.com/blog/how-to-install-virtual-environment-on-ubuntu-16-04/) is the instruction for installing it.

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
Default path is *./images/angle/angle1.jpg*, you can add your own path by *-input* as below

```sh
$ python3 CreditCardDetection.py -input ./images/contrast/contrast1.jpg
```
