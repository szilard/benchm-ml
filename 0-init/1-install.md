
Initial install (April 2015) instructions below. Some parts of the benchmark have been
re-run with upgraded versions of the tools, [see here](1a-versions.txt).


## Ubuntu 14.04 LTS

#### R & xgboost
```bash
echo "deb http://cran.rstudio.com/bin/linux/ubuntu trusty/" > /etc/apt/sources.list.d/r.list
apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E084DAB9
apt-get update 
apt-get install r-base-dev libcurl4-openssl-dev libssl-dev

R --vanilla << EOF
install.packages(c("data.table","readr","randomForest","gbm","glmnet","ROCR","devtools"), repos="http://cran.rstudio.com")
devtools::install_github("dmlc/xgboost", subdir = "R-package")
EOF
```

#### Python
```bash
wget https://3230d63b5fc54e62148e-c95ac804525aac4b6dba79b00b39d1d3.ssl.cf1.rackcdn.com/Anaconda-2.2.0-Linux-x86_64.sh
bash Anaconda-2.2.0-Linux-x86_64.sh
```

#### VW
```bash
apt-get install libtool libboost1.55-*

wget https://github.com/JohnLangford/vowpal_wabbit/archive/7.10.tar.gz
./autogen.sh 
make
make install

export LD_LIBRARY_PATH=/usr/local/lib
/usr/local/bin/vw --version
```

#### Java
```bash
add-apt-repository ppa:webupd8team/java
apt-get update
apt-get install oracle-java7-installer
```

#### H2O
```bash
wget http://h2o-release.s3.amazonaws.com/h2o/rel-nunes/2/h2o-2.8.6.2.zip
unzip h2o-2.8.6.2.zip 
R --vanilla << EOF
install.packages(c("statmod","RCurl","rjson"), repos="http://cran.rstudio.com")  
EOF
R CMD INSTALL h2o-2.8.6.2/R/h2o_2.8.6.2.tar.gz
```

#### Spark
```bash
wget http://d3kbcqa49mib13.cloudfront.net/spark-1.3.0-bin-hadoop2.4.tgz
tar xzf spark-1.3.0-bin-hadoop2.4.tgz
```


#### mxnet

```
sudo apt-get install build-essential git libatlas-base-dev libopencv-dev libcurl4-openssl-dev

git clone --recursive https://github.com/dmlc/mxnet

cd mxnet
in file `make/config.mk` change:
 USE_CUDA = 1
 USE_CUDA_PATH = /usr/local/cuda-7.5
 USE_CUDNN = 1
make -j4

cd R-package
sudo Rscript -e "library(devtools); library(methods); options(repos=c(CRAN='https://cran.rstudio.com')); install_deps(dependencies = TRUE)"
cd ..
make rpkg
sudo R CMD INSTALL mxnet_0.7.tar.gz
```


