# Table of Contents
* [About the Project](#about-the-project)
* [Getting Started](#getting-started)
  * [Install Python (Version 3.7 or higher)](#install-python)
  * [For MacOS Only: Install GCC and Xcode](#install-python)
  * [Install required dependencies for textract](#install-required-dependencies-for-textract)
  * [Install pipenv](#install-pipenv)
  * [Install AcX and Python dependencies](#install-acx-and-python-dependencies)
  * [Setting PYTHONPATH](#setting-pythonpath)
  * [Optional Steps](#optional-steps)
    * [Install Java](#install-java)
    * [Install audio/video dependencies for textract](#install-textract-audio)
    * [Install deep learning dependencies](#install-deep-learning-dependencies)
    * [Install dependencies for MadDog system and in- out- expanders](#install-dependencies-for-maddog-system-and-in-out-expanders)
    * [Install dependencies for LUKE out-expander](#install-dependencies-for-luke-out-expander)
* [Running the Benchmarks](#running-the-benchmarks)
* [Wikipedia Processing](#wikipedia-processing)
  * [Downloading and extracting Wikidump](#downloading-and-extracting-wikidump)
  * [Parsing the Wikidump](#parsing-the-wikidump)
* [Running the WebServer](#running-the-webserver)
* [More Information](#more-information)
# About The Project

AcX is an Acronym eXpander that synthesizes and extends the best of previous work on acronym expansion.

In addition, there are byproducts that future researchers can either extend or use as a basis of comparison.
1. Three benchmarks to evaluate in- out-expansion techniques and end-to-end systems.
2. A human-annotated dataset.

<!--I copy and pasted these from the paper as placeholders, edit them as you see fit -->

# Getting Started

To set up your environment for AcX follow the steps below.

## Install Python (Version 3.7 or higher) <a name="install-python"></a>

You can find information how to install Python at: https://www.python.org/.


## For MacOS Only: Install GCC and xcode <a name="install-for-mac"></a>

	Install [Xcode from the Apple store](https://apps.apple.com/us/app/xcode/)
	Install GCC
	
	```
	$ brew install gcc
	```
	
	> Note: You should have the [homebrew](https://brew.sh/) package manager installed.
## Install required dependencies for textract

* Ubuntu/Debian-based Linux distros:

    ```sh
    $ sudo apt install python-dev python3-dev python3-pip libxml2-dev libxslt1-dev antiword unrtf poppler-utils pstotext tesseract-ocr
    ```

* Arch-based Linux distros:

    ```sh
    $ sudo pacman -S base-devel python-pip libxml2 libxslt antiword unrtf poppler pstotext tesseract
    ```

* OpenSuse 15.0:

    ```sh
    $ sudo zypper install python-dev libxml2-dev libxslt1-dev antiword unrtf poppler-utils pstotext tesseract-ocr
    ```

* OpenSuse 15.1/15.2:

    ```sh
    $ sudo zypper install python-devel python3-devel libxml2-devel libxslt-devel antiword unrtf pstotext tesseract-ocr
    ```

* MacOS:

    ```sh
    $ brew install --cask xquartz
    $ brew install poppler antiword unrtf tesseract
    ```

    > Note: You should have the [homebrew](https://brew.sh/) package manager installed.

> Note: For other Operating Systems not listed here please visit the official [textract](https://textract.readthedocs.io/en/latest/installation.html) documentation.
 
## Install pipenv

You can install pipenv by just using pip:

```sh
$ pip install pipenv
```

> Note: For other installation methods please refer to [pipenv](https://pipenv.pypa.io/en/latest/install/#installing-pipenv) official documentation.

## Install AcX and Python dependencies
Install AcX by cloning its Github repository and install the dependencies using pipenv.
Large data files can be downloaded from its temporary location [here](https://amsuni-my.sharepoint.com/:f:/g/personal/j_p_pereira_uva_nl/EnJb8PNwvpZIjUoQSfSDu64BSayIrC-W-WQHecoMcDDYWA?e=UsT2ef)

Type the following in your terminal:
```sh
$ git clone git@github.com:joaolmpereira/acx-acronym-expander.git
$ cd acx-acronym-expander
$ pipenv install --dev
```

<!--
```sh
$ cd <project root folder>
$ pipenv install --dev
```
-->

Additional [NLTK](https://www.nltk.org/index.html) data packages need to be installed. Type the following in your terminal:

```sh
$ pipenv shell
$ pip install git+https://github.com/Jwink3101/sqlitedict.git@328d97146fcc4f03bd3ac817f26a9fb5d5d8e354
$ python -m spacy download en_core_web_sm
$ python
>>> import nltk
>>> nltk.download('punkt')
>>> nltk.download('stopwords')
>>> exit()
$ exit
```


## Setting PYTHONPATH

Additionaly the environment variable PYTHONPATH should be set so that custom modules can be imported. Type the following in your terminal (assuming that you are in the root folder of the project):

```sh
$ cd <project root folder>
$ touch .env
$ echo "PYTHONPATH=\${PYTHONPATH}:\${PWD}/acrodisam" >> .env
```

Pipenv will automatically load the .env file when you do:

```sh
$ pipenv shell
```

## Optional steps

### Install Java

If you want to run the original Java implementation of the Schwartz and Hearst algorithm for acronym and expansion extraction you need to install Java.

You can find information how to install Java at: https://www.oracle.com/java/
> Note: For MacOS users additional steps may be needed such as setting the JAVA_HOME environment variable and adding it to your PATH environment variable. Please refer to the offical java installation instructions of your MacOS version.



### Install audio/video dependencies for textract <a name="install-textract-audio"></a>

Textract has the ability to process audio/video files. If you want to process such files you can install these dependencies:


* Ubuntu/Debian-based Linux distros:

    ```sh
    $ sudo apt flac ffmpeg lame libmad0 libsox-fmt-mp3 sox libjpeg-dev swig libpulse-dev
    ```

* Arch-based Linux distros:

    ```sh
    $ sudo pacman -S flac ffmpeg lame libmad libid3tag twolame sox libjpeg-turbo swig libpulse
    ```

* OpenSuse 15.0:

    ```sh
    $ sudo zypper install flac ffmpeg lame libmad0 libsox-fmt-mp3 sox libjpeg-dev swig libpulse-dev libpulse-devel
    ```

* OpenSuse 15.1/15.2:

    ```sh
    $ sudo zypper install flac ffmpeg lame libmad0 sox libjpeg-devel swig libpulse-devel
    ```

* MacOS:

    ```sh
    $ brew install swig
    ```


### Install deep learning dependencies

Some deep learning algorithms require specific python packages. If you want to use these algorithms type the following in your terminal (you may do this at any time):

```sh
$ pipenv shell
$ pip install tensorflow
$ exit
```

### Install dependencies for MadDog system and in- out- expanders

>Please be aware that MadDog has a different license than ours.

If you want to use MadDog system or their in- out- expanders, type the following in your terminal:

```sh
$ pipenv shell
$ pip install -e git+https://github.com/amirveyseh/MadDog.git#egg=prototype
$ exit
```

### Install dependencies for LUKE out-expander
LUKE code needs to be manually imported and added to the Python path.
1. Download LUKE directly from its repository at https://github.com/studio-ousia/luke
2. Unpack and place it in our root folder
3. Update .env file to add LUKE folder to the Python path, e.g., type the following in your terminal:

```sh
$ cd <project root folder>
$ touch .env
$ echo "PYTHONPATH=\${PYTHONPATH}:\${PWD}/acrodisam:\${PWD}/luke" >> .env
```

4. Download LUKE pre trained model from https://drive.google.com/file/d/1BTf9XM83tWrq9VOXqj9fXlGm2mP5DNRF/view?usp=sharing
5. Place luke_large_ed.tar.gz into  <project root folder>/data/PreTrainedModels/LUKE folder

# Running the Benchmarks
In [acrodisam/benchmarkers/README.md](acrodisam/benchmarkers/README.md), we provide the instructions to run the benchmarks for in-expansion, out-expansion and end-to-end systems described in the paper submitted to VLDB22 entitled 
AcX: system, techniques, and experiments for Acronym eXpansion.


# Web Application Usage Example

This section presents an example of how to use the web application for AcX. 

In order for AcX to work you need to use a datasource. AcX can use different datasources varying in language and domain. Below is an example of how to use Wikipedia as a datasource.

## Wikipedia Processing

If you choose to use Wikipedia as your data source, there are a few extra steps you have to follow.

### Downloading and extracting Wikidump

Start by creating two folders (assuming that you are in the root folder of the project):

```sh
$ mkdir wikiextractor
$ mkdir acrodisam_app/data/FullWikipedia
```

Download the Wikidump from https://dumps.wikimedia.org/XXwiki/latest/XXwiki-latest-pages-articles.xml.bz2

> Note: Replace the XX with the code of the desired language (e.g. 'en' for English or 'pt' for Portuguese)

Place the file into the wikiextractor folder. Then type the following into your terminal (assuming that you are in the root folder of the project):

```sh
$ cd acrodisam_app
$ pipenv shell
$ python -m wikiextractor.WikiExtractor ../wikiextractor/XXwiki-latest-pages-articles.xml.bz2 -o data/FullWikipedia/
```

### Parsing the Wikidump

To parse the wikidump type the following into your terminal:

```sh
$ python acrodisam/DatasetParsers/FullWikipedia.py
```

After the script runs you can exit the python virtual environment by typing:

```sh
$ exit
```

## Running the Web Server

To run the webserver simply type into your terminal (assuming that you are in the root folder of the project):
 
```sh
$ cd acrodisam_app
$ pipenv shell
$ python acrodisam/web_app.py
```

The webserver is available at http://0.0.0.0:5000/, simply follow the directions given to input the article you want to extract acronyms from.

<!--- # More Information --->

 
