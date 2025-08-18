```text
    ___         __     _    ___     
   /   |  _____/ /____| |  / (_)____
  / /| | / ___/ __/ _ \ | / / / ___/
 / ___ |/ /  / /_/  __/ |/ / (__  ) 
/_/  |_/_/   \__/\___/|___/_/____/  
```

[![GitHub Pages](https://github.com/s9latimm/artevis/actions/workflows/github-pages.yml/badge.svg)]()

## Demo

<img src="web/art_0001.png" height="256"/> <img src="web/art_0002.png" height="256"/> <img src="web/art_0003.png" height="256"/>

## Setup

### Virtual Environment

#### Windows (Powershell)

```shell
$ python -m venv .venv
$ .\.activate.ps1
```

#### Linux

```shell
$ python -m venv .venv
$ source ./venv/bin/activate
```

### Dependencies

```shell
$ python -m pip install --upgrade pip
$ python -m pip install wheel
$ python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
$ python -m pip install -r requirements.txt
```

## Usage

```shell
$ python -m src.artevis
```

```text
usage: artevis [-h] [--cache] [--device {cpu,cuda}] [--steps <steps>] [--fps fps] [--size <size>]
               [--project <project>]

options:
  -h, --help           show this help message and exit
  --cache              Enable caching / storage of models in output folder
  --device {cpu,cuda}  device used for training (default: cpu)
  --steps <steps>      number of training steps (default: 5e+04)
  --fps fps            FPS (default: 6e+00)
  --size <size>        size of trained image (default: 256)
  --project <project>  choose project (default: mona)
```
