```text
    ___         __     _    ___     
   /   |  _____/ /____| |  / (_)____
  / /| | / ___/ __/ _ \ | / / / ___/
 / ___ |/ /  / /_/  __/ |/ / (__  ) 
/_/  |_/_/   \__/\___/|___/_/____/  
```

## Demo

<img src="web/art_0001.png" height="256"/> <img src="web/art_0002.png" height="256"/> <img src="web/art_0003.png" height="256"/> 

<video src="https://github.com/user-attachments/assets/73beaf10-07dc-4148-a6e8-293cc9279ddd" controls preload></video>

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

