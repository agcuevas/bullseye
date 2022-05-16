###############################################################################
# versiones
# cudnn-11.0-windows-x64-v8.0.4.30
# cuda_11.0.2_451.48_win10
# TensorRT-7.2.2.3.Windows10.x86_64.cuda-11.1.cudnn8.0
###############################################################################

# Configurar GPU
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


###############################################################################
#Shorcuts:
#ctrl+1 comentar
#ctrl+enter correr grupo
#f9, correr línea
# clear # borrar consola
# %reset #borra variables

###############################################################################
#instalar desde 0 (en este orden)
# anaconda (update 2.0)
# env test
# pkg ESTE ORDEN: spyder 5, jupyter+lab, numpy, matplotlib, pandas, pip tensorflow==2.4.0, sklearn (pip), seaborn
# pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
# TESTEAR con digitos.py (tf) y gan.py (torch) REINICIAR AL PASAR DE TF A TORCH

###############################################################################
#plan de trabajo
#funcionar/entender transformers
#transformers superr
#superr 3d
#mri superr 3d (contexto?)

###############################################################################
#tips
# pytorch,efficientnet_pytorch,,albumentations (conflicto con tensorflow)
#pip (se installan spyder recien abierto sin imports)
#pip install, show
#instalar kite
#preferences, interpreter--> environment (test), correr spider de anaconda env

#test pkg con digitos.py y gan.py

#bajar code snippets

#jupyter notebook --notebook-dir C:/work (en environment->test->open terminal)

#pb h5 ckpt, tensorflow/ pt pth torch

#set QT_SCALE_FACTOR=2
#cd  c:/anaconda3/scripts/
#anaconda-navigator

#conda list --export > c:/work/package-list.txt
#conda create -n myenv --file package-list.txt
#conda update --all
#python x.py para correr de prompt, %run x.py para correr desde consola ipython

#word2vec
#temas:agricultura, salud, finanzas, world data

###############################################################################
# Linux en windows
# https://docs.microsoft.com/en-us/windows/wsl/install-win10
# wsl2, ubuntu 20.04 (pass:ls7), instalar python y jupiter notebook, copiar archivos a linux
# desde windows cmd, comnado bash pasa a linux en misma carpeta
# https://medium.com/@sayanghosh_49221/jupyter-notebook-in-windows-subsystem-for-linux-wsl-f075f7ec8691
# instalar y actualizar jupyter
# sudo apt update && upgrade
# sudo apt install python3 python3-pip ipython3
# pip3 install jupyter
# pip install --upgrade jupyterhub
# pip install --upgrade --user nbconvert
# \\wsl$\Ubuntu-20.04\home\alex
# cp /mnt/c/work/inference_playground.ipynb /home/alex/inference_playground.ipynb


################################################################################
#docker
#docker agcuevas nF!
#docker run --rm -p 8888:8888 jupyter/scipy-notebook
#docker build -t docker-ml-model -f Dockerfile .

################################################################################
#probar carga de pretrained
#probar docker
#papers/videos
#transformers
#kaggle/code snippets

#google: best sites for remote jobs
################################################################################


