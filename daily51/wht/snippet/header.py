"""
################################################################
# colab GPU
!uname -a
!nvidia-smi --query-gpu=name,memory.total,memory.free,timestamp --format=csv,noheader
!wget https://ai5566.me/broker/download/broker_api/raw/ -O broker_api.py --quiet
!python3 -m broker_api download daily51 --dir daily51
# !make install_puppeteer_dependencies -C daily51 > /dev/null 2> /dev/null
!pip install -r daily51/requirements.txt > /dev/null 2> /dev/null
import torch; import numpy as np; import pandas as pd; import warnings
import matplotlib.pyplot as plt; import seaborn as sns;
import scipy; from scipy import stats; import statsmodels.api as sm; import sympy as sym
torch.set_printoptions(precision=6, sci_mode=False, linewidth=120, threshold=1000, edgeitems=3)
np.set_printoptions(precision=6, suppress=True, linewidth=120, threshold=1000, edgeitems=3)
pd.set_option('display.float_format', (lambda x: (f'%.{6}f')%x))
pd.set_option('display.max_rows', 1000); pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 810664); pd.set_option('display.width', 120)
from daily51.wht import *
from daily51.wht.cpu import *
import daily51.wht.web as qweb

################################################################
# local ipynb - import
import torch; import numpy as np; import pandas as pd; import warnings
import matplotlib.pyplot as plt; import seaborn as sns;
import scipy; from scipy import stats; import statsmodels.api as sm; import sympy as sym
torch.set_printoptions(precision=6, sci_mode=False, linewidth=120, threshold=1000, edgeitems=3)
np.set_printoptions(precision=6, suppress=True, linewidth=120, threshold=1000, edgeitems=3)
pd.set_option('display.float_format', (lambda x: (f'%.{6}f')%x))
pd.set_option('display.max_rows', 1000); pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 810664); pd.set_option('display.width', 810664)
from daily51.wht import *
from daily51.wht.cpu import *
import daily51.wht.web as qweb
import tqdm
tqdm.tqdm.get_lock().locks = []
import nest_asyncio
nest_asyncio.apply()

# local ipynb - homework
import homework
from homework import *
reload(homework)
pass


################################################################
# start django in ipynb
import os, sys, django
from django.apps import apps as django_apps

path_website = '/active/ai5566/website'
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ai5566web.settings')
os.environ.setdefault('DJANGO_ALLOW_ASYNC_UNSAFE', 'True')  # work around: https://docs.djangoproject.com/en/3.1/topics/async/#async-safety

if path_website not in sys.path:
    print(f"add path {path_website}")
    sys.path.append(path_website)

if not django_apps.ready:
    print("django setup")
    django.setup()
else:
    print("django ok")


################################################################
# "broker"
!uname -a
!nvidia-smi --query-gpu=name,memory.total --format=csv
!wget https://ai5566.me/broker/download/start/raw/ --no-cache --quiet -O - | sh > /dev/null 2> /dev/null


"""
