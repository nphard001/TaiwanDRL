"""
################################################################
# charts
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(rc={'figure.figsize': (11.7, 8.27)})
pd.plotting.register_matplotlib_converters()

demo = np.random.beta(0.5, 0.5, size=150)
sns.histplot(demo, kde=True)


################################################################
# printoptions
import torch; import numpy as np; import pandas as pd
torch.set_printoptions(precision=6, sci_mode=False, linewidth=120, threshold=1000, edgeitems=3)
np.set_printoptions(precision=6, suppress=True, linewidth=120, threshold=1000, edgeitems=3)
pd.set_option('display.float_format', (lambda x: (f'%.{6}f')%x))
pd.set_option('display.max_rows', 1000); pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 810664); pd.set_option('display.width', 810664)


################################################################
# debug
try:
    import ipdb
    with ipdb.launch_ipdb_on_exception():
        _main()
except (ImportError, ModuleNotFoundError):
    _main()


################################################################
# system locks
import tqdm
tqdm.tqdm.get_lock().locks = []
import nest_asyncio
nest_asyncio.apply()


################################################################
# colab async problem
import asyncio
from asyncio.unix_events import _UnixSelectorEventLoop
class LoopDoNotCheckJustRun(_UnixSelectorEventLoop):
    def _check_runnung(self):
        pass
loop = asyncio.get_event_loop()
loop.__class__ = LoopDoNotCheckJustRun


################################################################
# create GUI browser for testing
if "bsr" in dir():
    qweb.DelaySession.kill_chromium()
    del bsr
if "bsr" not in dir():
    bsr = qweb.DelaySession(headless=False)
    bsr.start()


################################################################
# Pillow images
from PIL import Image, ImageDraw
def draw_text(img, text_to_draw, color=(0, 0, 0), alpha=0, top=0, left=0, width=128, height=128, paster_size=(128, 128)):
    to_paste = img.convert('RGBA')
    to_draw = Image.new('RGBA', paster_size, (255, 255, 255, alpha))
    ImageDraw.Draw(to_draw).text((0, 0), text_to_draw, color)
    to_draw = to_draw.resize((width, height))
    # img_out = Image.alpha_composite(to_paste, to_draw)
    to_paste.paste(to_draw, (left, top), to_draw.convert("RGBA"))
    to_paste = to_paste.convert('RGB')
    return to_paste

base = Image.open("genmap2.jpg")
img2 = draw_text(base, "tester", top=64, left=128)
img2


################################################################
# AsyncIO for parsers
urls = []
for uts in UTS.objects.all().filter(meta__source="investing"):
    urls.append(uts.meta["url_hprice"])
    if len(urls) >= 5:
        break
psrs = [HistoricalPrice().set_all({"url": url}) for url in urls]

import asyncio
loop = asyncio.get_event_loop()
tasks = [loop.run_in_executor(None, psr.run) for psr in psrs]
loop.run_until_complete(asyncio.wait(tasks))

"""
