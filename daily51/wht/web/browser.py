"""Similar to ``tool51.web``"""
import sys
import time
import asyncio
from typing import Union, List
import psutil
import requests_html
import pyppeteer
from .static import build_html

# 20s session closed solution, source:
# https://github.com/miyakogi/pyppeteer/pull/160#issuecomment-448886155
# pip install websockets==6.0
# pip install websockets==8.1 --force-reinstall
#   8.1 + patch + China = stock at 2min
# pip install pyppeteer==0.0.25 --force-reinstall
#   8.1 + pyppeteer25 + patch + Taiwan = OK

# NOTE: trying pyppeteer 0.2.2


# def patch_pyppeteer():
#     import pyppeteer.connection
#     original_method = pyppeteer.connection.websockets.client.connect

#     def new_method(*args, **kwargs):
#         kwargs['ping_interval'] = None
#         kwargs['ping_timeout'] = None
#         return original_method(*args, **kwargs)

#     pyppeteer.connection.websockets.client.connect = new_method


# patch_pyppeteer()


class BrowserSession(requests_html.HTMLSession):
    """Advanced `requests_html` that always start a new browser to simulate user ops.
    For static uses just go for `StaticSession`.
    """
    browser_process_list = []
    browser_class_default_timeout = 60000  # longer than 30000

    @staticmethod
    def kill_chromium():
        proc_list = []
        for proc in psutil.process_iter():
            try:
                cmd = proc.cmdline()
                if len(cmd) > 0:
                    a = 'pyppeteer' in cmd[0] and 'local-chromium' in cmd[0]
                    b = 'chromium-browser' in cmd[0]
                    if a or b:
                        proc_list.append(proc)
            except (psutil.AccessDenied, OSError):
                pass

        if len(proc_list) == 0:
            print("no chromium found, we are good :)", file=sys.stderr)
            sys.exit(0)

        print("ready to kill:", file=sys.stderr)
        for proc in proc_list:
            print(proc.name(), proc.pid, proc.cmdline(), file=sys.stderr)

        for i in range(5):
            print(f"kill after {5-i} seconds", file=sys.stderr)
            time.sleep(1)

        for proc in proc_list:
            print("kill pid", proc.pid, file=sys.stderr)
            proc.kill()

    @staticmethod
    def clean_process():
        for p in BrowserSession.browser_process_list:
            try:
                p.kill()
            except (psutil.NoSuchProcess, psutil.ProcessLookupError):
                pass  # it is normal
        BrowserSession.browser_process_list = []

    def __init__(self, headless=True, timeout=None, *args, **kwargs):
        self.r = None
        self.headless = headless
        if timeout is None:
            self.browser_default_timeout = self.browser_class_default_timeout
        self.browser_default_timeout = timeout
        super().__init__(*args, **kwargs)

    def start(self, url="http://example.org"):
        """The ``start`` also issues a GET request. Goto some dummy page can save time."""
        self.r = self.get(url)
        self.r.html.render(keep_page=True)

        # adapt win/linux
        if "setDefaultNavigationTimeout" in dir(self.page):
            self.page.setDefaultNavigationTimeout(self.browser_default_timeout)
        if "setDefaultTimeout" in dir(self.page):
            self.page.setDefaultTimeout(self.browser_default_timeout)
        return self

    # APIs for "complete" something
    def page_update(self):
        """See `HTMLSession.render`"""
        page = self.page
        content = self.complete(page.content())
        html = build_html(content)
        self.r.html.__dict__.update(html.__dict__)
        self.r.html.page = page  # rebuild page handler
        return self

    # Browser & async
    @property
    def browser(self):
        """It may trigger async-in-async problem in jupyter.
        Try ``nest_asyncio``
        solution from:
        https://markhneedham.com/blog/2019/05/10/jupyter-runtimeerror-this-event-loop-is-already-running/
        """
        if not hasattr(self, "_browser"):
            self._browser = asyncio.get_event_loop().run_until_complete(
                pyppeteer.launch(headless=self.headless, **self.get_launch_options()))
            self.browser_process_list.append(self._browser._process)
        return self._browser

    @classmethod
    def get_launch_options(cls) -> dict:
        if not hasattr(cls, "_launch_options"):
            cls._launch_options = {
                'args': [
                    '--no-sandbox',
                    '--window-size=1920,1080',
                ],
                # ubuntu 18.04
                'executablePath': "/usr/lib/chromium-browser/chromium-browser",
            }
            # win10 for testing. run ``download_chromium`` yourself
            from pyppeteer.chromium_downloader import current_platform, chromium_executable
            if current_platform() == "win64":
                cls._launch_options["executablePath"] = str(chromium_executable())
        return cls._launch_options

    @classmethod
    def set_launch_options(cls, options):
        """soft update class-wise browser launch options"""
        cls._launch_options = cls.get_launch_options()
        cls._launch_options.update(options)

    @property
    def loop(self):
        if not hasattr(self, "_loop"):
            self._loop = asyncio.get_event_loop()
        return self._loop

    def complete(self, *args, **kwargs):
        return self.loop.run_until_complete(*args, **kwargs)

    # Dynamic objects
    @property
    def html(self) -> requests_html.HTML:
        return self.r.html

    @property
    def page(self) -> pyppeteer.page.Page:
        return self.r.html.page

    def find(self, keyword, first=True, **kwargs) -> Union[requests_html.Element, List]:
        return self.html.find(keyword, first=first, **kwargs)

    def complete_content(self, async_fn) -> bytes:
        self.complete(async_fn)
        return self.complete(self.page.content())

    def get_html(self) -> requests_html.HTML:
        """fetch a page snapshot in HTML format"""
        content = self.complete(self.page.content())
        return build_html(content, url=self.page.url)


class DelaySession(BrowserSession):
    """Implements user ops in a delay manner.
    It sleeps a given time after each action.
    """

    def __init__(self, headless=True, delay=0.1, *args, **kwargs):
        self.delay = delay
        super().__init__(headless, *args, **kwargs)

    async def sleep(self, seconds=None):
        seconds = seconds or self.delay
        return await asyncio.sleep(seconds)

    async def goto(self, url: str, *args, **kwargs):
        """see ``pyppeteer.page.Page.goto``"""
        await self.page.goto(url, waitUntil="load", *args, **kwargs)  # Testing
        await self.sleep()

    async def click(self, selector: str, *args, **kwargs):
        """sleep `delay` seconds after click"""
        await self.page.focus(selector, *args, **kwargs)  # solve "equities" no code picker
        await self.page.click(selector, *args, **kwargs)
        await self.sleep()

    async def select(self, selector: str, text: str, *args, **kwargs):
        await self.page.select(selector, text, *args, **kwargs)
        await self.sleep()

    async def type(self, selector: str, text: str, *args, **kwargs):
        """clean input by `Jeval`, and sleep `delay` seconds after that"""
        await self.page.querySelectorEval(selector, '(obj) => obj.value = ""')
        await self.page.type(selector, text, *args, **kwargs)
        await self.sleep()

    async def ignore(self, selector: str):
        """prevent things like banner blocks UI"""
        await self.page.addStyleTag(content=selector+"{display: none}")
        if await self.page.querySelector(selector):
            await self.page.querySelectorEval(
                selector, '(obj) => obj.setAttribute("style", "display: none")')

    async def force_click(self, selector: str):
        """To click invisible item"""
        if await self.page.querySelector(selector):
            await self.page.querySelectorEval(
                selector, '(obj) => obj.click()')
            await self.sleep()

    async def get_bounding_box(self, selector: str):
        ele = await self.page.querySelector(selector)
        return await ele.boundingBox()
