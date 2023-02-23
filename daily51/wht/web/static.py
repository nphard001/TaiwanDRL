import requests_html

DEFAULT_ENCODING = 'utf-8'
DEFAULT_URL = 'https://example.org/'
MOCK_USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'


def build_html(html_raw, url=DEFAULT_URL, default_encoding=DEFAULT_ENCODING) -> requests_html.HTML:
    """
    Build useful ``requests_html.HTML`` instance via raw ``str`` or ``bytes`` input
    """
    # to prevent partial utf-8 problem:
    #     UnicodeEncodeError: 'utf-8' codec can't encode character
    #     '\ud835' in position 502440: surrogates not allowed
    if isinstance(html_raw, str):
        html_raw = html_raw.encode(default_encoding, 'ignore')
    return requests_html.HTML(html=html_raw, url=url, default_encoding=default_encoding)


def get_static_html(url, mock=True) -> requests_html.HTML:
    """
    Get HTML instance via the standard requests_html process
    NOTES: A mock user is implemented, but it seems okay without that
    """
    session = requests_html.HTMLSession()
    if mock:
        r = session.get(url, headers={'User-Agent': MOCK_USER_AGENT})
    else:
        r = session.get(url)
    return r.html
