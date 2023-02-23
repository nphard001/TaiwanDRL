import os
import pytest
import daily51.wht.web as qweb


@pytest.mark.skipif(os.environ.get("TEST_LEVEL", "") != "long", reason="too long")
def test_mock_user():
    url = "https://www.taifex.com.tw/cht/9/futuresQADetail"
    html = qweb.get_static_html(url)
    print(type(html))
    print(len(html.text))
    assert "台積電" in html.text
    html = qweb.get_static_html(url, mock=False)
    print(type(html))
    print(len(html.text))
    assert "台積電" in html.text
