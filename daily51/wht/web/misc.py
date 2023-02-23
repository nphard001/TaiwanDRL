def display_html(input):
    from IPython.display import display, HTML
    if hasattr(input, "html"):
        raw_html = input.html
    else:
        raw_html = input
    return display(HTML(raw_html))


def kill_chrome():
    import psutil
    proc_list = []
    for proc in psutil.process_iter():
        cmd = proc.cmdline()
        if len(cmd) > 0:
            a = 'pyppeteer/local-chromium' in cmd[0]
            b = 'chromium-browser' in cmd[0]
            if a or b:
                proc_list.append(proc)
    for proc in proc_list:
        proc.kill()
    return len(proc_list)
