"""
Pickle v4 sugar functions

*Example*
print(f"raw_df = p4.loads_b64('{p4.dumps_b64(raw_df)}')")
"""
import pickle
import base64


def dumps(obj):
    return pickle.dumps(obj, protocol=4)


def dump(obj, file):
    if isinstance(file, str):
        with open(file, 'wb') as f:
            return pickle.dump(obj, f, protocol=4)
    return pickle.dump(obj, file, protocol=4)


def loads(bytes_object):
    return pickle.loads(bytes_object)


def load(file):
    if isinstance(file, str):
        with open(file, 'rb') as f:
            return pickle.load(f)
    return pickle.load(file)


def dumps_b64(obj):
    """p4.dumps(obj) and convert it into b64 string (utf-8)"""
    b64_string = base64.b64encode(dumps(obj)).decode("utf-8", "strict")
    return b64_string


def loads_b64(b64_string):
    """p4.loads(bytes_object) before b64 string decoding"""
    bytes_object = base64.b64decode(b64_string)
    return loads(bytes_object)


def dump_b64(obj, file):
    """p4.dumps_b64(obj) in text file"""
    if isinstance(file, str):
        with open(file, "w") as f:
            return f.write(dumps_b64(obj))
    return file.write(dumps_b64(obj))


def load_b64(file):
    """p4.loads_b64(obj) in text file"""
    if isinstance(file, str):
        with open(file, "r") as f:
            b64 = f.read()
    else:
        b64 = file.read()
    return loads_b64(b64.strip())  # prevent special chars
