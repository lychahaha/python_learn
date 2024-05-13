from hashlib import md5
from hashlib import sha512

md5_code = md5(b'123').hexdigest()

sha_code = sha512(b'123').hexdigest()
