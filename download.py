# dl=1 is important
url="https://www.dropbox.com/sh/el7olcqjjnljeb5/AACWu6X24lohLrkjKL5ki8Gea?dl=1"
import urllib.request
u = urllib.request.urlopen(url)
print("Downloading")
data = u.read()
u.close()

filename = 'mir-1k.zip'
 
with open(filename, "wb") as f :
    f.write(data)
