#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from getpass import getpass
import json 

#username = input("Please enter a username:")  # Your username
#password = getpass(f'Please enter a password for {username}:') # Your username
username = 'put_your_username_here'  # Your username
password = 'put_your_password_here' # Your username

# In[ ]:


with open("./conf/config.json") as fp:
    conf = json.loads(fp.read())['prod']

host = conf['host']
client_id = conf['_client_id']
audience = conf['_audience']
token_url = conf['_token_url']


# In[ ]:


from oauthlib.oauth2 import LegacyApplicationClient
from requests_oauthlib import OAuth2Session

session = OAuth2Session(client=LegacyApplicationClient(client_id=client_id))
token = session.fetch_token(username=username, password=password,
                                 token_url=token_url, audience=audience,
                                 client_id=client_id, include_client_id=True)
token

