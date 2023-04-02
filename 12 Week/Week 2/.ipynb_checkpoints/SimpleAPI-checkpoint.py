#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 10:39:14 2023

@author: david
"""
import json
import requests

# Get random activity from Bored API
response = requests.get("http://www.boredapi.com/api/activity/").json()
key = response['key']

# Use the returned key to get the specific activity and print it to the console
specific_response = requests.get(f"http://www.boredapi.com/api/activity?key={key}")
activity = json.loads(specific_response.content)
print(json.dumps(activity, indent = 4))