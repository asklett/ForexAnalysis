# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 20:08:06 2021

@author: askle
"""

import configparser
from datetime import datetime
from twilio.rest import Client
from github import Github


# loading configuration file
config = configparser.ConfigParser()
config.read('config.ini')


def send_mms(file_path, details):
    '''
    Sends a multi-media text message (MMS)

    Parameters
    ----------
    file_path : STRING
        File path of the .png image that is being sent

    details : LIST
        List of items to include in message, including pair name, quantity (units)
        to buy, target pips, and the projected profit

    Returns
    -------
    None.

    '''

    # Read in image file
    with open(file_path, 'rb') as file:
        content = file.read()

    # Upload to github
    git = Github(config['DEFAULT']['GitHubAccessToken'])
    repo = git.get_user().get_repo('ForexCharts')
    git_file = 'Chart.png'
    repo.create_file(git_file, "committing files", content, branch="main")

    # Send MMS message
    account_sid = config['DEFAULT']['Twilio_sid']
    auth_token = config['DEFAULT']['Twilio_token']

    client = Client(account_sid, auth_token)

    msg = ('\n\nPair: {0}' \
        + '\nUnits: {1}' \
        + '\nPips: {2} for ${3:1.2}' \
        + '\nTime: {4}').format(details[0], details[1], details[2], details[3], str(datetime.now())[5:16])

    client.messages \
        .create(
            body = msg,
            from_= config['DEFAULT']['Twilio_number'],
            media_url = 'http://raw.githubusercontent.com/asklett/ForexCharts/main/Chart.png',
            to = config['DEFAULT']['to_field']
            )

    # Delete file from github
    contents = repo.get_contents('Chart.png')
    repo.delete_file(contents.path, "committing files", contents.sha, branch="main")


def send_sms(details):
    '''
    Sends a standard SMS message

    Parameters
    ----------
    details : LIST
        List of items to include in message, including pair name, quantity (units)
        to buy, target pips, and the projected profit

    Returns
    -------
    None.

    '''

    account_sid = config['DEFAULT']['Twilio_sid']
    auth_token = config['DEFAULT']['Twilio_token']

    client = Client(account_sid, auth_token)

    msg = ('\n\nPair: {0}' \
        + '\nUnits: {1}' \
        + '\nPips: {2} for ${3:1.2}' \
        + '\nTime: {4}').format(details[0], details[1], details[2], details[3], str(datetime.now())[5:16])

    client.messages \
        .create(
            body= msg,
            from_= config['DEFAULT']['Twilio_number'],
            to = config['DEFAULT']['to_field']
            )
