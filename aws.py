#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 10:41:21 2023

@author: anna
"""

from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient, AWSIoTMQTTShadowClient
import logging
import json
import argparse 
import os
import time
from datetime import datetime
from io import BytesIO
from PIL import Image
import base64
import numpy as np

def prepareAttachement(image_array):
    
    buff = BytesIO()
    img = Image.fromarray(np.uint8(image_array)).convert(mode='RGB')

    img.save(buff, 'png')
    str_img = base64.b64encode(buff.getvalue())
    
    return (str(str_img.decode('ascii')))

def customShadowCallback_Update(payload, responseStatus, token):

    # Display status and data from update request
    if responseStatus == "timeout":
        print("Update request " + token + " time out!")

    if responseStatus == "accepted":
        payloadDict = json.loads(payload)
        print("Update request with token: " + token + " accepted!")


    if responseStatus == "rejected":
        print("Update request " + token + " rejected!")

# Function called when a shadow is deleted
def customShadowCallback_Delete(payload, responseStatus, token):

     # Display status and data from delete request
    if responseStatus == "timeout":
        print("Delete request " + token + " time out!")

    if responseStatus == "accepted":
        print("~~~~~~~~~~~~~~~~~~~~~~~")
        print("Delete request with token: " + token + " accepted!")
        print("~~~~~~~~~~~~~~~~~~~~~~~\n\n")

    if responseStatus == "rejected":
        print("Delete request " + token + " rejected!")


# Read in command-line parameters
def parseArgs():

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--endpoint", action="store", required=True, dest="host", help="Your device data endpoint")
    parser.add_argument("-r", "--rootCA", action="store", required=True, dest="rootCAPath", help="Root CA file path")
    parser.add_argument("-c", "--cert", action="store", dest="certificatePath", help="Certificate file path")
    parser.add_argument("-k", "--key", action="store", dest="privateKeyPath", help="Private key file path")
    parser.add_argument("-p", "--port", action="store", dest="port", type=int, help="Port number override")
    parser.add_argument("-n", "--thingName", action="store", dest="thingName", default="Bot", help="Targeted thing name")
    parser.add_argument("-id", "--clientId", action="store", dest="clientId", default="basicShadowUpdater", help="Targeted client id")

    args = parser.parse_args()
    return args


# Configure logging
# AWSIoTMQTTShadowClient writes data to the log
def configureLogging():

    logger = logging.getLogger("AWSIoTPythonSDK.core")
    logger.setLevel(logging.DEBUG)
    streamHandler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)

def connectAndSendToAWS(photoId, image_array):
    
    certsPath = os.path.join('aws_certs')
    rootCAPath = os.path.join(certsPath, 'AmazonRootCA1.pem')
    privateKeyPath = os.path.join(certsPath, 'private.pem.key')
    certificatePath = os.path.join(certsPath, 'certificate.pem.crt')
    endpointPath = os.path.join(certsPath, 'ats.iot.eu-central-1.amazonaws.com')
    
    # Init AWSIoTMQTTShadowClient
    myAWSIoTMQTTClient = AWSIoTMQTTShadowClient("JetsonNano")
    myAWSIoTMQTTClient.configureEndpoint0(endpointPath, 8883)
    myAWSIoTMQTTClient.configureCredentials(
            rootCAPath, 
            privateKeyPath, 
            certificatePath)
    
    # AWSIoTMQTTShadowClient connection configuration
    myAWSIoTMQTTClient.configureAutoReconnectBackoffTime(1, 32, 20)
    myAWSIoTMQTTClient.configureConnectDisconnectTimeout(20) # 10 sec
    myAWSIoTMQTTClient.configureMQTTOperationTimeout(15) # 5 sec
        
    connecting_time = time.time() + 20
    
    if time.time() < connecting_time:
    # Connect to AWS IoT
        myAWSIoTMQTTClient.connect()

    # Create a device shadow handler, use this to update and delete shadow document
    deviceShadowHandler = myAWSIoTMQTTClient.createShadowHandlerWithName("JetsonNano", True)
    
    # Delete current shadow JSON doc
    deviceShadowHandler.shadowDelete(customShadowCallback_Delete, 5)
    
    now = datetime.utcnow()
    ctime = now.strftime('%Y-%m-%dT%H:%M:%SZ')
    # Create message payload
    payload = {"state":{"reported": {"varroaDetection": 1, "message": "infeceted bee was detected, id:" + id, "timestamp": str(ctime), 'img': prepareAttachement(image_array) }}}
    
   
    myAWSIoTMQTTClient.publish("iot/topic", json.dumps(payload), 0)
    # Update shadow
    deviceShadowHandler.shadowUpdate(json.dumps(payload), customShadowCallback_Update, 5)
