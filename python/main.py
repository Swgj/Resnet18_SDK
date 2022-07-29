#!/usr/bin/env python
# coding=utf-8

"""
Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
Description: main.
Author: MindX SDK
Create: 2021
History: NA
"""

# import StreamManagerApi.py
from StreamManagerApi import StreamManagerApi, MxDataInput, InProtobufVector, MxProtobufIn
import MxpiDataType_pb2 as MxpiDataType
from PIL import Image
import numpy as np

# resize and crop
def resize(img, size, interpolation=Image.BILINEAR):
    return img.resize(size[::-1], interpolation)

def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img

# preprocessor
def preprocess(in_file):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = Image.open(in_file).convert('RGB')
    w = img.size[0]
    h = img.size[1]
    if w > h:
        input_size = np.array([256, 256 * w / h])
    else:
        input_size = np.array([256 * h / w, 256])
    input_size = input_size.astype(int)
    print(input_size)

    img = resize(img, input_size)  # transforms.Resize(256)
    img = np.array(img, dtype=np.float32)
    img = center_crop(img, 224, 224)   # transforms.CenterCrop(224)
    img = img / 255.  # transforms.ToTensor()
    img[..., 0] = (img[..., 0] - mean[0]) / std[0]
    img[..., 1] = (img[..., 1] - mean[1]) / std[1]
    img[..., 2] = (img[..., 2] - mean[2]) / std[2]

    img = img.transpose(2, 0, 1)   # HWC -> CHW
    return img

#generate protobuf
def gen_protobuf(in_file):
    img_np = preprocess(in_file)
    vision_list = MxpiDataType.MxpiVisionList()
    vision_vec = vision_list.visionVec.add()
    vision_vec.visionInfo.format = 0
    vision_vec.visionInfo.width = 224
    vision_vec.visionInfo.height = 224
    vision_vec.visionInfo.widthAligned = 224
    vision_vec.visionInfo.heightAligned = 224

    vision_vec.visionData.memType = 0
    vision_vec.visionData.dataStr = img_np.tobytes()
    vision_vec.visionData.dataSize = len(img_np)

    protobuf = MxProtobufIn()
    protobuf.key = b"appsrc0"
    protobuf.type = b'MxTools.MxpiVisionList'
    protobuf.protobuf = vision_list.SerializeToString()
    protobuf_vec = InProtobufVector()

    protobuf_vec.push_back(protobuf)
    return protobuf_vec

if __name__ == '__main__':
    # init stream manager
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("../pipeline/resnet18.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream
    dataInput = MxDataInput()
    protobuf = gen_protobuf("../data/test.jpg")


    # Inputs data to a specified stream based on streamName.
    streamName = b'resnet18_classification'
    inPluginId = 0
    uniqueId = streamManagerApi.SendProtobuf(streamName, inPluginId, protobuf)
    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()

    # Obtain the inference result by specifying streamName and uniqueId.
    inferResult = streamManagerApi.GetResult(streamName, uniqueId)
    if inferResult.errorCode != 0:
        print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
            inferResult.errorCode, inferResult.data.decode()))
        exit()

    # print the infer result
    print(inferResult.data.decode())

    # destroy streams
    streamManagerApi.DestroyAllStreams()
