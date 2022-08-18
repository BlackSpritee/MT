#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
import sys
sys.path.append('..')

import load_data
import model_infer

def init(params):
    return MLP_INFERENCE_CONTEXT(params)

class MLP_INFERENCE_CONTEXT:

    def __init__(self ,params):
        self.params = params

        '''
            2、初始化模型
        '''


    '''
    3、自定义inference逻辑
    @:param
     batch_data: 
        type : python list ,contains multiple tfrecord/hive Row data(each Row is a dict)
        length = self.get_batch_size()
            Example:
                batch_data = 
                [
                {'url':url1 ,'image/encoded':xxx},
                {'url':url2 ,'image/encoded':xxx},
                {'url':url3 ,'image/encoded':xxx}
                ]
    @:return
    '''
    def do_inference(self,batch_data = []):

        '''
        # batch size = 1的人脸识别例子
        model = self.model
        # 输出数据

        for i in xrange(len(batch_data)):#在例子中，无论传入的batch_size为多少，只取1
            start_time = time.time()
            data = batch_data[i]
            # 图像字段需要从bytearray转为str
            img = str(data['image/encoded'])
            url = data['url']
            nparr = np.fromstring(img, np.uint8)
            img_nparr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            has_face = model.recog(img_nparr)
            end = time.time()
            # numpy.int转为int
            has_face = int(has_face)

            result.append({'has_face':has_face,'url':url})
            print "%s [INFO] ,url = %s ,inference time cost = %.2f  ,face = %d" %(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                                                                                             url ,time.time() - start_time ,has_face)
        '''

        print(batch_data)
        text,mask=load_data.load_infer(batch_data)
        print(text,mask)
        dataset = torch.utils.data.TensorDataset(text,mask)
        print("dataset 加载成功")
        model=model_infer.BERT_Classifier(9)
        temp=model_infer.infer(model,len(batch_data),dataset)
        result=batch_data
        for index in range(len(temp)):
            for key in temp[index]:
                result[index][key]=temp[index][key]
        print(result)
        return result