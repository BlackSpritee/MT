# -*- coding: utf-8 -*-
import re
import jieba
from pyspark import HiveContext, SparkContext
from pyspark.sql.functions import col
from pyspark.sql import SparkSession
import re
import pandas as pd
sc = SparkContext()
hive_ctx = HiveContext(sc)

log4jLogger = sc._jvm.org.apache.log4j
LOGGER = log4jLogger.LogManager.getLogger("org.apache.spark")
LOGGER.setLevel(log4jLogger.Level.ERROR)
LOGGER.info("=======>pyspark script logger initialized")

if __name__ == '__main__':

    ugc_comment = """select distinct
          reviewbody
     from mart_ugc.dpmid_ugcreview_basereview
     where reviewid is not null and
     reviewbody is not null and
     reviewbody not in ('')"""
    # ugc_comment = """select
    #       reviewbody
    #  from upload_table.tempugc_table"""

    read_df = hive_ctx.sql(ugc_comment)


    def _convert_to_unicode(text):
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))

    def get_turple(filename):
        # 获取人物、事件词表，匹配使用
        words_dict = []
        words_list = sc.textFile('viewfs:///user/hadoop-aipnlp/baichuanyang/'+filename + '.txt').collect() #collect() rdd -> list
        for word in words_list:
            sub,obj,rel=word.split('\t')
            words_dict.append([_convert_to_unicode(sub).strip(),_convert_to_unicode(obj).strip(),_convert_to_unicode(rel).strip()])
        return words_dict

    turple=get_turple('cross_16w')
    #----加载词表完成！


    def spark_data_process(rows):
        def word_split(rows):
            '''
            :param rows: list(str,str...)
            :return:list(str,str...)
            '''
            jieba.load_userdict("entity_dict.txt")
            res = []
            for row in rows:
                row =_convert_to_unicode(row.reviewbody).replace(u'\\\\n', u'\n').replace(u'\\\\t', u'\t').replace(u'\\\\r', u'\r')  # 替换清洗
                # row = row.replace(u'\\\\n', u'\n').replace(u'\\\\t', u'\t').replace(u'\\\\r', u'\r')  # 替换清洗
                sentence_spliter = u"\.\.\.|。|！|!|\?|？|~|\n|\r|\t|…|～"  # ｜   |;|；| |,|，｜更细粒度
                sents = re.split(sentence_spliter, row)
                for sentence in sents:
                    l_content = jieba.lcut(sentence.replace('\x00', '').strip())  # 切词
                    s_content = u' '.join(l_content)
                    if s_content != u'' and s_content != None:
                        res.append(s_content)
            return res

        def get_lines_list(rows):
            # 读切好词的字符串，转换成词的列表
            word_list = []
            for row in rows:
                row = row.strip().replace(u'\\\\n', u'\n').replace(u'\\\\t', u'\t').replace(u'\\\\r', u'\r')
                word_list.append(row.split())
            return word_list

        cut_lines = word_split(rows)
        lines_list = get_lines_list(cut_lines)

        result = []
        for line in lines_list:
            if line == None or line == u' ':
                continue
            for index in range(len(line)-2):
                if line[index] in concept_dictset and line[index+1] in property_dict:
                    if line[index] == '' or line[index]=='':
                        continue
                    result.append([u''.join(line) , line[index]+ u'_' + line[index+1], line[index], line[index+1]])
        return result

    cut_comment_table_name = 'mart_aikg.people_event_comment_table'
    read_df.rdd.repartition(3000).mapPartitions(spark_data_process).toDF().toDF("line", "key", "sub","obj").write.format(
             "hive").mode("overwrite").saveAsTable(cut_comment_table_name)
