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
    # ugc_comment = """select distinct comment
    #   from mart_ugc.cut_comment_data
    # """
    ugc_comment = """select distinct
          reviewbody
     from mart_ugc.dpmid_ugcreview_basereview
     where reviewid is not null and
     reviewbody is not null and
     reviewbody not in ('')
     limit 1000000000"""

    read_df = hive_ctx.sql(ugc_comment)


    def _convert_to_unicode(text):
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))

    def get_dictset(filename):
        # 获取需求、供给、需求词表，匹配使用
        words_dict = set()
        words_list = sc.textFile('viewfs:///user/hadoop-aipnlp/baichuanyang/'+filename + '.txt').collect() #collect() rdd -> list
        for word in words_list:
            words_dict.add(_convert_to_unicode(word).strip())
        return words_dict
    commodity_dictset = get_dictset('commodity_category')
    property_dict = get_dictset('release_property_value')
    #----加载词表完成！


    def spark_data_process(rows,commodity_dictset,property_dict):
        def word_split(rows):
            '''
            :param rows: list(str,str...)
            :return:list(str,str...)
            '''
            jieba.load_userdict("userdict.txt")
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

        def match_cut_text(flag, word_list, word_set):
            tag = [['[SUBST]', '[SUBED]'], ['[PROPST]', '[PROPED]']]
            result = word_list[:]
            index = 0
            for word in word_list:
                if word in word_set:
                    result.insert(index, tag[flag][0])
                    index += 1
                    result.insert(index + 1, tag[flag][1])
                    index += 1
                index += 1
            return result

        def find_tagcomment(line):
            '''
            :param line: list(str)
            :return: str
            '''
            flag = False
            if u'[SUBST]' in line and u'[PROPST]' in line:

                # 属性值和商品名一样的标签处理  [SUBST] [PROPST] 牛排 [PROPED] [SUBED]  ——> [SUBST] 牛排 [SUBED]
                index = 0
                while index < len(line) - 1:
                    if (line[index] == u'[SUBST]' and line[index + 1] == u'[PROPST]'):
                        line.pop(index + 1)
                        line.pop(index + 2)  # 注意删除一个元素之后 后面列表相当于整体往前挪一个
                        index += 2
                    index += 1
                property_commodity=[]
                # 删除原子商品（不含属性值）的标签 （是否可以放到下面的遍历？减少速度）  并删除单独的属性值（不和商品连接）标签
                index = 0
                while index < len(line)-1:
                    if index == 0: #开头是sub 说明前面没有属性 去掉
                        if line[0] == u'[SUBST]':
                            line.pop(0)
                            line.pop(1)
                    elif index+1==len(line)-1 and line[index+1] == u'[PROPED]': #结尾是属性 说明后面没有商品 去掉
                        line.pop(index+1)
                        line.pop(index-1)
                        break
                    else:
                        if (line[index] == u'[SUBST]' and line[index - 1] != u'[PROPED]'):
                            line.pop(index)
                            line.pop(index + 1)
                            index-=1
                        if (line[index+1] != u'[SUBST]' and line[index] == u'[PROPED]'):
                            line.pop(index)
                            line.pop(index-2)
                            index -= 2  # 注意一下pop  index前面的数据，index相当于后移
                        if line[index] == u'[PROPED]' and line[index + 1] == u'[SUBST]':  # 属性商品的连接  [PROPST] 蒜香 [PROPED] [SUBST] 虾仁 [SUBED] ——> [SUBST] 蒜香虾仁 [SUBED]
                            flag = True
                            property_commodity.append([line[index - 1],line[index + 2]])

                    index += 1
            if flag==False:#不存在属性和供给连接的情况
                return ''
            return property_commodity

        cut_lines = word_split(rows)
        lines_list = get_lines_list(cut_lines)

        matched_wordlist = []
        for line in lines_list:
            if line == None or line == u' ':
                continue
            line_sub = match_cut_text(0, line, commodity_dictset)
            line_sub_prop = match_cut_text(1, line_sub, property_dict)
            matched_wordlist.append(line_sub_prop)
            #----匹配两实体和属性值 并加标签完成！

        result = []
        for line in matched_wordlist:
            result_lines = find_tagcomment(line)
            if result_lines != ''and result_lines !=[]:
                for result_line in result_lines:
                    result.append([result_line[0],result_line[1],result_line[0]+result_line[1]]) #注意转化成转化成df
        return result



    cut_comment_table_name = 'mart_aikg.property_commodity_table'
    read_df.rdd.repartition(3000).mapPartitions(lambda rows: spark_data_process(rows,commodity_dictset,property_dict)).toDF().toDF("property","commodity","property_commodity").write.format(
        "hive").mode("overwrite").saveAsTable(cut_comment_table_name)

