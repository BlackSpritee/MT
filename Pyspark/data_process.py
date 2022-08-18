import jieba
import re


def word_split(rows):
    '''
    :param rows: list(str,str...)
    :return:list(str,str...)
    '''
    jieba.load_userdict("userdict.txt")
    res = []
    for row in rows:
        row = row.replace(u'\\\\n', u'\n').replace(u'\\\\t', u'\t').replace(u'\\\\r', u'\r')  # 替换清洗
        sentence_spliter = u"\.\.\.|。|！|!|\?|？|~|\n|\r|\t|…|～"  # ｜   |;|；| |,|，｜更细粒度
        sents = re.split(sentence_spliter, row)
        for sentence in sents:
            l_content = jieba.lcut(sentence.replace('\x00', '').strip())  # 切词
            s_content = u' '.join(l_content)
            if s_content != '' and s_content != None:
                res.append(s_content)
    return res


def cut_save():
    #原始的评论数据
    with open('用户评论表-329047328-1641972012683.txt', 'r', encoding='utf-8') as r:
        rows = r.readlines()
    #切词保存数据
    with open('cut_data_329047328-1641972012683.txt', 'w', encoding='utf-8') as f:
        rows = word_split(rows)
        for row in rows:
            if row != ' ' and row != '\n':
                f.write(row + '\n')


def get_lines_list():
    with open('cut_data_329047328-1641972012683.txt', 'r', encoding='utf-8') as f:
        word_list = []
        rows = f.readlines()
        for row in rows:
            row = row.replace(u'\\\\n', u'\n').replace(u'\\\\t', u'\t').replace(u'\\\\r', u'\r')
            word_list.append(row.split())
    return word_list


def get_dictset(filename):
    words_dict = set()
    with open(filename+'.txt', 'r', encoding='utf-8') as r:
        for word in r.readlines():
            words_dict.add(word[:-1].strip())
    return words_dict

def match_cut_text(flag, word_list, word_set):
    tag = [['[SUBST]', '[SUBED]'], ['[OBJST]', '[OBJED]'], ['[PROPST]', '[PROPED]']]
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

def find_propertypv():
    '''
    release属性表 删除小于pv小于10以下的属性值 (sql 已实现）
    :return:
    '''
    pv_list=[]
    with open('property_pv.txt','r',encoding='utf-8') as r:
        pv=r.readlines()
    for line in pv:
        pv_list.append(line.strip().split('\t')[0])
    with open('property_value.txt','r',encoding='utf-8') as f:
        propty_value=f.readlines()
    # with open('release_property_value.txt','w',encoding='utf-8') as w:
    #     for propty in propty_value:
    #         for pv in pv_list:
    #             if propty==pv[0] and pv[1]>=10.0:
    #                 print(propty+'\n')
    #                 w.write(propty+'\n')
    with open('release_property_value.txt', 'w', encoding='utf-8') as w:
        for propty in propty_value:
            propty=propty.strip()
            if propty in pv_list :
                w.write(propty+'\n')

def find_tagcomment(line):
    '''
    :param line: list(str)
    :return: str
    '''
    flag = False
    if u'[SUBST]' in line and u'[OBJST]' in line and u'[PROPST]' in line:
        # 属性值和需求值一样的标签处理
        index = 0
        while index < len(line) - 1:
            if (line[index] == u'[PROPED]' and line[index + 1] == u'[OBJED]'):
                # 多此一举
                # if index < len(line) - 2 and line[
                #     index + 2] == '[SUBST]':  # [OBJST] [PROPST] 不要 [PROPED] [OBJED] [SUBST] 辣油 [SUBED]  ->[SUBST] 不要辣油 [SUBED]
                #     line[index + 3] = line[index + -1] + line[index + 3]
                #     line = line[:index - 3] + line[index + 2:]
                # else:
                line.pop(index)
                line.pop(index - 2)
            index += 1

        # 属性值和商品名一样的标签处理  [SUBST] [PROPST] 牛排 [PROPED] [SUBED]  ——> [SUBST] 牛排 [SUBED]
        index = 0
        while index < len(line) - 1:
            if (line[index] == u'[SUBST]' and line[index + 1] == u'[PROPST]'):
                line.pop(index + 1)
                line.pop(index + 2)  # 注意删除一个元素之后 后面列表相当于整体往前挪一个
                index += 2
            index += 1

        # 属性值和商品名一样的标签处理  '[SUBST]', '[OBJST]', '麻将', '[OBJED]', '[SUBED]'->'[SUBST]', '麻将',  '[SUBED]'
        index = 0
        while index < len(line) - 1:
            if (line[index] == u'[SUBST]' and line[index + 1] == u'[OBJST]'):
                line.pop(index + 1)
                line.pop(index + 2)  # 注意删除一个元素之后 后面列表相当于整体往前挪一个
                index += 2
            index += 1
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

                # 带有属性的商品，合并标签（下面的属性值在商品后面的，过滤掉，只找属性值在商品前面的标签
                # if (line[index]=='[SUBED]' and line[index+1]=='[PROPST]'):#商品属性的连接 [SUBST] 空调  [SUBED] [PROPST] 温度 [PROPED] ——> [SUBST] 空调温度 [SUBED]
                #     line[index-1]=str(line[index-1])+str(line[index+2])
                #     line=line[:index+1]+(line[index+4:])
                if line[index] == u'[PROPED]' and line[index + 1] == u'[SUBST]':  # 属性商品的连接  [PROPST] 蒜香 [PROPED] [SUBST] 虾仁 [SUBED] ——> [SUBST] 蒜香虾仁 [SUBED]
                    flag = True
                    line[index - 2] = u'[SUBST]'
                    line[index - 1] = line[index - 1] + line[index + 2]
                    line = line[:index] + (line[index + 3:])
            index += 1
    if flag==False:#不存在属性和供给连接的情况
        return ''
    def subobj_combine(line,distance):#筛选出所有的组合 obj和sub距离小于等于distance
        lines=[]
        sub_words=set()
        obj_words=set()
        index=0
        while index<len(line):
            if line[index]==u'[SUBST]':
                sub_words.add(line[index+1])
                line.pop(index)
                line.pop(index+1)
            if line[index]==u'[OBJST]':
                obj_words.add(line[index+1])
                line.pop(index)
                line.pop(index+1)
            index+=1
        for sub_word in sub_words:
            for obj_word in obj_words:
                line_new=line[:]
                sub_index=line.index(sub_word)
                obj_index=line.index(obj_word)
                if abs(sub_index-obj_index)<=distance:
                    line_new.insert(sub_index,u'[SUBST]')
                    line_new.insert(sub_index+2,u'[SUBED]')
                    second_insert_index=line_new.index(obj_word)
                    line_new.insert(second_insert_index,u'[OBJST]')
                    line_new.insert(second_insert_index+2,u'[OBJED]')
                    lines.append(line_new)
        return lines
    return subobj_combine(line,10)




if __name__ == '__main__':
    # cut_save() #切句切词处理
    ## find_propertypv() #已经用sql实现
    lines_list = get_lines_list()
    concept_dictset = get_dictset('release_concept_value')
    commodity_dictset = get_dictset('commodity_category')
    property_dict = get_dictset('release_property_value')
    print("----加载词表完成！")

    result = []
    for line in lines_list:
        if line == None or line == ' ':
            continue
        line_obj = match_cut_text(0, line, commodity_dictset)
        line_obj_sub = match_cut_text(1, line_obj, concept_dictset)
        line_obj_sub_prop = match_cut_text(2, line_obj_sub, property_dict)
        result.append(line_obj_sub_prop)
    print("----匹配两实体和属性值 并加标签完成！")

    with open('tag_data_329047328-1641972012683.txt', 'w', encoding='utf-8') as f:
        for i,line in enumerate(result):
            result_lines = find_tagcomment(line)
            if result_lines != ''and result_lines !=[]:
                for result_line in result_lines:
                    sub = u''
                    obj = u''
                    for j, word in enumerate(result_line):
                        if word == u'[SUBST]':
                            sub = result_line[j + 1]
                        if word == u'[OBJST]':
                            obj = result_line[j + 1]
                f.write(' '.join(result_line) + '\t'+sub+'_'+obj+'\t'+sub+'\t'+obj+'\n')

        print("----write finished!")


