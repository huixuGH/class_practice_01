# 导入工具包
import numpy as np
import operator
import pandas as pd

"""
函数说明:kNN算法,分类器

参数:
	testing ： 用于分类的一个数据
	train_data ： 训练集数据
	train_label ： 训练集数据标签
	k ： kNN算法参数,选择距离最小的k个点
返回值:
	sortedClassCount[0][0] ： 返回分类结果

"""

def classify(testing, train_data, train_label, k):
    # numpy函数shape[0]返回train_data的行数
    train_data_size = train_data.shape[0]
    # 在列向量方向上重复testing共1次(横向),行向量方向上重复train_data_size次(纵向)
    diffMat = np.tile(testing, (train_data_size, 1)) - train_data
    # 二维特征相减后平方
    diffMat_sqrt = diffMat ** 2
    # sum(1)行相加
    distance = diffMat_sqrt.sum(axis=1)
    # 返回distances中元素从小到大排序后的索引值
    sorted_distance_index = distance.argsort()
    # 定一个记录类别次数的字典
    classCount = {}
    for i in range(k):
        # 取出前k个元素的类别
        voteIlabel = train_label[sorted_distance_index[i]]
        # dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        # 计算类别次数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # key=operator.itemgetter(1)根据字典的值进行排序
    # reverse降序排序字典
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回次数最多的类别,即所要分类的类别
    return sortedClassCount[0][0]

"""
函数说明:手写数字识别分类

参数:
	train_data ： 训练集数据
	train_label ： 训练集数据标签
	test_data ： 测试集数据
	k ： kNN算法参数,选择距离最小的k个点
返回值:
	无
"""

def handwritingClassTest(train_data,train_label,test_data,k):
    # 错误检测计数
    #error_num = 0.0
    # 测试数据的数量
    test_lence = len(test_label)
    # 从文件中解析出测试集的类别并进行分类测试
    test_result = []
    for i in range(test_lence):
        testing = test_data[i]
        classifierResult = classify(testing, train_data,train_label, k)
        test_result.append([i+1,classifierResult])
        # 把分类所得的结果以特定的格式保存到new_result.csv文件夹中
        result = pd.DataFrame(test_result,columns=['ImageId','Label'])
        result.to_csv('new_result.csv',index=False)
    #     if (classifierResult != test_label[i]):
    #         error_num += 1.0
    # print("总共错了%d个数据\n正确率为%.2f%%" % (error_num, (test_lence-error_num)*100 / test_lence))
    # print("")



"""
函数说明:主函数

参数:
	无
返回值:
	无

"""
if __name__ == '__main__':


    """读取文件信息，存放到矩阵中"""
    #读取训练集CSV格式文件
    train = pd.read_csv('train.csv')
    #转化数据格式为数组格式，以便后续运算
    train_data = train.iloc[:,1:].values
    train_label = train.iloc[:,0].values
    #读取测试集CSV格式文件
    test = pd.read_csv('test.csv')
    test_data = test.iloc[:,:].values

    handwritingClassTest(train_data,train_label,test_data,4)


    """读取文件信息，存放到矩阵中"""
    # train = pd.read_csv("train.csv")
    # train_data = train.iloc[:10000,1:].values
    # train_label = train.iloc[:10000,0].values
    # test_data = train.iloc[10000:12000,1:].values
    # test_label = train.iloc[10000:12000,0].values
    #
    # for i in range(1,10):
    #    print("k=%d时:"% i)
    #    handwritingClassTest(train_data,train_label,test_data,test_label,i)
