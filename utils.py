#coding=utf-8

'''
Some simple utility functions used for Chinese word segmentation
for python 3.x
'''

def is_Chinese_char(uchar) :
    '''
    判断一个Unicode字符是否为中文字符（可能不完全）
    '''
    if len(uchar) != 1 :
        raise TypeError('expected a character, but a string found!')
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5' or uchar in '，。、；‘’“”：？！【】《》（）＜＞￥' :
        return True  
    else :  
        return False 

def is_digit(uchar) :
    '''
    判断一个Unicode字符是否为数字
    '''
    if len(uchar) != 1 :
        raise TypeError('expected a character, but a string found!')
    if uchar >= u'\u0030' and uchar <= u'\u0039' :
        return True
    else :
        return False

def is_alpha(uchar) :
    '''
    判断一个Unicode字符是否为字母
    '''
    if len(uchar) != 1 :
        raise TypeError('expected a character, but a string found!')
    if (uchar >= u'\u0041' and uchar <= u'\u005a' 
    or uchar >= u'\u0061' and uchar<=u'\u007a') :
        return True
    else :
        return False  

if __name__ == "__main__" :
    teststr = "这是1个中文字符串的测试，看一看Python3中对于Unicode字符串的处理"
    for uchar in teststr :
        print(uchar, end=" ")
    print()
    for uchar in teststr :
        if is_Chinese_char(uchar) :
            print("C ", end=" ")
        elif is_digit(uchar) :
            print("D", end=" ")
        elif is_alpha(uchar) :
            print("A", end=" ")
    
