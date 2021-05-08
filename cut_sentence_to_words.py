# -*- coding: utf-8 -*-
import jieba
import io
# 加载自己的自己的金融词库
jieba.load_userdict("financialWords.txt")

def main():
    with io.open('news201708.txt','r',encoding='utf-8') as content:
        for line in content:
            seg_list = jieba.cut(line)
#           print '/'.join(seg_list)
            with io.open('seg201708.txt', 'a', encoding='utf-8') as output:
                output.write(' '.join(seg_list))

if __name__ == '__main__':
    main()
