bigram_count_dict：双字词频      
bigram_word_to_end_dict：key: 双字字尾; value: 包含该字尾的双字词
bigram_word_to_start_dict：key: 双字字头; value: 包含该字头的双字词
pingze_rhythm_dict：key: 平: 1; 仄: 2;  value: rhythm word, psy.txt中每个类别中的代表，例如'平声一东：'中的'东'
pingze_words_dict：key: 平: 1; 仄: 2;  value: 所有读音为平或者仄的字          
psy.txt：平仄字表                   
qsc.txt：全宋词                                     
reverse_pingze_word_dict：key: 字; value: 字对应的平(1),仄(2)  
reverse_rhythm_word_dict：key: 字; value: 字对应的rhythm word，用于韵脚
rhythm_count_dict：key: rhythm word; value: 代表的rhythm出现的次数        
rhythm_word_dict：key: rhythm word; value: 对应这个rhythm的所有字
sentences：按照'，。[ ]'讲全宋词分割成句子
split_sentences：每个句子的分词，使用jieba分词包
title_delimiter_dict：key: 词牌名; value: 对应的标点
title_pingze_dict：key: 词牌名; value: 对应的平(1)仄(2)通用(0)
word_count_dict：韵脚出现的次数。key: 每句最后一个字对应的rhythm word; value: 该韵脚出现的次数
word_model：用split_sentences训练的word2vec word model.

所有汉字保存为unicode
python调用时使用
item.decode('unicode-escape').encode('utf-8')
