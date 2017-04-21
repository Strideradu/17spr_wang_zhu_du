#-*- coding: utf-8 -*-
# vim: set bg=dark noet ts=4 sw=4 fdm=indent :


import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')
try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser
import logging
import re
import simplejson as json
import jieba
import jieba.posseg as pseg
from gensim import models
import random
import operator

from title_rhythm import TitleRhythmDict

basepath = os.path.abspath(os.path.dirname(__file__))


def my_unicode(lst):
    return repr(lst).decode('unicode-escape')


def my_unicode_sd(d):
    lst = [word for (word, count) in d]
    return my_unicode(lst)


def my_unicode_d(d):
    lst = [word for word, count in d.iteritems()]
    return my_unicode(lst)


class Generator(object):
    """ Generator of Chinese Poem
    """

    def __init__(self, basepath, conf):
        self.basepath = basepath
        self._ci_words_file = os.path.join(
            self.basepath, conf.get('ci', 'ci_words_file'))
        self._ci_rhythm_file = os.path.join(
            self.basepath, conf.get('ci', 'ci_rhythm_file'))
        self._ci_result_file = os.path.join(
            self.basepath, conf.get('ci', 'ci_result_file'))
        self._support_titles = conf.get('ci', 'support_titles')

        # user input
        self._important_words = []
        self._title = ""
        self._force_data_build = False

        # load from data file
        self._title_pingze_dict = {}
        self._title_delimiter_dict = {}
        self._pingze_words_dict = {}
        self._pingze_rhythm_dict = {}
        self._rhythm_word_dict = {}
        self._reverse_rhythm_word_dict = {}
        self._reverse_pingze_word_dict = {}

        self._sentences = []

        # split related data
        self._split_sentences = []
        self._word_model = None

        # word count related
        self._word_count_dict = {}
        self._rhythm_count_dict = {}

        self._bigram_word_to_start_dict = {}
        self._bigram_word_to_end_dict = {}
        self._bigram_count_dict = {}

        # storage of related precalculated data
        self._data_files = [
            "title_pingze_dict", "title_delimiter_dict", "pingze_words_dict", "pingze_rhythm_dict", "rhythm_word_dict", "reverse_rhythm_word_dict", "reverse_pingze_word_dict", "word_count_dict", "rhythm_count_dict", "split_sentences", "bigram_word_to_start_dict", "bigram_word_to_end_dict", "bigram_count_dict", "sentences"
        ]

        # store generated poem
        self._result = ""
        # store error reason if no poem generated
        self._error_info = ""

        self._search_ratio = 0

    @property
    def search_ratio(self):
        return self._search_ratio

    @property
    def important_words(self):
        return self._important_words

    @property
    def title(self):
        return self._title

    @property
    def force_data_build(self):
        return self._force_data_build

    @search_ratio.setter
    def search_ratio(self, value):
        self._search_ratio = value

    @important_words.setter
    def important_words(self, value):
        self._important_words = value

    @title.setter
    def title(self, value):
        self._title = value

    @force_data_build.setter
    def force_data_build(self, value):
        self._force_data_build = value

    def _get_top_words_with_count(self, word_count_dict, topN=1):
        words = []
        if not word_count_dict:
            return u""
        word_count_dict = sorted(word_count_dict.items(
        ), key=operator.itemgetter(1), reverse=True)
        for i, (word, count) in enumerate(word_count_dict):
            if i >= topN:
                break
            words.append((word, count))
        return words

    def _get_top_word_uniform_random(self, word_count_dict, topN=1):
        words_with_count = self._get_top_words_with_count(
            word_count_dict, topN)
        words = []
        [words.append(word) for (word, count) in words_with_count]
        idx = random.randint(0, len(words) - 1)
        return words[idx]

    def _get_top_word_weight_random(self, word_count_dict, topN=1):
        words_with_count = self._get_top_words_with_count(
            word_count_dict, topN)
        return self._weighted_choice(words_with_count)

    def _show_word_sentence(self, format_sentence, word_sentence, logger, comment="omg"):
        logger.info("%s: format_sentence %s" %
                    (comment, my_unicode(format_sentence)))
        tmp_sentence = []
        for i in range(len(format_sentence)):
            if i in word_sentence:
                tmp_sentence.append(word_sentence[i])
            else:
                tmp_sentence.append("X")
        logger.info("%s: word_sentence %s" %
                    (comment, my_unicode(tmp_sentence)))

    def _show_word_sentences(self, format_sentences, word_sentences, logger, comment="omg"):
        [self._show_word_sentence(format_sentence, word_sentence, logger, comment) for (
            format_sentence, word_sentence) in zip(format_sentences, word_sentences)]

    def _build_title_pingze_dict(self, logger):
        for title, content_rhythm in TitleRhythmDict.iteritems():
            # print title
            # print content_rhythm
            sentences = re.findall(r"[0-9]+", content_rhythm)
            new_sentences = []
            for sentence in sentences:
                new_sentence = ""
                for word in sentence:
                    if not int(word):
                        new_sentence += "0"
                    elif not (int(word) % 2):
                        new_sentence += "2"
                    else:
                        new_sentence += "1"
                new_sentences.append(new_sentence)
            self._title_pingze_dict[title.decode()] = new_sentences
            delimiters = []
            for word in content_rhythm:
                if word in [",", ".", "`", "|"]:
                    delimiters.append(word)
            self._title_delimiter_dict[title.decode()] = delimiters

    def _build_pingze_rhythm_words_dict(self, logger):
        with open(self._ci_rhythm_file, 'r') as fp_r:
            count = 1
            while 1:
                line = fp_r.readline()
                line = line.strip().decode("utf-8")
                if not line:
                    continue
                if line == "END":
                    break
                if u"：" in line:  # Chinese title part
                    # print line
                    # print len(line)
                    next_line = fp_r.readline().strip().decode("utf-8")

                    rhythm_word = line[-2]

                    is_ping = True
                    if u"平" in line:  # ping related
                        self._pingze_rhythm_dict.setdefault(
                            '1', []).append(rhythm_word)
                        is_ping = True
                    else:  # ze related
                        self._pingze_rhythm_dict.setdefault(
                            '2', []).append(rhythm_word)
                        is_ping = False

                    # build reverse dict for count later
                    invalid_flag = False
                    invalid_value = []
                    words = []
                    for word in next_line:
                        if word == u"[":
                            invalid_flag = True
                        if invalid_flag:
                            invalid_value.append(word)
                            if word == u"]":
                                invalid_flag = False
                            continue
                        self._reverse_rhythm_word_dict[word] = rhythm_word
                        if is_ping:  # ping related
                            self._reverse_pingze_word_dict[word] = '1'
                        else:  # ze related
                            self._reverse_pingze_word_dict[word] = '2'
                        words.append(word)

                    self._rhythm_word_dict[rhythm_word] = words

                    if u"平" in line:  # ping related
                        self._pingze_words_dict.setdefault(
                            '1', []).extend(words)
                    else:  # ze related
                        self._pingze_words_dict.setdefault(
                            '2', []).extend(words)

                #count += 1
                # if count > 2:
                #	break

    def _count_general_rhythm_words(self, logger):
        with open(self._ci_words_file, 'r') as fp_r:
            count = 1
            while 1:
                line = fp_r.readline()
                line = line.strip().decode("utf-8")
                if not line:
                    continue
                if line == "END":
                    break
                if (u"，" not in line) and (u"。" not in line):  # only use content part for stats
                    continue

                sentences = re.split(u"[，。]", line)
                for sentence in sentences:
                    if sentence:
                        self._sentences.append(sentence)

                        final_word = sentence[-1]
                        # print 'final', final_word
                        if final_word not in self._reverse_rhythm_word_dict:
                            # print 'not exist', final_word
                            continue
                        rhythm_word = self._reverse_rhythm_word_dict[
                            final_word]
                        # print 'rhythm', rhythm_word
                        if final_word not in self._word_count_dict:
                            self._word_count_dict[final_word] = 1
                        else:
                            self._word_count_dict[final_word] += 1
                        if rhythm_word not in self._rhythm_count_dict:
                            self._rhythm_count_dict[rhythm_word] = 1
                        else:
                            self._rhythm_count_dict[rhythm_word] += 1

                        # build 2-gram
                        for idx, word in enumerate(sentence):
                            if idx >= len(sentence) - 1:
                                break
                            first_word = word
                            second_word = sentence[idx + 1]
                            if (first_word == u'、') or (second_word == u'、'):
                                continue
                            bigram_key = '__'.join([first_word, second_word])
                            if bigram_key not in self._bigram_count_dict:
                                self._bigram_count_dict[bigram_key] = 1
                            else:
                                self._bigram_count_dict[bigram_key] += 1
                            self._bigram_word_to_start_dict.setdefault(
                                first_word, []).append(bigram_key)
                            self._bigram_word_to_end_dict.setdefault(
                                second_word, []).append(bigram_key)

                # print line
                # print 'bigram'
                # print self._bigram_count_dict
                # print self._bigram_word_to_start_dict
                # print self._bigram_word_to_end_dict

                #count += 1
                # if count > 10:
                #	break

    def _split_words(self, logger):
        """ split words with jieba"""
        with open(self._ci_words_file, 'r') as fp_r:
            count = 1
            while 1:
                line = fp_r.readline()
                line = line.strip().decode("utf-8")
                if not line:
                    continue
                if line == "END":
                    break
                if (u"，" not in line) and (u"。" not in line):  # only use content part for stats
                    continue

                # print line
                words = jieba.cut(line)
                words = list(words)
                # print '/ '.join(words)
                self._split_sentences.append(words)
                count += 1
                # if count > 10:
                #	break

    def _build_word2vec(self, logger):
        """ build word2vec for words"""
        if not self._split_words:
            logger.error("no split words, skip")
        else:
            self._word_model = models.Word2Vec(
                self._split_sentences, min_count=5)
            self._word_model.save(os.path.join(
                self.basepath, "data", "word_model"))

    def _init_data_build(self, logger):
        """ generate title, pingze, rhythm, word relationship"""
        # mapping title to ping&ze
        self._build_title_pingze_dict(logger)

        # mapping pingze, rhythm to words
        self._build_pingze_rhythm_words_dict(logger)

        # mapping rhythm_end to words,
        self._count_general_rhythm_words(logger)

        # split words
        self._split_words(logger)

        # build word2vec
        self._build_word2vec(logger)

        # save related data
        for data_file in self._data_files:
            value = getattr(self, "_" + data_file)
            with open(os.path.join(self.basepath, "data", data_file), "w") as fp_w:
                json.dump(value, fp_w)

    def _load_data_build(self, logger):
        for data_file in self._data_files:
            with open(os.path.join(self.basepath, "data", data_file), "r") as fp_r:
                value = json.load(fp_r)
                setattr(self, "_" + data_file, value)
        self._word_model = models.Word2Vec.load(
            os.path.join(self.basepath, "data", "word_model"))

    def _get_format_with_title(self, title, logger):
        if title not in self._title_pingze_dict:
            return -1
        return self._title_pingze_dict[title]

    def _check_position_by_sentence_length(self, sentence_length, logger):
        if sentence_length == 7:
            return [0, 2, 4, 5]
        elif sentence_length == 6:
            return [0, 2, 4]
        elif sentence_length == 5:
            return [0, 2, 4]
        elif sentence_length == 4:
            return [0, 2]
        elif sentence_length == 3:
            return [0]
        else:
            return []

    def _weighted_choice(self, choices, already_check_choices=[]):
        sub_choices = []
        for (c, w) in choices:
            if c not in already_check_choices:
                sub_choices.append((c, w))
        total = sum(w for (c, w) in sub_choices)
        r = random.uniform(0, total)
        upto = 0
        for c, w in sub_choices:
            if upto + w >= r:
                return c
            upto += w

    def _compare_words(self, format_words, input_words):
        for (format_word, input_word) in zip(format_words, input_words):
            if format_word == '0':  # no check needed
                continue
            if format_word != input_word:
                return False
        return True

    def _combine_candidate_word_with_single_sentence(self, format_sentence, candidate_words, already_used_words, logger):
        """
        In each sentence, put one candidate word in it
        with consideration of pingze as well as postion and already used condition
        """
        position_word_dict = {}

        logger.info('single sentence: format_sentence %s' %
                    my_unicode(format_sentence))
        logger.debug('single sentence: already_used_words %s' %
                     my_unicode(already_used_words))

        # remove already used words
        logger.debug('single sentence: origin_candidate_words %s' %
                     my_unicode(candidate_words))
        new_candidate_words = [
            word for word in candidate_words if word[0] not in already_used_words]
        logger.debug('single sentence: new_candidate_words %s' %
                     my_unicode(new_candidate_words))
        if not new_candidate_words:
            logger.warning("use all words, that should not happen")
            new_candidate_words = candidate_words

        sentence_length = len(format_sentence)

        # chekc delimiter for sentence
        positions = self._check_position_by_sentence_length(
            sentence_length, logger)
        if not positions:  # don't consider position, only consider pingze
            logger.info(
                "sentence_length[%d] dont check position, as not defined" % sentence_length)

        logger.debug("single sentence: positions %s" % str(positions))

        # random fill first
        random_already_check_words = []
        is_word_found = False
        for i in range(10):

            # randomly select one candidate word
            candidate_word = self._weighted_choice(
                new_candidate_words, random_already_check_words)
            if not candidate_word:
                raise ValueError("candidate_word %s not exist in %s" % (
                    candidate_word, my_unicode(new_candidate_words)))
            random_already_check_words.append(candidate_word)
            logger.debug(
                "single sentence: iter[%d] candidate_word %s" % (i, candidate_word))

            # get word pingze
            word_pingze = []
            word_pingze_flag = True
            for candidate_word_elem in candidate_word:
                if candidate_word_elem not in self._reverse_pingze_word_dict:
                    word_pingze_flag = False
                    break
                word_pingze.append(self._reverse_pingze_word_dict[
                                   candidate_word_elem])
            logger.debug("single sentence: iter[%d] candidate_word %s, word_pingze %s" % (
                i, candidate_word, my_unicode(word_pingze)))

            if (not word_pingze_flag) or (len(word_pingze) != len(candidate_word)):
                logger.warning("word_pingze len[%d] not equal to word len[%d]" % (
                    len(word_pingze), len(candidate_word)))
                continue

            for j in range(len(positions) - 1):  # dont check rhythm part
                pos_start = positions[j]
                pos_end = positions[j + 1]
                tmp_word = format_sentence[pos_start:pos_end]
                logger.debug('iter[%d] pos_iter[%d] word_pingze %s, tmp_word %s' % (
                    i, j, word_pingze, tmp_word))
                if (len(tmp_word) == len(word_pingze)) and (self._compare_words(tmp_word, word_pingze)):
                    # write word with position
                    for p, m in enumerate(range(pos_start, pos_end)):
                        position_word_dict[m] = candidate_word[p]
                    is_word_found = True
                    break

            if is_word_found:
                already_used_words.append(candidate_word)
                logger.info(
                    "single sentence: add candidate_word %s to word_sentence" % candidate_word)
                break

        return position_word_dict

    def _filter_simliar_words(self, whole_similar_words, logger):
        filtered_similar_words = []
        for (word, similarity) in whole_similar_words:
            logger.debug("word[%s] len[%d]" % (word, len(word)))

            word_elems = pseg.cut(word)
            word_flag_valid = False
            for word_elem, flag in word_elems:
                logger.debug("word[%s] word_elem[%s] flag[%s]" %
                             (word, word_elem, flag))
                if flag in ['n', 'ns', 'nr', 't']:
                    word_flag_valid = True
                    break

            if len(word) < 2 and (not word_flag_valid):
                continue

            filtered_similar_words.append((word, similarity))
        return filtered_similar_words

    def _combine_important_word_with_sentence(self, important_words, format_sentences, logger):
        """
        make every sentence has one related importanct word
        and promise pingze order as well as position order

        we try to use whole word to find similar words first,
        if not, then use each word to find
        """
        word_sentences = []

        sentence_length = len(format_sentences)
        candidate_length = 5 * sentence_length

        # if put all words in word2vec.most_similar function, and any one of words not exist will lead to call fail
        # so try to check all words and get most common valid words, ugly but
        # seems no official func given
        useful_important_words = []
        for important_word in important_words:
            try:
                similar_words = self._word_model.most_similar(
                    positive=[important_word], topn=candidate_length)
            except KeyError as e1:
                pass
            else:
                useful_important_words.append(important_word)

        # trick here if no useful word given
        if not useful_important_words:
            logger.warning("no valid tags %s in user input, trick" %
                           my_unicode(useful_important_words))
            useful_important_words = [u"菊花"]

        # cut useful words, it seems too many useful words not ok than simple
        # one
        max_useful_words_len = 3
        if len(useful_important_words) > max_useful_words_len:
            useful_important_words = useful_important_words[
                :max_useful_words_len]

        whole_similar_words = []
        try:
            whole_similar_words = self._word_model.most_similar(
                positive=useful_important_words, topn=candidate_length)
            logger.info("get whole_similar_words %s based on useful_important_words %s as whole" % (
                my_unicode(whole_similar_words), my_unicode(useful_important_words)))
        except KeyError as e:
            logger.exception(e)

        # Oops, we don't know what user want, create one randomly
        if not whole_similar_words:
            logger.warning(
                "Oops, no similar word generated based on important_word[%s] seperately" % str(important_word))
        else:
            # filter word type and word length
            whole_similar_words = self._filter_simliar_words(
                whole_similar_words, logger)
            logger.info("filtered whole_similar_words %s based on important_words %s as whole" % (
                my_unicode(whole_similar_words), my_unicode(important_words)))

            # order list of tuple, and fetch the first candidate_length of
            # candidates
            whole_similar_words = sorted(
                whole_similar_words, key=operator.itemgetter(1), reverse=True)
            candidate_words = whole_similar_words[:candidate_length]
            logger.info("get candidate_words %s based on important_words %s" % (
                my_unicode(candidate_words), my_unicode(important_words)))

        # at now, we promise whole_similar_words have enough data
        # now, combine them with sentences
        already_used_words = []
        for format_sentence in format_sentences:
            word_sentence = self._combine_candidate_word_with_single_sentence(
                format_sentence, candidate_words, already_used_words, logger)
            word_sentences.append(word_sentence)

        return word_sentences

    def _generate_common_rhythm(self, is_ping=True):
        """ generate common rhythm"""

        candidate_rhythms = self._pingze_rhythm_dict[
            "1"] if is_ping else self._pingze_rhythm_dict["2"]
        # print 'rhythm_count', self._rhythm_count_dict

        candidate_rhythm_count_dict = {}
        for candidate_rhythm in candidate_rhythms:
            if candidate_rhythm in self._rhythm_count_dict:
                candidate_rhythm_count_dict[
                    candidate_rhythm] = self._rhythm_count_dict[candidate_rhythm]

        candidate_rhythm_count_dict = sorted(
            candidate_rhythm_count_dict.items(), key=operator.itemgetter(1), reverse=True)

        count = 0
        narrow_candidate_rhythms = []
        for (rhythm, rhythm_count) in candidate_rhythm_count_dict:
            narrow_candidate_rhythms.append((rhythm, rhythm_count))
            count = count + 1
            if count > 5:
                break

        selected_rhythm = self._weighted_choice(narrow_candidate_rhythms)
        return selected_rhythm

    def _generate_common_words(self, rhythm, is_ping=True):
        """ generate common words"""

        candidate_words = self._rhythm_word_dict[rhythm]

        candidate_word_count_dict = {}
        for candidate_word in candidate_words:
            if candidate_word in self._word_count_dict:
                candidate_word_count_dict[
                    candidate_word] = self._word_count_dict[candidate_word]

        candidate_word_count_dict = sorted(
            candidate_word_count_dict.items(), key=operator.itemgetter(1), reverse=True)
        return candidate_word_count_dict

    def _generate_common_rhythm_words(self, is_ping, logger):
        """ generate rhythm words
        first, generate common rhythm
        second, generate words based on rhythm
        """

        logger.info(
            "generate_rhythm: generate common rhythm for isping[%d]" % int(is_ping))
        rhythm = self._generate_common_rhythm(is_ping)
        logger.info("generate_rhythm: use rhythm[%s] for is_ping[%d] generatoin" % (
            rhythm, int(is_ping)))
        logger.info(
            "generate_rhythm: generate common words for isping[%d]" % int(is_ping))
        word_count_dict = self._generate_common_words(rhythm, is_ping)
        logger.info("generate_rhythm: word_count_dict %s for isping[%d]" % (
            my_unicode_sd(word_count_dict), int(is_ping)))
        return word_count_dict

    def _generate_rhythm(self, format_sentences, word_sentences, logger):
        """ generate rhythm"""

        logger.info("generate_rhythm: format_sentences")

        # generate ping word with count
        ping_word_count_dict = self._generate_common_rhythm_words(True, logger)

        # genrate ze word with count
        ze_word_count_dict = self._generate_common_rhythm_words(False, logger)

        already_used_rhythm_words = []
        for format_sentence, word_sentence in zip(format_sentences, word_sentences):
            logger.info("generate_rhythm: format_sentence %s, word_sentence %s" % (
                my_unicode(format_sentence), my_unicode(word_sentence)))
            rhythm_word = ""
            if format_sentence[-1] == '1':
                rhythm_word = self._weighted_choice(
                    ping_word_count_dict, already_used_rhythm_words)
            elif format_sentence[-1] == '2':
                rhythm_word = self._weighted_choice(
                    ze_word_count_dict, already_used_rhythm_words)
            elif format_sentence[-1] == '0':
                rhythm_word = self._weighted_choice(
                    ping_word_count_dict + ze_word_count_dict, already_used_rhythm_words)
            else:
                logger.error("rhythm_type[%s] illegal" % format_sentence[-1])
            already_used_rhythm_words.append(rhythm_word)
            logger.debug("generate_rhythm: use rhythm_word %s" % rhythm_word)

            word_sentence[len(format_sentence) - 1] = rhythm_word

    def _fill_word(self, direction, tofill_position, format_sentence, word_sentence, global_repeat_words, current_repeat_dict, level, logger):
        """ fill word by related word, and position"""

        logger.debug("fill_word: level[%d] fill word" % level)

        seed_word = word_sentence[tofill_position - direction]
        logger.debug("fill_word: level[%d] tofill_position[%d] seed_word %s" % (
            level, tofill_position, seed_word))

        # check 2-gram dict and pingze order
        if direction > 0:
            bigram_word_dict = self._bigram_word_to_start_dict
            verb_position = -1
        else:
            bigram_word_dict = self._bigram_word_to_end_dict
            verb_position = 0

        logger.debug("fill_word: level[%d] verb_position[%d]" % (
            level, verb_position))

        if seed_word in bigram_word_dict:
            candidate_words = bigram_word_dict[seed_word]
            candidate_verb_count_dict = {}
            for candidate_word in candidate_words:

                candidate_verb = candidate_word[verb_position]
                #logger.debug("fill_word: level[%d] candidate_verb %s, candidate_word %s with seed_word %s" % (level, candidate_verb, candidate_word, seed_word))
                if candidate_verb not in self._reverse_pingze_word_dict:
                    #logger.debug("fill_word: level[%d] candidate_verb %s no pingze info, skip" % (level, candidate_verb))
                    continue

                # not use repeated word
                if candidate_verb in global_repeat_words:
                    #logger.debug("fill_word: level[%d] candidate_verb %s in global repeat words %s, skip" % (level, candidate_verb, my_unicode(global_repeat_words)))
                    continue

                # not use too many repeated word in one sentence
                if candidate_verb in current_repeat_dict:
                    if current_repeat_dict[candidate_verb] > 2:
                        logger.debug("fill_word: level[%d] candidate_verb %s in current repeat words, skip" % (
                            level, candidate_verb))
                        continue

                # check pingze order first
                format_tofill_position = format_sentence[tofill_position]
                candidate_verb_position = self._reverse_pingze_word_dict[
                    candidate_verb]
                #logger.debug("fill_word: level[%d] candidate_verb %s format_pingze_position %s, verb_position %s" % (level, candidate_verb, format_tofill_position, candidate_verb_position))
                if (format_tofill_position != '0') and (candidate_verb_position != format_tofill_position):
                    #logger.debug("fill_word: level[%d] candidate_verb %s pingze not match, skip" % (level, candidate_verb))
                    continue

                # set initial, protect not exists
                candidate_verb_count_dict[candidate_verb] = 1
                if candidate_word in self._bigram_count_dict:
                    candidate_verb_count_dict[
                        candidate_verb] = self._bigram_count_dict[candidate_word]

            if candidate_verb_count_dict:  # there exists some valid verbs
                #selected_word = ""
                # definitive select max one
                #max_count = -1
                # for candidate_verb, count in candidate_verb_count_dict.iteritems():
                #	if count > max_count:
                #		max_count = count
                #		selected_word = candidate_verb
                #logger.debug("fill_word: level[%d] select_word %s with count %d" % (level, selected_word, max_count))

                # random select word
                topN = 5
                selected_word = self._get_top_word_weight_random(
                    candidate_verb_count_dict, topN)
                logger.debug("fill_word: level[%d] select_word %s with random topN %d" % (
                    level, selected_word, topN))
            else:
                logger.error(
                    "fill_word: level[%d] no candidate word" % (level))
                if candidate_words:  # no pingze satisfy, random select one
                    idx = random.randint(0, len(candidate_words) - 1)
                    selected_word = candidate_words[idx][verb_position]
                    logger.debug("fill_word: level[%d] select_word %s with idx %d" % (
                        level, selected_word, idx))
                else:
                    raise ValueError(
                        "word exist in bigram_word_dict, but it's empty")
        else:  # word not exists in 2-gram
            logger.error(
                "fill_word: level[%d] seed_word %s not exist in 2-gram" % (level, seed_word))

        # select and fill
        word_sentence[tofill_position] = selected_word
        if selected_word not in current_repeat_dict:
            current_repeat_dict[selected_word] = 1
        else:
            current_repeat_dict[selected_word] += 1

        logger.info("fill_word: level[%d] tofill_position[%d] seed_word %s, fill_word %s" % (
            level, tofill_position, seed_word, selected_word))

    def _up_fill_direction(self, tofill_position, sentence_length, logger):
        """ some words are connected tight than other"""
        format_positions = self._check_position_by_sentence_length(
            sentence_length, logger)

        # we dont know, use down-fill
        if not format_positions:
            return False
        if tofill_position in format_positions:
            return False
        else:
            return True

    def _search_generate(self, format_sentence, word_sentence, global_repeat_words, current_repeat_dict, already_used_sentences, already_used_rhythm_words, logger):
        """ try to search already exist word"""

        # no search for only rhythm sentence
        if len(word_sentence) <= 1:
            return False

        sentence_length = len(format_sentence)
        if sentence_length <= 2:
            return False

        for sentence in self._sentences:
            if sentence_length != len(sentence):
                continue
            for word in word_sentence.values():
                if word not in sentence:
                    continue

            # now, check rhythm
            current_rhythm_word = word_sentence[len(format_sentence) - 1]
            if current_rhythm_word not in self._reverse_rhythm_word_dict:
                continue
            current_rhythm = self._reverse_rhythm_word_dict[
                current_rhythm_word]
            sentence_word = sentence[-1]
            if sentence_word not in self._reverse_rhythm_word_dict:
                continue
            if sentence_word in already_used_rhythm_words:
                continue
            sentence_rhythm = self._reverse_rhythm_word_dict[sentence_word]

            if sentence_rhythm == current_rhythm:
                sentence_dict = {}
                for i, word in enumerate(sentence):
                    sentence_dict[i] = word
                if sentence_dict in already_used_sentences:
                    continue
                u = random.random()
                if u < 0.8:
                    continue
                return sentence_dict
        return False

    def _sub_generate(self, format_sentence, word_sentence, global_repeat_words, current_repeat_dict, logger, level=0):
        """ recursion generate single sentence"""

        sentence_length = len(format_sentence)
        word_sentence_length = len(word_sentence.keys())

        logger.info("sub_generate: level[%d]" % level)
        logger.debug("sub_generate: level[%d] sentence_len %d, word_filled_len %d" % (
            level, sentence_length, word_sentence_length))

        # all word position filled, return
        if word_sentence_length == sentence_length:
            logger.info("sub_generate: recursion finish")
            return

        # show candidate positions based on current filled positions
        candidate_positions = []
        for i in range(sentence_length):
            if i in word_sentence:
                continue
            if (i - 1) in word_sentence or (i + 1) in word_sentence:
                candidate_positions.append(i)
        logger.debug("sub_generate: level[%d] candidate_positions %s" % (
            level, str(candidate_positions)))
        if not candidate_positions:
            raise ValueError("candidate_positions len zero, illegal")
        if len(candidate_positions) == 1:  # no choice but use this
            tofill_position = candidate_positions[0]
        else:  # random choose one in choices

            # always fill rhythm_word related at end
            # if sentence_length - word_sentence_length > 1:
            #	if (sentence_length - 2) in candidate_positions:
            #		candidate_positions.remove(sentence_length - 2)

            idx = random.randint(0, len(candidate_positions) - 1)
            tofill_position = candidate_positions[idx]
        logger.debug("sub_generate: level[%d] tofill_position %d" % (
            level, tofill_position))

        up_fill_direction = (tofill_position - 1) in word_sentence
        down_fill_direction = (tofill_position + 1) in word_sentence
        both_fill_direction = up_fill_direction and down_fill_direction

        if both_fill_direction:  # consider format, choose only one, consider later
            if self._up_fill_direction(tofill_position, sentence_length, logger):
                up_fill_direction = True
                down_fill_direction = False
            else:
                up_fill_direction = False
                down_fill_direction = True

        logger.debug("sub_generate: level[%d] up_fill_direction[%d] down_fill_direction[%d]" % (
            level, up_fill_direction, down_fill_direction))

        # fill word one by one
        if up_fill_direction:
            logger.debug(
                "sub_generate: level[%d] use up_fill method" % (level))
            self._fill_word(1, tofill_position, format_sentence, word_sentence,
                            global_repeat_words, current_repeat_dict, level, logger)
        else:
            logger.debug(
                "sub_generate: level[%d] use down_fill method" % (level))
            self._fill_word(-1, tofill_position, format_sentence, word_sentence,
                            global_repeat_words, current_repeat_dict, level, logger)

        level = level + 1
        self._sub_generate(format_sentence, word_sentence,
                           global_repeat_words, current_repeat_dict, logger, level)

    def _fill_result_with_format(self, result_sentence_list):
        """ fill result with format"""

        result = ""

        delimiters = self._title_delimiter_dict[self._title]
        idx_delimiter = 0
        for result_sentence in result_sentence_list:
            result += result_sentence
            result += delimiters[idx_delimiter]
            if (idx_delimiter + 1 < len(delimiters)) and (delimiters[idx_delimiter + 1] == "|"):
                result += " | "
                idx_delimiter += 1
            idx_delimiter += 1
        return result

    def _generate(self, format_sentences, word_sentences, logger):
        """ generate poem based on important words and rhythm word"""

        result_sentence_list = []

        # generate each sentence
        # avoid words between sentences
        global_repeat_words = []
        already_used_rhythm_words = []
        already_used_sentences = []
        for i, (format_sentence, word_sentence) in enumerate(zip(format_sentences, word_sentences)):
            result_sub_sentence = ""

            # avoid too many same word in one sentence
            current_repeat_dict = {}
            for word in word_sentence.values():
                if word not in current_repeat_dict:
                    current_repeat_dict[word] = 1
                else:
                    current_repeat_dict[word] += 1

            self._show_word_sentence(
                format_sentence, word_sentence, logger, "omg origin:s %d" % (i + 1))
            u = random.random()
            if u < self._search_ratio:
                search_sentence = self._search_generate(
                    format_sentence, word_sentence, global_repeat_words, current_repeat_dict, already_used_sentences, already_used_rhythm_words, logger)
                if not search_sentence:
                    self._sub_generate(
                        format_sentence, word_sentence, global_repeat_words, current_repeat_dict, logger)
                else:
                    logger.info(
                        "[%d] use search generate for word sentence" % i)
                    word_sentence = search_sentence
                    already_used_sentences.append(search_sentence)
            else:
                self._sub_generate(format_sentence, word_sentence,
                                   global_repeat_words, current_repeat_dict, logger)
            self._show_word_sentence(
                format_sentence, word_sentence, logger, "omg final:s %d" % (i + 1))

            for word in word_sentence.values():
                result_sub_sentence += word
                global_repeat_words.append(word)
            already_used_rhythm_words.append(
                word_sentence[len(format_sentence) - 1])
            result_sentence_list.append(result_sub_sentence)

        # fill with delimiter
        if self._title not in self._title_delimiter_dict:
            print 'here'
            return u'，'.join(result_sentence_list)
        elif len(self._title_delimiter_dict[self._title]) != (len(result_sentence_list) + 1):
            print 'here2'
            raise
            return u'，'.join(result_sentence_list)
        else:
            return self._fill_result_with_format(result_sentence_list)

    def init(self, logger):

        if self._force_data_build:
            self._init_data_build(logger)
        else:
            try:
                self._load_data_build(logger)
            except Exception as e:
                logger.exception(e)
                self._init_data_build(logger)

    def check(self, input_param_dict, logger):
        """ select ci-title with supported titles"""
        return ""

        if ('title' in input_param_dict) and (input_param_dict['title'] not in self._support_titles):
            return "%s 不是候选的词牌名" % input_param_dict['title']

    def generate(self, logger):
        """ main function for poem generated"""

        # get title related sentences
        format_sentences = self._get_format_with_title(self._title, logger)
        if format_sentences < 0:
            raise ValueError("title[%s] not defined in dict" % self._title)

        # combine important words with format sentences
        word_sentences = self._combine_important_word_with_sentence(
            self._important_words, format_sentences, logger)
        self._show_word_sentences(format_sentences, word_sentences, logger)

        # decide rhythm and related words
        self._generate_rhythm(format_sentences, word_sentences, logger)
        self._show_word_sentences(format_sentences, word_sentences, logger)

        # now, generate poem
        result_sentences = self._generate(
            format_sentences, word_sentences, logger)
        logger.info("titile[%s] generate ci %s" %
                    (self._title, result_sentences))
        return result_sentences


if __name__ == '__main__':
    confpath = os.path.join(basepath, 'conf/poem.conf')
    conf = ConfigParser.RawConfigParser()
    conf.read(confpath)
    logging.basicConfig(filename=os.path.join(basepath, 'logs/chinese_poem.log'), level=logging.DEBUG,
                        format='[%(filename)s:%(lineno)s - %(funcName)s %(asctime)s;%(levelname)s] %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S'
                        )
    logger = logging.getLogger('ChinesePoem')

    generator = Generator(basepath, conf)
    title = u"浣溪沙"
    import random
    for i in range(200):
        important_word_list = [u'东风', u'何处', u'人间', u'风流', u'西风', u'归来', u'春风', u'归去', u'梅花', u'相思', u'江南', u'如今', u'回首', u'阑干', u'多少', u'千里', u'明月', u'万里', u'相逢', u'芳草', u'天涯', u'黄昏', u'当年', u'风雨', u'依旧', u'一笑', u'尊前', u'当时', u'斜阳', u'风月', u'多情', u'故人', u'不知', u'无人', u'不见', u'凄凉', u'深处', u'平生', u'匆匆', u'无限', u'春色', u'一枝', u'时节', u'流水', u'扁舟', u'功名', u'西湖', u'今日', u'憔悴', u'一点', u'何事', u'黄花', u'一片', u'十分', u'消息', u'杨柳', u'神仙', u'心事', u'长安', u'去年', u'而今', u'鸳鸯', u'芙蓉', u'不是', u'桃花', u'人生', u'今夜', u'桃李', u'断肠', u'寂寞', u'往事', u'佳人', u'悠悠', u'惟有', u'一声', u'几度', u'蓬莱', u'何时', u'一曲', u'如何', u'燕子', u'无数', u'青山', u'肠断', u'时候', u'无情', u'海棠', u'清明', u'天上', u'笙歌', u'东君', u'明年', u'少年', u'昨夜', u'千古', u'一番', u'如许', u'天气',
                               u'行人', u'今年', u'如此', u'谁知', u'梨花', u'何人', u'垂杨', u'思量', u'帘幕', u'明朝', u'分付', u'缥缈', u'江山', u'富贵', u'只有', u'秋风', u'不须', u'酒醒', u'记得', u'庭院', u'依然', u'相对', u'十年', u'十里', u'有人', u'精神', u'今宵', u'夕阳', u'风光', u'歌舞', u'携手', u'落花', u'寒食', u'殷勤', u'几时', u'不堪', u'可怜', u'一夜', u'明日', u'何妨', u'旧时', u'分明', u'夜来', u'相见', u'今朝', u'重阳', u'几许', u'秋千', u'烟雨', u'自有', u'徘徊', u'珠帘', u'无奈', u'萧萧', u'乾坤', u'谁家', u'先生', u'黄金', u'登临', u'门外', u'只恐', u'不管', u'盈盈', u'惆怅', u'等闲', u'年少', u'行云', u'楼台', u'从今', u'归路', u'赢得', u'几番', u'容易', u'不到', u'飞絮', u'夜深', u'一年', u'花枝', u'中秋', u'白发', u'一时', u'不如', u'还是', u'纷纷', u'碧云', u'十二', u'此时', u'春光', u'年华', u'风吹', u'使君', u'池塘', u'红尘', u'清风', u'瑶池', u'风露', u'长生', u'一杯', u'无处', u'飞来', u'楼上', u'何须']
        important_words = random.sample(important_word_list, 2)
        user_input_dict = dict(
            title=u"浣溪沙", important_words=important_words, force_data_build=False)
        # Init
        u = random.random() * 0.8
        # print 'ratio', u
        generator.search_ratio = u
        generator.force_data_build = user_input_dict["force_data_build"]
        generator.init(logger)

        # Generate poem
        print 'title', title
        print 'important_words', important_words[0], important_words[1]
        error_info = generator.check(user_input_dict, logger)
        if not error_info:
            generator.important_words = user_input_dict["important_words"]
            generator.title = user_input_dict["title"]

            logger.info("generate poem for title %s, with important words %s" % (
                generator.title, my_unicode(generator.important_words)))
            print generator.generate(logger)
        else:
            logger.error("dont generate poem because of %s" % error_info)
            print error_info
