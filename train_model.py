'''
Model training script
First draft by Scott Liu @ 2020 Mar
Second revision by Scott Liu @ 2020 Sep
'''

# Import statements
import os
import csv
import numpy as np
from metrics import get_f1
from plot.plot_utils import plot_topic_capacity_horizontal3
from architecture import ExperimentalAdditiveFactorModel

NAMES = ["HW1", "HW2", "HW3", "HW4", "HW5", "HW6", "HW7", "HW8", "HW9"]
SCORE_FILES = ["encrypted.HW1_written__scores.csv", "encrypted.HW2_written__scores.csv", "encrypted.HW3_written__scores.csv", "encrypted.HW4_written__scores.csv", "encrypted.HW5_written__scores.csv", "encrypted.HW6_written__scores.csv", "encrypted.HW7_written__scores.csv", "encrypted.HW8_written__scores.csv", "encrypted.HW9_written__scores.csv"]
TAG_FILES = ["Tags - Homework_1.tsv", "Tags - Homework_2.tsv", "Tags - Homework_3.tsv", "Tags - Homework_4.tsv", "Tags - Homework_5.tsv", "Tags - Homework_6.tsv", "Tags - Homework_7.tsv", "Tags - Homework_8.tsv", "Tags - Homework_9.tsv"]

NAME_INDEX = 0
EMAIL_INDEX = 3
QUESTION_START_INDEX = 11

LOAD_ONLY = False

class StringToIndex():
    '''
    Useful helper class to keep track of string indices
    '''
    def __init__(self):
        self.map = {}
        self.index = 0

    def get(self, string):
        return self.map[string]

    def register(self, string):
        if string in self.map:
            return
        self.map[string] = self.index
        self.index += 1

    def count(self):
        return self.index

    def get_all(self):
        return list(self.map.keys())

def form_Q_matrix(tag_question_map, question_index_map, tag_index_map):
    '''
    tag_question_map is a dictionary that maps tags to lists of questions
    question_index_map is a dictionary that maps questions to unique indicies
    tag_index_map is a dictionary that maps tags to unique indicies
    The output Q matrix should have a shape: [num_tags x num_questions]
    '''
    Q = np.zeros(shape=(tag_index_map.count(), question_index_map.count()))
    for tag, questions in tag_question_map.items():
        row = tag_index_map.get(tag)
        for question in questions:
            col = question_index_map.get(question)
            Q[row][col] = 1
    return Q.astype(np.float32)

def make_tag_question_map(name, tag_list, question_index_map, tag_index_map, num_questions):
    '''
    Make a map from question to tags:
    It should return a dictionary in the form:
    {
        "tag_1": ["name_Q1", "name_Q2"],
        ...
    }
    It also updates the question_index_map and the tag_index_map
    '''
    # cross-check with the score file to make sure the lengths are the same
    if len(tag_list) != num_questions:
        raise RuntimeError(f"lengths of scores not equal to tags for {name}, consider double checking\n\
            length of tag_list: {len(tag_list)}, number of questions: {num_questions}")
    # at this point we've verified that the files are consistent with each other
    tag_question_map = {}
    for idx, tags in enumerate(tag_list):
        question = name + f"_Q{idx}"
        question_index_map.register(question)
        for tag in tags:
            tag_index_map.register(tag)
            if tag not in tag_question_map:
                tag_question_map[tag] = []
            tag_question_map[tag].append(question)
    return tag_question_map

def merge_tag_question_maps(tag_question_maps):
    '''
    Merges a list of tag_questions_maps into a single tag_questions_map
    '''
    tag_question_map = {}
    for tqm in tag_question_maps:
        for tag, questions in tqm.items():
            if tag not in tag_question_map:
                tag_question_map[tag] = []
            tag_question_map[tag].extend(questions)
    return tag_question_map


def shuffle_together(*args):
    '''
    Shuffle multiple lists in the same way
    '''
    from random import shuffle
    z = list(zip(*args))
    shuffle(z)
    return zip(*z)

def strip_points(s):
    start = s.find("(")
    end = s.find(" pts)")
    return float(s[start+1:end])

def clean_scores_and_tags(score_list, max_scores, tag_list):
    # find the indices where max_score is 0
    bad_indices = [idx for idx, val in enumerate(max_scores) if val == 0]
    new_score_list = []
    new_max_scores = []
    new_tag_list = []
    for scores in score_list:
        new_score_list.append([scores[idx] for idx in range(len(max_scores)) if idx not in bad_indices])
    new_max_scores = [max_scores[idx] for idx in range(len(max_scores)) if idx not in bad_indices]
    new_tag_list = [tag_list[idx] for idx in range(len(max_scores)) if idx not in bad_indices]
    return new_score_list, new_max_scores, new_tag_list

def read_scores(filename):
    user_list = []
    score_list = []
    max_scores = []
    with open(f"data/scores/{filename}", "r") as f:
        data = [line for line in f]
    header, scores = data[0], data[1:]
    # deal with header
    header_items = next(csv.reader([header.strip()]))
    max_scores = list(map(strip_points, header_items[QUESTION_START_INDEX:]))
    # deal with the rest
    for line in scores:
        items = line.strip().split(",")
        email = items[EMAIL_INDEX]
        try:
            scores = [float(x) for x in items[QUESTION_START_INDEX:]]
        except:
            continue # this is also possible for no submission
        if len(scores) == 0: # some entries are empty
            continue
        user_list.append(email)
        score_list.append(scores)
    return user_list, score_list, max_scores


def get_email_to_name():
    email_to_name = {}
    for filename in SCORE_FILES:
        with open(f"data/scores/{filename}", "r") as f:
            data = [line for line in f]
        header, scores = data[0], data[1:]
        for line in scores:
            items = line.strip().split(",")
            name = items[NAME_INDEX]
            email = items[EMAIL_INDEX]
            email_to_name[email] = name
    return email_to_name

def read_tags(filename):
    tag_list = []
    with open(f"data/tags/{filename}", "r") as f:
        data = [line for line in f]
    for line in data[1:]:
        groups = line.strip().split('\t')
        bg_tags = [x.strip() for x in groups[1].split(',')]
        ml_tags = [x.strip() for x in groups[2].split(',')]
        meta_tags = [x.strip() for x in groups[3].split(',')]
        concat = bg_tags + ml_tags + meta_tags
        concat_not_empty = [x for x in concat if x]
        tag_list.append(concat_not_empty)
    return tag_list

def read_tag_translation():
    translation = {}
    with open("data/tags/tag_translation.csv", "r") as f:
        data = [line for line in f]
    for line in data:
        elements = line.strip().split(",")
        translation[elements[0]] = elements[1]
    return translation

def main():
    question_index_map = StringToIndex() # custom class
    tag_index_map = StringToIndex() # custom class
    user_index_map = StringToIndex() # custom class
    tag_question_maps = []
    user_list_bundle, score_list_bundle, max_scores_bundle = [], [], []
    for name, score_file, tag_file in zip(NAMES, SCORE_FILES, TAG_FILES):
        print("Reading {}".format(name))
        user_list, score_list, max_scores = read_scores(score_file)
        tag_list = read_tags(tag_file)
        score_list, max_scores, tag_list = clean_scores_and_tags(score_list, max_scores, tag_list)
        user_list_bundle.append(user_list)
        score_list_bundle.append(score_list)
        max_scores_bundle.append(max_scores)
        tag_question_maps.append(make_tag_question_map(name, tag_list, question_index_map, tag_index_map, len(max_scores)))
    final_tag_question_map = merge_tag_question_maps(tag_question_maps)
    Q = form_Q_matrix(final_tag_question_map, question_index_map, tag_index_map)

    # construct the data for model training
    users, questions, scores = [], [], []
    question_starting_index = 0
    for idx, name in enumerate(NAMES):
        user_list = user_list_bundle[idx]
        score_list = score_list_bundle[idx]
        max_scores = max_scores_bundle[idx]
        for i in range(len(user_list)):
            user = user_list[i]
            scores_row = score_list[i]
            user_index_map.register(user)
            user_index = user_index_map.get(user)
            for j in range(len(scores_row)):
                normalized_score = scores_row[j] / max_scores[j]
                users.append(user_index)
                questions.append(j + question_starting_index)
                scores.append(normalized_score)
        # we have to update the question_starting_index by an offset equal to the number of questions
        question_starting_index += len(score_list[0])

    # shuffle and split
    users, questions, scores = shuffle_together(users, questions, scores)
    val_split = int(len(users) * 0.2)
    test_split = int(len(users) * 0.1)
    train_users, val_users, test_users = users[:-(val_split+test_split)], users[-(val_split+test_split):-test_split], users[-test_split:]
    train_questions, val_questions, test_questions = questions[:-(val_split+test_split)], questions[-(val_split+test_split):-test_split], questions[-test_split:]
    train_scores, val_scores, test_scores = scores[:-(val_split+test_split)], scores[-(val_split+test_split):-test_split], scores[-test_split:]
    print("Train/Val/Test counts: {}/{}/{}".format(len(train_users), len(val_users), len(test_users)))

    all_users = user_index_map.get_all()
    all_questions = question_index_map.get_all()
    all_tags = tag_index_map.get_all()

    n_users = len(user_index_map.get_all())
    n_questions = len(question_index_map.get_all())

    print("##### MODEL: EAFM ####")
    n_KCs = Q.shape[0]
    if not LOAD_ONLY:
        model3 = ExperimentalAdditiveFactorModel(n_users, n_questions, Q)
        model3.auto_fit(train_users, train_questions, train_scores,
                        val_users, val_questions, val_scores, lr=1e-3, reg=1e-7, patience=50)
        model3.save("model/eafm.bin")
    else:
        model3 = ExperimentalAdditiveFactorModel.load("model/eafm.bin")
    acc = model3.predict_and_eval(test_users, test_questions, test_scores)
    print(f"Test accuracy {acc}")
    preds = model3.predict(test_users, test_questions)
    f1, precision, recall = get_f1(np.asarray(test_scores), preds)
    print("F1 score: {}\tprecision: {}\trecall: {}".format(f1, precision, recall))
    # student-topic vector: alpha
    alpha = model3.alpha.cpu().detach().numpy()
    for i in range(3):
        import random
        idx = random.randint(0, len(all_users) - 1)
        print("random student:", all_users[idx])
        topic_capacities = alpha[idx]
        print("topic capacities of this student", topic_capacities)
        name = all_users[idx]
        hashing = True
        if hashing:
            import hashlib
            m = hashlib.md5()
            m.update(name.encode())
            name = m.hexdigest()[:6].upper() + " (hashed)"
        plot_topic_capacity_horizontal3(name, all_tags, topic_capacities.tolist())
    exit()



if __name__ == "__main__":
    main()
