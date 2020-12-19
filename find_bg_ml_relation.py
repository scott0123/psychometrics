'''
Find the relationship between the background topics and machine learning topics
First draft by Scott Liu @ 2020 Oct
'''

# Import statements
import os
import csv
import numpy as np
from metrics import get_f1
from plot.plot_utils import topic_capacity_image, plot_relation_heatmap
from architecture import ExperimentalAdditiveFactorModel, BackgroundRelationModel
from train_model import NAMES, SCORE_FILES, TAG_FILES, EMAIL_INDEX, QUESTION_START_INDEX
from train_model import StringToIndex, form_Q_matrix, make_tag_question_map, merge_tag_question_maps
from train_model import shuffle_together, strip_points, clean_scores_and_tags, read_scores, read_tags

def main():

    question_index_map = StringToIndex() # custom class
    tag_index_map = StringToIndex() # custom class
    user_index_map = StringToIndex() # custom class
    tag_question_maps = []
    user_list_bundle, score_list_bundle, max_scores_bundle = [], [], []
    for name, score_file, tag_file in zip(NAMES, SCORE_FILES, TAG_FILES):
        user_list, score_list, max_scores = read_scores(score_file)
        tag_list = read_tags(tag_file)
        score_list, max_scores, tag_list = clean_scores_and_tags(score_list, max_scores, tag_list)
        user_list_bundle.append(user_list)
        score_list_bundle.append(score_list)
        max_scores_bundle.append(max_scores)
        tag_question_maps.append(make_tag_question_map(name, tag_list, question_index_map, tag_index_map, len(max_scores)))
    final_tag_question_map = merge_tag_question_maps(tag_question_maps)

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

    all_users = user_index_map.get_all()
    all_questions = question_index_map.get_all()
    all_tags = tag_index_map.get_all()

    n_users = len(user_index_map.get_all())
    n_questions = len(question_index_map.get_all())

    print("##### Loading model: EAFM ####")
    model3 = ExperimentalAdditiveFactorModel.load("model/eafm.bin")

    # student-topic vector: alpha
    alpha = model3.alpha.cpu().detach().numpy()
    if alpha.shape[0] != len(all_users) or alpha.shape[1] != len(all_tags):
        print("Student capacity parameter shape: {}".format(alpha.shape))
        print("Number of unique users in data: {}".format(len(all_users)))
        print("Number of unique tags in data: {}".format(len(all_tags)))
        raise RuntimeError("Dimension mismatch: please make sure the model is trained with the same data")

    # find alpha_bg and alpha_ml
    alpha_bg_list = []
    alpha_ml_list = []
    bg_tags = []
    ml_tags = []
    for idx, tag in enumerate(all_tags):
        if tag.startswith("bkgrd"):
            alpha_bg_list.append(alpha[:,idx])
            bg_tags.append(tag)
        elif tag.startswith("ml"):
            alpha_ml_list.append(alpha[:,idx])
            ml_tags.append(tag)
    alpha_bg = np.vstack(alpha_bg_list).T
    alpha_ml = np.vstack(alpha_ml_list).T

    # split train / val / test
    print("alpha_bg.shape:", alpha_bg.shape)
    print("alpha_ml.shape:", alpha_ml.shape)
    train_alpha_bg, val_alpha_bg, test_alpha_bg = alpha_bg[:-140], alpha_bg[-140:-70], alpha_bg[-70:]
    train_alpha_ml, val_alpha_ml, test_alpha_ml = alpha_ml[:-140], alpha_ml[-140:-70], alpha_ml[-70:]

    # train or load
    load = False
    if not load:
        model = BackgroundRelationModel(alpha_bg.shape[1], alpha_ml.shape[1])
        model.auto_fit(train_alpha_bg, train_alpha_ml, val_alpha_bg, val_alpha_ml,
                        lr=1e-3, reg=1e-3, patience=50)
        model.save("model/bg.bin")
    else:
        model = BackgroundRelationModel.load("model/bg.bin")

    # visualize relation
    relation_matrix = model.W.cpu().detach().numpy()
    plot_relation_heatmap(relation_matrix, bg_tags, ml_tags)

if __name__ == "__main__":
    main()
