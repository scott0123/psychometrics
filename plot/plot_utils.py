import io
import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

def read_tag_translation():
    translation = {}
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, "../data/tags/tag_translation.csv"), "r") as f:
        data = [line for line in f]
    for line in data:
        elements = line.strip().split(",")
        translation[elements[0]] = elements[1]
    return translation

def plot_topic_capacity(name, topics, capacity):
    '''
    name: name of student
    topics: list of strings
    capacity: list of numbers
    '''
    assert(len(topics) == len(capacity))
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["figure.dpi"] = 150
    plt.bar(topics, capacity, width=0.5, edgecolor="k", linewidth=1)
    plt.xlabel("Topics")
    plt.ylabel("Capacities")
    #plt.xticks(rotation=30, ha="center")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_topic_capacity_horizontal(name, topics, capacities):
    '''
    name: name of student
    topics: list of strings
    capacities: list of numbers
    '''
    assert(len(topics) == len(capacities))
    #capacities -= np.min(capacities) - 0.1
    sns.set_theme()
    f, ax = plt.subplots(figsize=(6, 10))
    sns.set_color_codes("pastel")
    df = pd.DataFrame({"Topics": topics, "Capacities": capacities})
    sns.barplot(x="Capacities", y="Topics", data=df, orient="h", color="b")
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    minus = min(min(capacities), -0.1)
    plus = max(max(capacities), 0.1)
    plt.xticks([minus, 0, plus], ["-", "0", "+"])
    plt.title(f"Student: {name}")
    plt.tight_layout()
    plt.show()

def topic_capacity_image(name, topics, capacities):
    '''
    name: name of student
    topics: list of strings
    capacities: list of numbers
    '''
    assert(len(topics) == len(capacities))
    #capacities -= np.min(capacities) - 0.1
    sns.set_theme()
    f, ax = plt.subplots(figsize=(6, 10))
    sns.set_color_codes("pastel")
    df = pd.DataFrame({"Topics": topics, "Capacities": capacities})
    sns.barplot(x="Capacities", y="Topics", data=df, orient="h", color="b")
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    minus = min(min(capacities), -0.1)
    plus = max(max(capacities), 0.1)
    plt.xticks([minus, 0, plus], ["-", "0", "+"])
    plt.title(f"Student: {name}")
    plt.tight_layout()
    plt_buffer = io.BytesIO()
    plt.savefig(plt_buffer,  format="png")
    plt_buffer.seek(0)
    image = plt_buffer.read()
    return image

# Split it into three plots
def plot_topic_capacity_horizontal3(name, topics, capacities):
    '''
    name: name of student
    topics: list of strings
    capacities: list of numbers
    '''
    assert(len(topics) == len(capacities))
    tag_translation = read_tag_translation()
    # collect
    topics_by_group = {"bkgrd": [], "meta": [], "ml": []}
    capacities_by_group = {"bkgrd": [], "meta": [], "ml": []}
    for i in range(len(topics)):
        if topics[i].startswith("bkgrd"):
            topics_by_group["bkgrd"].append(topics[i])
            capacities_by_group["bkgrd"].append(capacities[i])
        elif topics[i].startswith("meta"):
            topics_by_group["meta"].append(topics[i])
            capacities_by_group["meta"].append(capacities[i])
        else:
            topics_by_group["ml"].append(topics[i])
            capacities_by_group["ml"].append(capacities[i])
    for key in ["bkgrd", "ml", "meta"]:
        topics = topics_by_group[key]
        capacities = capacities_by_group[key]
        sns.set_theme()
        fig_height = min(10, len(topics) // 2 + 1)
        f, ax = plt.subplots(figsize=(6, fig_height))
        sns.set_color_codes("pastel")
        translated_topics = [tag_translation[tag] for tag in topics]
        df = pd.DataFrame({"Topics": translated_topics, "Capacities": capacities})
        sns.barplot(x="Capacities", y="Topics", data=df, orient="h", color="b")
        ax.yaxis.grid(True)
        ax.xaxis.grid(True)
        minus = min(min(capacities), -0.1)
        plus = max(max(capacities), 0.1)
        plt.xticks([minus, 0, plus], ["-", "0", "+"])
        plt.title(f"Student: {name}")
        plt.tight_layout()
        plt.show()

# Split it into three plots
def topic_capacity_horizontal3(name, topics, capacities):
    '''
    name: name of student
    topics: list of strings
    capacities: list of numbers
    '''
    assert(len(topics) == len(capacities))
    tag_translation = read_tag_translation()
    # collect
    topics_by_group = {"bkgrd": [], "meta": [], "ml": []}
    capacities_by_group = {"bkgrd": [], "meta": [], "ml": []}
    for i in range(len(topics)):
        if topics[i].startswith("bkgrd"):
            topics_by_group["bkgrd"].append(topics[i])
            capacities_by_group["bkgrd"].append(capacities[i])
        elif topics[i].startswith("meta"):
            topics_by_group["meta"].append(topics[i])
            capacities_by_group["meta"].append(capacities[i])
        else:
            topics_by_group["ml"].append(topics[i])
            capacities_by_group["ml"].append(capacities[i])
    plots = []
    for key in ["bkgrd", "ml", "meta"]:
        topics = topics_by_group[key]
        capacities = capacities_by_group[key]
        sns.set_theme()
        fig_height = min(10, len(topics) // 2 + 1)
        f, ax = plt.subplots(figsize=(6, fig_height))
        sns.set_color_codes("pastel")
        translated_topics = [tag_translation[tag] for tag in topics]
        df = pd.DataFrame({"Topics": translated_topics, "Capacities": capacities})
        sns.barplot(x="Capacities", y="Topics", data=df, orient="h", color="b")
        ax.yaxis.grid(True)
        ax.xaxis.grid(True)
        minus = min(min(capacities), -0.1)
        plus = max(max(capacities), 0.1)
        plt.xticks([minus, 0, plus], ["-", "0", "+"])
        plt.title(f"Student: {name}")
        plt.tight_layout()
        plt_buffer = io.BytesIO()
        plt.savefig(plt_buffer,  format="png")
        plt_buffer.seek(0)
        plots.append(plt_buffer.read())
    return plots[0], plots[1], plots[2]

def plot_relation_heatmap(relation_matrix, bg_tags, ml_tags):
    tag_translation = read_tag_translation()
    translated_bg_tags = [tag_translation[tag] for tag in bg_tags]
    translated_ml_tags = [tag_translation[tag] for tag in ml_tags]
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(relation_matrix.T, square=True, cmap=sns.diverging_palette(220, 20, as_cmap=True),
        center=0, xticklabels=translated_bg_tags, yticklabels=translated_ml_tags)
    plt.tight_layout()
    plt.show()

def compare_distributions(values1, name1, values2, name2):
    df1 = pd.DataFrame({"Values":values1, "Type":name1})
    df2 = pd.DataFrame({"Values":values2, "Type":name2})
    df = pd.concat([df1, df2])
    sns.displot(df, x="Values", hue="Type", kind="kde")
    plt.show()
