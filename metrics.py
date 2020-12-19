import numpy as np


def get_f1(y_true, y_pred):
    y_pred = (y_pred >= 0.5).astype(int)
    TP = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    FP = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    TN = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    FN = np.sum(np.logical_and(y_pred == 0, y_true == 1))
    f1 = 2 * TP / (2 * TP + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return f1, precision, recall


def main():
    true = np.random.randint(0, 2, size=(50))
    pred = np.random.randint(0, 2, size=(50))
    f1, precision, recall = get_f1(true, pred)
    print("true", true)
    print("pred", pred)
    print("f1", f1)
    print("precision", precision)
    print("recall", recall)


if __name__ == "__main__":
    main()
