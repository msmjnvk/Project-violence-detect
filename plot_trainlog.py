"""
This is to generate graph according to accuracy metrix.
"""
import csv
import matplotlib.pyplot as plt

def main(training_log):
    with open(training_log) as fin:
        reader = csv.reader(fin)
        next(reader, None)  # skip the header
        accuracies = []
        top_5_accuracies = []
        cnn_benchmark = []  # this is ridiculous
        for epoch,acc,loss,top_k_categorical_accuracy,val_acc,val_loss,val_top_k_categorical_accuracy in reader:
            accuracies.append(float(val_acc))
            top_5_accuracies.append(float(val_top_k_categorical_accuracy))
            cnn_benchmark.append(0.65) #benchmark which is set

        plt.plot(accuracies)
        plt.plot(top_5_accuracies)
        plt.plot(cnn_benchmark)
        plt.show()

if __name__ == '__main__':
    training_log = 'data/logs/mlp-training-1489455559.7089438.log'  #log path
    main(training_log)
