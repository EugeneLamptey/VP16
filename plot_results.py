import matplotlib.pyplot as plt
def plot(result):
    #Creating barchart plot
    metrics = ['Accuracy', 'Recall', 'Specificity', 'F1-score']
    heights = [result['accuracy'], result['1']['recall'], result['0']['recall'], result['macro avg']['f1-score']]
    heights


    plt.xlabel('metrics')
    plt.ylabel('Scores')
    plt.title('Model performance before preprocessing')

    # Plotting the bar chart
    bars = plt.bar(metrics, heights, width=0.5)

    #Adding annotations to each bar
    for bar, height in zip(bars, heights):
        plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, height, f'{height:.2f}', ha='center', va='bottom')
    plt.show()