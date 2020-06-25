from NBC import *

# Lay du lieu
dataset=load_data('./data.csv')

# Chia he so
splitRatio=0.8 # 80% data la test
# Train, test split
trainingSet, testSet = split_data(dataset, splitRatio)
print('Data size %d \nTraining size=%d \nTest size=%d' %(len(dataset), len(trainingSet), len(testSet)))

# prepare model
summaries = summarize_by_class(trainingSet)
get_data_label(trainingSet)

# test model
predictions = get_predictions(summaries, testSet)
accuracy = get_accuracy(testSet, predictions)
print('Do chinh xac: %f' %(accuracy))

# Ket qua
print('Du lieu test')
print(testSet[-1][0:-1])

print('\nKet qua du doan:')
print(predictions[-1])

print('\nKet qua thuc')
print(testSet[-1][-1])