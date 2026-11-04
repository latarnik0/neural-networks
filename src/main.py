import numpy as np
import random

NUMBER_OF_IRIS_PARAMETERS = 4
LEARNING_FACTOR = 0.001
NUMBER_OF_EPOCHS = 100

# perceptron
class Perceptron:
    def __init__(self, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.random.uniform(0, 1, NUMBER_OF_IRIS_PARAMETERS)
        self.bias = random.uniform(0, 1)

    # Funkcja aktywacji Heaviside'a
    def activation_function(self, vector_sum_of_neuron):
        if vector_sum_of_neuron >= 0.0:
            return 1.0
        else:
            return -1.0

    # predykcja - output neuronu
    def predict(self, dataset_inputs):
        vector_sum = np.dot(dataset_inputs, self.weights) + self.bias
        return self.activation_function(vector_sum)

    # proces uczenia
    def train(self, training_data):
        for epoch in range(self.epochs):
            errors = 0
            for features, label in training_data.items():
                inputs = np.array(features)
                prediction = self.predict(inputs)

                # jesli predykcja zła - odpowiednia aktualizacja wag
                if prediction != label:
                    update = self.learning_rate * (label - prediction)
                    self.weights += update * inputs
                    self.bias += update
                    errors += 1

            # jesli siec nie popelnia bledow - koniec uczenia
            if errors == 0:
                print(f"Learning phase ended in {epoch+1} epoch.")
                break

# metoda uczenia Adaline
class Adaline:
    def __init__(self, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.random.uniform(0, 1, NUMBER_OF_IRIS_PARAMETERS)
        self.bias = random.uniform(0, 1)

    # output przed funkcją aktywacji
    def net_input(self, dataset_inputs):
        return np.dot(dataset_inputs, self.weights) + self.bias

    # funkcja aktywacji Heaviside'a
    def activation_function(self, vector_sum_of_neuron):
        if vector_sum_of_neuron >= 0.0:
            return 1.0
        else:
            return -1.0

    # predykcja
    def predict(self, dataset_inputs):
        return self.activation_function(self.net_input(dataset_inputs))

    # proces uczenia
    def train(self, training_data):
        for epoch in range(self.epochs):
            for features, label in training_data.items():
                inputs = np.array(features)
                
                net_output = self.net_input(inputs) 
                error = label - net_output
                
                update = self.learning_rate * error
                self.weights += update * inputs
                self.bias += update

# klasyfikacja 3 klas
class OneVsRestClassifier:
    def __init__(self, classifier_class, classes, learning_rate, epochs):
        self.classes = classes
        self.classifiers = {}
        for c in classes:
            self.classifiers[c] = classifier_class(learning_rate, epochs)
            
    def train(self, training_data):
        for c in self.classes:
            binary_train_data = {
                features: (1.0 if label == c else -1.0) 
                for features, label in training_data.items()
            }
            self.classifiers[c].train(binary_train_data)
            
    def predict(self, dataset_inputs):
        best_class = None
        max_activation = -float('inf')
        
        for c, clf in self.classifiers.items():
            activation = np.dot(dataset_inputs, clf.weights) + clf.bias
            if activation > max_activation:
                max_activation = activation
                best_class = c
                
        return best_class


# liczmy klasy w pliku
def count_classes(filename):
    iris_setosa = 0
    iris_versicolor = 0

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
                
            entry_list = line.split(',')
            
            if entry_list[-1] == "Iris-setosa":
                iris_setosa += 1
            elif entry_list[-1] == "Iris-versicolor":
                iris_versicolor += 1

    return iris_setosa, iris_versicolor

# czytanie pliku z danymi - zwraca zbior uczacy, walidacyjny i testowy (70% / 15% / 15%)
def read_iris_data(filename, setosa_num_of_records, versicolor_num_of_records):
    learning_dataset = {}
    validation_dataset = {}
    test_dataset = {}
    
    iris_setosa_counter = 0
    iris_versicolor_counter = 0

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
                
            entry_list = line.split(',')
            
            features = tuple(map(float, entry_list[:4]))
            species = entry_list[-1]
            
            if species == "Iris-setosa":
                label = -1.0
                iris_setosa_counter += 1

                if iris_setosa_counter <= 0.7 * setosa_num_of_records:
                    learning_dataset[features] = label
                elif iris_setosa_counter <= 0.85 * setosa_num_of_records:
                    validation_dataset[features] = label
                else:
                    test_dataset[features] = label
                    
            elif species == "Iris-versicolor":
                label = 1.0
                iris_versicolor_counter += 1

                if iris_versicolor_counter <= 0.7 * versicolor_num_of_records:
                    learning_dataset[features] = label
                elif iris_versicolor_counter <= 0.85 * versicolor_num_of_records:
                    validation_dataset[features] = label
                else:
                    test_dataset[features] = label

    return learning_dataset, validation_dataset, test_dataset


# czytanie z pliku dla klasyfikacji 3 klas (70% / 15% / 15%)
def read_iris_data_3_classes(filename, train_limit_per_class=35, val_limit_per_class=42):
    learning_dataset = {}
    validation_dataset = {}
    test_dataset = {}
    
    counters = {"Iris-setosa": 0, "Iris-versicolor": 0, "Iris-virginica": 0}

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
                
            entry_list = line.split(',')
            features = tuple(map(float, entry_list[:4]))
            species = entry_list[-1]
            
            if species in counters:
                counters[species] += 1
                if counters[species] <= train_limit_per_class:
                    learning_dataset[features] = species
                elif counters[species] <= val_limit_per_class:
                    validation_dataset[features] = species
                else:
                    test_dataset[features] = species

    return learning_dataset, validation_dataset, test_dataset

# jakosc modelu
def evaluate_model(model, dataset):
    correct = 0
    for features, true_label in dataset.items():
        if model.predict(np.array(features)) == true_label:
            correct += 1
    accuracy = (correct / len(dataset)) * 100 if len(dataset) > 0 else 0
    return correct, len(dataset), accuracy

if __name__ == "__main__":
    filename = 'iris_data.txt'
    
    num_setosa, num_versicolor = count_classes(filename)
    train_data, val_data, test_data = read_iris_data(filename, num_setosa, num_versicolor)
    
    # ----------------- TEST PERCEPTRONU -----------------
    print("--- Perceptron learning ---")
    perceptron = Perceptron(learning_rate=LEARNING_FACTOR, epochs=NUMBER_OF_EPOCHS)
    perceptron.train(train_data)
    
    c_val, t_val, acc_val = evaluate_model(perceptron, val_data)
    c_test, t_test, acc_test = evaluate_model(perceptron, test_data)
    print(f"Perceptron - Validation accuracy: {acc_val}% ({c_val}/{t_val})")
    print(f"Perceptron - Test accuracy: {acc_test}% ({c_test}/{t_test})\n")

    # ----------------- TEST ADALINE -----------------
    print("--- Adaline learning ---")
    adaline = Adaline(learning_rate=LEARNING_FACTOR, epochs=NUMBER_OF_EPOCHS)
    adaline.train(train_data)
    
    c_val, t_val, acc_val = evaluate_model(adaline, val_data)
    c_test, t_test, acc_test = evaluate_model(adaline, test_data)
    print(f"Adaline - Validation accuracy: {acc_val}% ({c_val}/{t_val})")
    print(f"Adaline - Test accuracy: {acc_test}% ({c_test}/{t_test})\n")

    # ----------------- TEST 3 KLAS (One-vs-Rest) -----------------
    print("--- Trenowanie One-vs-Rest (3 klasy) z użyciem Adaline ---")
    classes_list = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    train_data_3, val_data_3, test_data_3 = read_iris_data_3_classes(filename)
    
    ovr_classifier = OneVsRestClassifier(
        classifier_class=Adaline, 
        classes=classes_list, 
        learning_rate=LEARNING_FACTOR, 
        epochs=NUMBER_OF_EPOCHS
    )
    ovr_classifier.train(train_data_3)
    
    c_val, t_val, acc_val = evaluate_model(ovr_classifier, val_data_3)
    c_test, t_test, acc_test = evaluate_model(ovr_classifier, test_data_3)
    print(f"One-vs-Rest - Validation accuracy: {acc_val}% ({c_val}/{t_val})")
    print(f"One-vs-Rest - Test accuracy: {acc_test}% ({c_test}/{t_test})")