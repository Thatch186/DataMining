# Importar a função read_csv do seu código
from to_csv import read_csv

# Carregar o conjunto de dados iris_missing_data
dataset = read_csv("iris_missing_data.csv", features=True, label=True)

# Exibir as informações do conjunto de dados
print("Shape:", dataset.shape())
print("Features:", dataset.get_features())
print("Label:", dataset.get_label())

# Exibir o número de valores nulos em cada feature numérica
null_counts = dataset.count_nulls()
numeric_features = dataset.get_numeric_features()
for feat, null_count in zip(numeric_features, null_counts):
    print("Feature:", feat, "Null Count:", null_count)

# Calcular a média das features numéricas
mean_values = dataset.get_mean()
print("Mean Values:", mean_values)

# Calcular a mediana das features numéricas
median_values = dataset.get_median()
print("Median Values:", median_values)

# Substituir os valores nulos pelas médias
dataset.replace_nulls(method='mean')

# Exibir o número de valores nulos em cada feature numérica após a substituição
null_counts_after = dataset.count_nulls()
for feat, null_count_after in zip(numeric_features, null_counts_after):
    print("Feature:", feat, "Null Count (After):", null_count_after)
